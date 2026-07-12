"""Shared machinery behind the model engines.

The engines agree on how they assemble the `Outcomes` panel and, for the
stochastic engines, on how the runner drives their per-iteration randomness.
That skeleton lives here: the panel finalizer, the population sampler, and a
base class each for the deterministic and stochastic engines. Each engine
subclasses one and fills in a per-run hook.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from heormodel.models._accrual import aggregate
from heormodel.models._interventions import (
    InterventionSpec,
    comparator_of,
    merge_decision_levers,
    normalize_interventions,
)
from heormodel.models.outcomes import INTERVENTION_LEVEL, ITERATION_LEVEL, Outcomes
from heormodel.models.protocol import EngineResult

if TYPE_CHECKING:
    from heormodel.run.seeds import SeedManager

#: How an engine accepts its population: a size, a sampler, or ``None``.
PopulationSpec = int | Callable[[np.random.Generator, int], pd.DataFrame] | None

_COST_COL = "cost"


def iteration_key(label: Any) -> int:
    """Turn an iteration label into a stable integer seed key."""
    try:
        return int(label)
    except (TypeError, ValueError):
        digest = hashlib.blake2b(repr(label).encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big")


def finalize_outcomes(
    data: pd.DataFrame,
    interventions: list[str],
    index: pd.Index,
    effect: str,
    comparator: str | None,
) -> Outcomes:
    """Reindex per-arm rows onto the full balanced panel and wrap as `Outcomes`."""
    full_index = pd.MultiIndex.from_product(
        [interventions, index], names=[INTERVENTION_LEVEL, ITERATION_LEVEL]
    )
    return Outcomes(data.reindex(full_index), effect=effect, comparator=comparator)


class PopulationSampler:
    """Resolve the ``population``/``n_individuals`` arguments and sample from them.

    Holds the size and an optional attribute sampler ``fn(rng, n) -> DataFrame``;
    ``sample`` returns a featureless frame or validates the sampler's output.
    """

    def __init__(self, population: PopulationSpec, n_individuals: int) -> None:
        if isinstance(population, bool):  # bool is an int subclass; reject it explicitly
            raise TypeError("population must be an int, a callable, or None.")
        if isinstance(population, int):
            self.n = population
            self._fn: Callable[..., pd.DataFrame] | None = None
        elif population is None:
            self.n = n_individuals
            self._fn = None
        elif callable(population):
            self.n = n_individuals
            self._fn = population
        else:
            raise TypeError("population must be an int, a callable, or None.")
        if self.n <= 0:
            raise ValueError("Population size must be positive.")

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        if self._fn is None:
            return pd.DataFrame(index=pd.RangeIndex(self.n))
        attrs = self._fn(rng, self.n)
        if not isinstance(attrs, pd.DataFrame):
            raise TypeError("population sampler must return a DataFrame.")
        if len(attrs) != self.n:
            raise ValueError(f"population sampler returned {len(attrs)} rows, expected {self.n}.")
        return attrs.reset_index(drop=True)


class _EngineBase:
    """State every engine configures the same way."""

    _interventions: dict[str, dict[str, Any]]
    _comparator: str | None
    _discount_rate: float
    _effect: str

    def _configure(
        self, interventions: InterventionSpec, discount_rate: float, effect: str
    ) -> None:
        self._interventions = normalize_interventions(interventions)
        self._comparator = comparator_of(interventions)
        self._discount_rate = float(discount_rate)
        self._effect = effect


class DeterministicEngine(_EngineBase):
    """Shared ``evaluate`` loop for the deterministic engines.

    A subclass calls ``_configure`` and implements ``_payoff(params,
    intervention)`` returning the discounted ``(cost, effect)`` for one arm.
    """

    def _payoff(self, params: pd.Series, intervention: str) -> tuple[float, float]:
        raise NotImplementedError

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate every intervention on every draw, returning `Outcomes` on ``draws.index``."""
        if draws.empty:
            raise ValueError("draws is empty.")
        costs: list[float] = []
        effects: list[float] = []
        keys: list[tuple[str, object]] = []
        for label, (_, raw_params) in zip(draws.index, draws.iterrows(), strict=True):
            for name, decision_levers in self._interventions.items():
                params = merge_decision_levers(raw_params, decision_levers)
                cost, effect = self._payoff(params, name)
                costs.append(cost)
                effects.append(effect)
                keys.append((name, label))
        data = pd.DataFrame(
            {_COST_COL: costs, self._effect: effects},
            index=pd.MultiIndex.from_tuples(keys, names=[INTERVENTION_LEVEL, ITERATION_LEVEL]),
        )
        return finalize_outcomes(
            data, list(self._interventions), draws.index, self._effect, self._comparator
        )


class StreamingEngine(_EngineBase):
    """Shared per-iteration streaming loop for the stochastic engines.

    The base owns the loop: it validates ``collect``, spawns the population and
    simulation streams per iteration (common random numbers across arms by
    default, an independent stream per arm when ``_independent_streams`` is set),
    averages each arm to one `Outcomes` row, and assembles the optional logs.

    A subclass calls ``_configure``, sets ``_independent_streams``,
    ``_population`` (a `PopulationSampler`) and its size alias ``_n``, and
    implements ``_run_arm``; it may override ``_simulation_randomness`` to turn
    each arm's `SeedSequence` into what its kernel consumes (the default returns
    the sequence itself).
    """

    _independent_streams: bool
    _population: PopulationSampler

    def _simulation_randomness(self, source: np.random.SeedSequence) -> Any:
        """Per-arm simulation randomness derived from ``source`` (default: itself)."""
        return source

    def _run_arm(
        self,
        params: pd.Series,
        intervention: str,
        attrs: pd.DataFrame,
        sim_randomness: Any,
        *,
        collect_events: bool,
    ) -> tuple[dict[str, NDArray[np.float64]], pd.DataFrame | None]:
        """One (iteration, intervention): per-individual accruals and the event log."""
        raise NotImplementedError

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate every draw from a fixed default stream and return `Outcomes`.

        The narrow `heormodel.models.ModelEngine` entry point: it seeds each
        iteration from a fixed default stream, so a direct call is reproducible.
        Run through `heormodel.run.run_psa` to choose the seed, run in parallel,
        and collect the event or individual logs.
        """
        from heormodel.run.seeds import SeedManager

        return self.evaluate_streamed(draws, streams=SeedManager(0)).outcomes

    def evaluate_streamed(
        self, draws: pd.DataFrame, *, streams: SeedManager, collect: str | None = None
    ) -> EngineResult:
        """Simulate every draw under ``streams``, collecting the ``collect`` log.

        Each iteration draws a stream keyed by its index, so results do not depend
        on how the run is chunked across workers. ``collect`` is ``None`` (outcomes
        only), ``"events"`` (the event log), or ``"individuals"`` (per-individual
        cost and effect); the matching field of the returned `EngineResult` is set.
        """
        if collect not in (None, "events", "individuals"):
            raise ValueError(
                f"collect must be None, 'events', or 'individuals', got {collect!r}."
            )
        if draws.empty:
            raise ValueError("draws is empty.")
        collect_events = collect == "events"
        collect_individuals = collect == "individuals"
        interventions = list(self._interventions)
        rows: list[pd.DataFrame] = []
        logs: list[pd.DataFrame] = []
        for label, (_, raw_params) in zip(draws.index, draws.iterrows(), strict=True):
            for name, params, attrs, sim_randomness in self._arms(streams, label, raw_params):
                accruals, events = self._run_arm(
                    params, name, attrs, sim_randomness, collect_events=collect_events
                )
                rows.append(aggregate(accruals, name, label))
                if collect_events:
                    assert events is not None
                    events.insert(0, ITERATION_LEVEL, label)
                    events.insert(0, INTERVENTION_LEVEL, name)
                    logs.append(events)
                elif collect_individuals:
                    logs.append(self._individual_frame(accruals, name, label))
        data = pd.concat(rows)
        fill = {col: 0.0 for col in data.columns if col not in (_COST_COL, self._effect)}
        if fill:  # unreported disaggregated components read as zero
            data = data.fillna(value=fill)
        outcomes = finalize_outcomes(
            data, interventions, draws.index, self._effect, self._comparator
        )
        combined = pd.concat(logs, ignore_index=True) if logs else None
        return EngineResult(
            outcomes,
            events=combined if collect_events else None,
            individuals=combined if collect_individuals else None,
        )

    def _arms(
        self, streams: SeedManager, label: object, raw_params: pd.Series
    ) -> Iterator[tuple[str, pd.Series, pd.DataFrame, Any]]:
        """Yield ``(intervention, params, attrs, sim_randomness)`` per arm, CRN unless split."""
        iter_seq = streams.child_sequence(iteration_key(label))
        n_arms = len(self._interventions)
        if self._independent_streams:
            sub = iter_seq.spawn(2 * n_arms)
            for j, (name, decision_levers) in enumerate(self._interventions.items()):
                attrs = self._population.sample(np.random.default_rng(sub[2 * j]))
                yield (
                    name,
                    merge_decision_levers(raw_params, decision_levers),
                    attrs,
                    self._simulation_randomness(sub[2 * j + 1]),
                )
        else:
            pop_seq, shared_source = iter_seq.spawn(2)
            shared_attrs = self._population.sample(np.random.default_rng(pop_seq))
            shared_randomness = self._simulation_randomness(shared_source)
            for name, decision_levers in self._interventions.items():
                yield (
                    name,
                    merge_decision_levers(raw_params, decision_levers),
                    shared_attrs.copy(),
                    shared_randomness,
                )

    def _individual_frame(
        self, accruals: dict[str, NDArray[np.float64]], intervention: str, label: object
    ) -> pd.DataFrame:
        frame = pd.DataFrame(accruals)
        frame.insert(0, "individual", np.arange(len(frame)))
        frame.insert(0, ITERATION_LEVEL, label)
        frame.insert(0, INTERVENTION_LEVEL, intervention)
        return frame
