"""Individual-level (microsimulation) engine.

`MicrosimModel` simulates an individual-level population per PSA iteration and
emits the standard `Outcomes` schema. One class covers two clocks:

- ``clock="discrete"`` (default) advances every individual on a fixed cycle
  grid, sampling state transitions from per-cycle probabilities. History
  dependence enters through attribute columns the engine maintains (``cycle``
  and ``time_in_state``). Supply a ``transition`` callback.
- ``clock="continuous"`` races competing time-to-event samplers, takes the
  earliest, and accrues continuously between events. No cycle grid; the horizon
  truncates. Supply a ``hazards`` callback.

The engine configures once and evaluates on draws: the constructor takes the
model structure, and `evaluate` takes only the parameter draw matrix, returning
`Outcomes` indexed by ``draws.index``. Randomness comes from a `SeedManager`
injected at construction; each iteration draws a stream keyed by its index, so
iteration ``i`` is reproducible in isolation and results do not depend on how a
run is chunked across workers. Cost and utility accrual, discounting, and
aggregation live in the shared `heval.models._accrual` module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from heval.models._accrual import accrue, aggregate, integrate_flow
from heval.models.outcomes import ITERATION_LEVEL, STRATEGY_LEVEL, Outcomes
from heval.run.seeds import SeedManager

PopulationSpec = int | Callable[[np.random.Generator, int], pd.DataFrame] | None
StrategySpec = Mapping[str, Mapping[str, Any]]


def _iteration_key(label: Any) -> int:
    """Turn an iteration label into a stable integer seed key."""
    try:
        return int(label)
    except (TypeError, ValueError):
        digest = hashlib.blake2b(repr(label).encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big")


class _MicrosimBase:
    """Shared configuration and per-iteration streaming for the microsim clocks.

    Not an engine API: `MicrosimModel` implements ``_simulate`` per clock. The
    only surface either clock shares with the outside world is the `Outcomes`
    contract.
    """

    def __init__(
        self,
        *,
        states: tuple[str, ...],
        payoffs: Callable[..., tuple[NDArray[np.float64], NDArray[np.float64]]],
        population: PopulationSpec,
        strategies: StrategySpec,
        seed_manager: SeedManager,
        n_individuals: int,
        initial_state: str | int,
        discount_rate: float,
        effect: str,
        independent_streams: bool,
    ) -> None:
        if len(states) < 2:
            raise ValueError("Provide at least two states.")
        if not strategies:
            raise ValueError("Provide at least one strategy.")
        self._states = tuple(states)
        self._payoffs = payoffs
        self._strategies = {name: dict(overrides) for name, overrides in strategies.items()}
        self._seed_manager = seed_manager
        self._discount_rate = float(discount_rate)
        self._effect = effect
        self._independent_streams = bool(independent_streams)
        if isinstance(population, bool):  # bool is an int subclass; reject it explicitly
            raise TypeError("population must be an int, a callable, or None.")
        if isinstance(population, int):
            self._n = population
            self._population_fn: Callable[..., pd.DataFrame] | None = None
        elif population is None:
            self._n = n_individuals
            self._population_fn = None
        elif callable(population):
            self._n = n_individuals
            self._population_fn = population
        else:
            raise TypeError("population must be an int, a callable, or None.")
        if self._n <= 0:
            raise ValueError("Population size must be positive.")
        self._initial_index = self._state_index(initial_state)

    def _state_index(self, state: str | int) -> int:
        if isinstance(state, str):
            if state not in self._states:
                raise ValueError(f"Unknown state {state!r}; states are {self._states}.")
            return self._states.index(state)
        idx = int(state)
        if not 0 <= idx < len(self._states):
            raise ValueError(f"initial_state index {idx} out of range.")
        return idx

    def _sample_population(self, rng: np.random.Generator) -> pd.DataFrame:
        if self._population_fn is None:
            return pd.DataFrame(index=pd.RangeIndex(self._n))
        attrs = self._population_fn(rng, self._n)
        if not isinstance(attrs, pd.DataFrame):
            raise TypeError("population sampler must return a DataFrame.")
        if len(attrs) != self._n:
            raise ValueError(f"population sampler returned {len(attrs)} rows, expected {self._n}.")
        return attrs.reset_index(drop=True)

    def _merge_overrides(self, params: pd.Series, overrides: Mapping[str, Any]) -> pd.Series:
        if not overrides:
            return params
        merged = params.copy()
        for key, value in overrides.items():
            merged[key] = value
        return merged

    def _simulate(
        self, params: pd.Series, attrs: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raise NotImplementedError

    def evaluate(
        self, draws: pd.DataFrame, *, trace: bool = False
    ) -> Outcomes | tuple[Outcomes, pd.DataFrame]:
        """Simulate the population for every draw and average to `Outcomes`.

        Args:
            draws: Parameter draw matrix (rows = iterations). Its index becomes
                the outcome iteration index.
            trace: Also return per-individual cost and effect as a long
                ``DataFrame``, the optional individual-level side channel.

        Returns:
            `Outcomes`, or ``(Outcomes, trace)`` when ``trace`` is set.
        """
        if draws.empty:
            raise ValueError("draws is empty.")
        strategy_names = list(self._strategies)
        rows: list[pd.DataFrame] = []
        traces: list[pd.DataFrame] = []
        for label, (_, raw_params) in zip(draws.index, draws.iterrows(), strict=True):
            iter_seq = self._seed_manager.child_sequence(_iteration_key(label))
            shared_attrs: pd.DataFrame | None = None
            shared_txn: np.random.SeedSequence | None = None
            if self._independent_streams:
                sub = iter_seq.spawn(2 * len(strategy_names))
            else:
                pop_seq, shared_txn = iter_seq.spawn(2)
                shared_attrs = self._sample_population(np.random.default_rng(pop_seq))
            for j, (name, overrides) in enumerate(self._strategies.items()):
                params = self._merge_overrides(raw_params, overrides)
                if self._independent_streams:
                    attrs = self._sample_population(np.random.default_rng(sub[2 * j]))
                    rng = np.random.default_rng(sub[2 * j + 1])
                else:
                    assert shared_attrs is not None and shared_txn is not None
                    attrs = shared_attrs.copy()
                    rng = np.random.default_rng(shared_txn)  # common random numbers
                cost, eff = self._simulate(params, attrs, rng)
                rows.append(aggregate({"cost": cost, self._effect: eff}, name, label))
                if trace:
                    traces.append(
                        pd.DataFrame(
                            {
                                STRATEGY_LEVEL: name,
                                ITERATION_LEVEL: label,
                                "individual": np.arange(len(cost)),
                                "cost": cost,
                                self._effect: eff,
                            }
                        )
                    )
        data = pd.concat(rows)
        full_index = pd.MultiIndex.from_product(
            [strategy_names, draws.index], names=[STRATEGY_LEVEL, ITERATION_LEVEL]
        )
        outcomes = Outcomes(data.reindex(full_index), effect=self._effect)
        if trace:
            return outcomes, pd.concat(traces, ignore_index=True)
        return outcomes


class MicrosimModel(_MicrosimBase):
    """Individual-level microsimulation engine, discrete- or continuous-time.

    The ``clock`` argument selects the simulation kernel and which callback the
    constructor expects.

    ``clock="discrete"`` vectorizes over individuals and loops over cycles. The
    state is an integer vector; each cycle one ``rng.random(n)`` draw and a
    cumulative-probability comparison samples the next state. History enters
    through two attribute columns the engine maintains and passes to
    ``transition`` and ``payoffs``: ``cycle`` (0-based cycle index) and
    ``time_in_state`` (cycles the individual has spent in its current state).

    ``clock="continuous"`` races competing time-to-event samplers. ``hazards``
    returns sampled times to each competing event; the engine takes the
    earliest, advances the clock to it, and accrues cost and utility
    continuously over the elapsed segment. There is no cycle grid; ``horizon``
    truncates. The current clock is passed to ``hazards`` and ``payoffs`` as a
    ``time`` attribute column.

    ``discount_rate`` is an annual rate on an annual clock. ``cycle_length``
    scales the discrete clock: with ``cycle_length=0.5`` each cycle discounts
    by half a year.

    Args:
        states: State labels; the first is the default starting state.
        payoffs: Discrete clock: ``fn(params, state, attrs) -> (cost, qaly)``,
            the per-cycle cost and effect of each individual's current state.
            Continuous clock: ``fn(params, state, attrs) -> (cost_rate,
            qaly_rate)``, the per-year flows. Each returns shape ``(n,)``.
        strategies: Map of strategy name to a parameter-override dict merged
            into ``params`` for that strategy. Order is preserved in `Outcomes`.
        seed_manager: Root of all randomness.
        clock: ``"discrete"`` (default) or ``"continuous"``.
        transition: Discrete clock only. ``fn(params, state, attrs, rng) ->
            probs``, shape ``(n, n_states)`` with each row summing to 1.
        hazards: Continuous clock only. ``fn(params, state, attrs, rng) ->
            times``, shape ``(n, n_states)``, the sampled time to each
            destination state. Use ``inf`` where a transition cannot occur,
            including a state's own column and every column of an absorbing
            state.
        population: Attribute sampler ``fn(rng, n) -> DataFrame`` for
            heterogeneity, or an ``int`` count for a featureless population.
            ``None`` uses ``n_individuals`` with no attributes.
        cycle_length: Discrete clock: years per cycle.
        horizon: Discrete clock: number of cycles to simulate. Continuous clock:
            time horizon in years; trajectories truncate here.
        discount_rate: Annual discount rate for costs and effects (0.03 by
            default).
        half_cycle_correction: Discrete clock: halve the first and last cycle's
            accrual.
        n_individuals: Population size when ``population`` is a sampler or None.
        initial_state: Starting state label or index.
        effect: Name of the effect column (QALYs by default).
        independent_streams: Give each strategy its own population and
            simulation stream instead of common random numbers.
        duration_groups: Discrete clock only. Optional map of attribute name to
            a set of state labels. For each entry the engine maintains a
            per-individual counter of consecutive cycles spent in that set of
            states (0 on the first such cycle, reset when the individual leaves
            the set) and passes it to ``transition`` and ``payoffs`` in
            ``attrs``. Unlike ``time_in_state``, which counts one exact state, a
            duration group spans several states, so a sojourn that progresses
            (Sick to Sicker) keeps counting.
        max_events: Continuous clock only. Safety cap on events per individual
            before raising.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heval.models import MicrosimModel
        >>> from heval.run import SeedManager
        >>> def transition(params, state, attrs, rng):
        ...     probs = np.zeros((len(state), 2))
        ...     probs[state == 0] = [1 - params["p_die"], params["p_die"]]
        ...     probs[state == 1] = [0.0, 1.0]  # dead is absorbing
        ...     return probs
        >>> def payoffs(params, state, attrs):
        ...     alive = (state == 0).astype(float)
        ...     return alive * params["cost"], alive
        >>> engine = MicrosimModel(
        ...     states=("healthy", "dead"), transition=transition, payoffs=payoffs,
        ...     population=500, strategies={"care": {}}, horizon=10,
        ...     seed_manager=SeedManager(0), half_cycle_correction=False)
        >>> draws = pd.DataFrame({"p_die": [0.1], "cost": [1000.0]},
        ...                      index=pd.RangeIndex(1, name="iteration"))
        >>> engine.evaluate(draws).strategies
        ['care']
    """

    def __init__(
        self,
        *,
        states: tuple[str, ...],
        payoffs: Callable[..., tuple[NDArray[np.float64], NDArray[np.float64]]],
        strategies: StrategySpec,
        seed_manager: SeedManager,
        clock: str = "discrete",
        transition: Callable[..., NDArray[np.float64]] | None = None,
        hazards: Callable[..., NDArray[np.float64]] | None = None,
        population: PopulationSpec = None,
        cycle_length: float = 1.0,
        horizon: float = 60,
        discount_rate: float = 0.03,
        half_cycle_correction: bool = True,
        n_individuals: int = 1_000,
        initial_state: str | int = 0,
        effect: str = "qaly",
        independent_streams: bool = False,
        duration_groups: Mapping[str, Sequence[str | int]] | None = None,
        max_events: int = 10_000,
    ) -> None:
        if clock not in ("discrete", "continuous"):
            raise ValueError(f"clock must be 'discrete' or 'continuous', got {clock!r}.")
        super().__init__(
            states=states,
            payoffs=payoffs,
            population=population,
            strategies=strategies,
            seed_manager=seed_manager,
            n_individuals=n_individuals,
            initial_state=initial_state,
            discount_rate=discount_rate,
            effect=effect,
            independent_streams=independent_streams,
        )
        self._clock = clock
        if clock == "discrete":
            if transition is None:
                raise TypeError("clock='discrete' requires a transition callback.")
            if hazards is not None:
                raise TypeError("hazards is only valid with clock='continuous'.")
            if horizon < 1:
                raise ValueError("horizon must be at least one cycle.")
            self._transition = transition
            self._cycle_length = float(cycle_length)
            self._horizon: float = int(horizon)
            self._half_cycle_correction = bool(half_cycle_correction)
            self._duration_groups = {
                name: np.array([self._state_index(s) for s in members], dtype=np.int64)
                for name, members in (duration_groups or {}).items()
            }
        else:
            if hazards is None:
                raise TypeError("clock='continuous' requires a hazards callback.")
            if transition is not None:
                raise TypeError("transition is only valid with clock='discrete'.")
            if horizon <= 0:
                raise ValueError("horizon must be positive.")
            self._hazards = hazards
            self._horizon = float(horizon)
            self._max_events = int(max_events)

    def _simulate(
        self, params: pd.Series, attrs: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._clock == "discrete":
            return self._simulate_discrete(params, attrs, rng)
        return self._simulate_continuous(params, attrs, rng)

    def _simulate_discrete(
        self, params: pd.Series, attrs: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = len(attrs)
        n_states = len(self._states)
        horizon = int(self._horizon)
        state = np.full(n, self._initial_index, dtype=np.int64)
        time_in_state = np.zeros(n, dtype=np.int64)
        durations = {name: np.zeros(n, dtype=np.int64) for name in self._duration_groups}
        n_points = horizon + 1
        cost_grid = np.empty((n, n_points), dtype=np.float64)
        eff_grid = np.empty((n, n_points), dtype=np.float64)
        for c in range(n_points):
            view = attrs.assign(cycle=c, time_in_state=time_in_state, **durations)
            cost, eff = self._payoffs(params, state, view)
            cost_grid[:, c] = cost
            eff_grid[:, c] = eff
            if c < horizon:
                probs = np.asarray(self._transition(params, state, view, rng), dtype=np.float64)
                if probs.shape != (n, n_states):
                    raise ValueError(
                        f"transition must return shape {(n, n_states)}, got {probs.shape}."
                    )
                if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
                    raise ValueError("transition rows must each sum to 1.")
                new_state = self._sample_next(probs, rng)
                moved = new_state != state
                time_in_state = np.where(moved, 0, time_in_state + 1)
                for name, members in self._duration_groups.items():
                    was_in = np.isin(state, members)
                    now_in = np.isin(new_state, members)
                    durations[name] = np.where(now_in, np.where(was_in, durations[name] + 1, 0), 0)
                state = new_state
        times = np.arange(n_points, dtype=np.float64) * self._cycle_length
        weights = np.ones(n_points, dtype=np.float64)
        if self._half_cycle_correction:
            weights[0] = weights[-1] = 0.5
        cost_total = accrue(cost_grid, times, self._discount_rate, weights=weights)
        eff_total = accrue(eff_grid, times, self._discount_rate, weights=weights)
        return cost_total, eff_total

    @staticmethod
    def _sample_next(
        probs: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.int64]:
        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0  # guard against floating-point shortfall in the last bin
        u = np.asarray(rng.random(probs.shape[0]))
        return (u[:, None] < cdf).argmax(axis=1).astype(np.int64)

    def _simulate_continuous(
        self, params: pd.Series, attrs: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = len(attrs)
        n_states = len(self._states)
        horizon = float(self._horizon)
        state = np.full(n, self._initial_index, dtype=np.int64)
        clock = np.zeros(n, dtype=np.float64)
        cost_total = np.zeros(n, dtype=np.float64)
        eff_total = np.zeros(n, dtype=np.float64)
        active = np.ones(n, dtype=bool)
        for _ in range(self._max_events):
            if not active.any():
                break
            view = attrs.assign(time=clock)
            times = np.asarray(self._hazards(params, state, view, rng), dtype=np.float64)
            if times.shape != (n, n_states):
                raise ValueError(
                    f"hazards must return shape {(n, n_states)}, got {times.shape}."
                )
            dest = np.argmin(times, axis=1)
            dt = times[np.arange(n), dest]
            remaining = horizon - clock
            segment = np.where(active, np.minimum(dt, remaining), 0.0)
            cost_rate, eff_rate = self._payoffs(params, state, view)
            disc_cost = integrate_flow(clock, segment, self._discount_rate)
            disc_eff = integrate_flow(clock, segment, self._discount_rate)
            cost_total += np.where(active, cost_rate * disc_cost, 0.0)
            eff_total += np.where(active, eff_rate * disc_eff, 0.0)
            event = active & np.isfinite(dt) & (dt <= remaining)
            clock = np.where(active, clock + segment, clock)
            state = np.where(event, dest, state)
            active = event
        else:
            if active.any():
                raise RuntimeError(
                    "max_events exceeded before every individual reached the horizon or an "
                    "absorbing state; raise max_events or check the hazards."
                )
        return cost_total, eff_total
