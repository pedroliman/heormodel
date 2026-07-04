"""Discrete-event simulation engine, a thin wrapper around SimPy.

`DESEngine` runs a SimPy model once per PSA iteration and strategy and emits the
standard `Outcomes` schema. It is not a new discrete-event kernel: the SimPy
``Environment``, the process functions, and the ``Resource`` objects stay the
user's own code. `heval` adds only what SimPy leaves out for a health economic
model:

- a per-entity `_DESToolkit` for cost and utility accrual with discounting,
  reusing `heval.models._accrual` between events exactly as the continuous-time
  microsimulation engine does,
- per-iteration seeding from a `SeedManager`, so results do not depend on how a
  run is chunked across workers,
- an optional per-entity event log (the trace side channel), from which queueing
  and utilization reports are derived without touching engine internals.

The engine keeps the same three commitments as the microsimulation engines.
Configure once and evaluate on draws: the constructor takes the model, and
`evaluate` takes only the parameter draw matrix, returning `Outcomes` indexed by
``draws.index``. Seed each iteration from an injected `SeedManager`. Accrue and
discount through the shared `_accrual` module. Discounting is continuous
(``exp(-rate * t)``), matching `ContinuousTimeMicrosimEngine`, because a DES runs
in continuous time.

``simpy`` is an optional dependency: install with ``uv pip install 'heval[des]'``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from heval.models._accrual import aggregate, discount_factor, integrate_flow
from heval.models.outcomes import (
    ITERATION_LEVEL,
    STRATEGY_LEVEL,
    Outcomes,
)
from heval.run.seeds import SeedManager

if TYPE_CHECKING:
    import simpy

EntitySpec = int | Callable[[np.random.Generator, int], pd.DataFrame] | None
ResourceFn = Callable[[Any, pd.Series, str], Mapping[str, Any]]
ProcessFn = Callable[..., Any]
StrategySpec = Mapping[str, Mapping[str, Any]]

# Event-log column names, the schema of the optional trace side channel.
_LOG_COLS = ("strategy", "iteration", "entity", "t", "event", "state", "resource")


def _require_simpy() -> Any:
    try:
        import simpy
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "The discrete-event engine requires simpy; install it with "
            "uv pip install 'heval[des]'."
        ) from err
    return simpy


def _iteration_key(label: Any) -> int:
    """Turn an iteration label into a stable integer seed key."""
    try:
        return int(label)
    except (TypeError, ValueError):
        digest = hashlib.blake2b(repr(label).encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big")


class _DESToolkit:
    """What `heval` adds on top of SimPy, one instance per entity per run.

    Handed to each process as ``toolkit``. It accrues discounted cost and effect
    for its entity, marks trajectory segments and resource events in the shared
    event log, and exposes the entity's derived generator as ``rng``.
    """

    def __init__(
        self,
        *,
        env: simpy.Environment,
        rng: np.random.Generator,
        resources: Mapping[str, Any],
        entity_id: int,
        horizon: float,
        discount_cost: float,
        discount_effect: float,
        log: list[dict[str, Any]] | None,
    ) -> None:
        self.env = env
        self.rng = rng
        self._resources = resources
        self._entity_id = entity_id
        self._horizon = horizon
        self._dc = discount_cost
        self._de = discount_effect
        self._log = log
        self.cost = 0.0
        self.effect = 0.0
        self.components: dict[str, float] = {}
        self._state: str | None = None

    def accrue_cost(self, amount: float, *, component: str | None = None) -> None:
        """Accrue a one-off cost at the current time, discounted to time zero.

        Args:
            amount: Cost incurred now (``env.now``).
            component: Optional label; the amount is also added to a
                disaggregated component subtotal carried into `Outcomes`.
        """
        discounted = float(amount) * float(discount_factor(self.env.now, self._dc, continuous=True))
        self.cost += discounted
        if component is not None:
            self.components[component] = self.components.get(component, 0.0) + discounted

    def accrue_over(
        self,
        start: float,
        end: float,
        cost_rate: float,
        effect_rate: float,
        *,
        component: str | None = None,
    ) -> None:
        """Accrue a continuous flow over an absolute time interval ``[start, end]``.

        The interval is clamped to ``[0, horizon]`` and the flow is integrated and
        discounted by absolute time, so the result does not depend on when the
        call is made. Use it to bill a segment already elapsed, such as the
        queueing time an entity just endured: call it right after the request is
        granted, with ``start`` the arrival time and ``end`` ``env.now``.

        Args:
            start: Segment start in the environment's time unit.
            end: Segment end.
            cost_rate: Cost per unit time over the segment.
            effect_rate: Effect (e.g. utility) per unit time over the segment.
            component: Optional label for the cost subtotal.
        """
        a = max(0.0, float(start))
        b = min(self._horizon, float(end))
        dur = max(0.0, b - a)
        cost = float(cost_rate) * float(integrate_flow(a, dur, self._dc))
        self.effect += float(effect_rate) * float(integrate_flow(a, dur, self._de))
        self.cost += cost
        if component is not None:
            self.components[component] = self.components.get(component, 0.0) + cost

    def accrue_rate(
        self,
        cost_rate: float,
        effect_rate: float,
        duration: float,
        *,
        component: str | None = None,
    ) -> None:
        """Accrue a continuous cost and effect flow over the next ``duration``.

        Integrates forward from the current time, truncated at the horizon, so a
        segment that would run past ``horizon`` contributes only up to it. Call it
        before the matching ``yield env.timeout(duration)``. Equivalent to
        ``accrue_over(env.now, env.now + duration, ...)``.

        Args:
            cost_rate: Cost per unit time over the segment.
            effect_rate: Effect (e.g. utility) per unit time over the segment.
            duration: Segment length in the environment's time unit.
            component: Optional label for the cost subtotal.
        """
        now = float(self.env.now)
        self.accrue_over(
            now, now + float(duration), cost_rate, effect_rate, component=component
        )

    def state(self, name: str) -> None:
        """Mark the entity as entering trajectory segment ``name`` in the log."""
        self._state = name
        self._record("state", state=name)

    def request(self, resource_name: str) -> _RequestContext:
        """Context manager around a SimPy resource request that logs queueing.

        Use it as ``with toolkit.request("clinician") as req: yield req``. The
        request and its grant are timestamped in the event log, so waiting times
        and utilization come from the log, never from engine internals.
        """
        if resource_name not in self._resources:
            raise KeyError(
                f"Unknown resource {resource_name!r}; declared resources are "
                f"{sorted(self._resources)}."
            )
        return _RequestContext(self, resource_name, self._resources[resource_name])

    def _record(
        self, event: str, *, state: str | None = None, resource: str | None = None
    ) -> None:
        if self._log is None:
            return
        self._log.append(
            {
                "entity": self._entity_id,
                "t": float(self.env.now),
                "event": event,
                "state": state if state is not None else self._state,
                "resource": resource,
            }
        )


class _RequestContext:
    """A logging context manager wrapping one ``resource.request()``."""

    def __init__(self, toolkit: _DESToolkit, name: str, resource: Any) -> None:
        self._tk = toolkit
        self._name = name
        self._resource = resource
        self._req: Any = None

    def __enter__(self) -> Any:
        self._tk._record("request", resource=self._name)
        self._req = self._resource.request()
        self._req.callbacks.append(self._on_grant)
        return self._req

    def _on_grant(self, event: Any) -> None:
        self._tk._record("grant", resource=self._name)

    def __exit__(self, *exc: Any) -> None:
        self._resource.release(self._req)
        self._tk._record("release", resource=self._name)


class DESEngine:
    """Discrete-event simulation engine wrapping SimPy.

    Each process is the user's own SimPy code with signature
    ``process(env, entity, params, strategy, toolkit)``. ``entity`` is that
    individual's attribute row, ``params`` the iteration's draw merged with the
    strategy overrides, ``strategy`` the strategy name, and ``toolkit`` the
    `_DESToolkit` that accrues cost and effect and logs the trajectory. Per
    iteration and strategy the engine builds the environment, samples entities,
    creates the shared resources, runs every process to ``horizon``, collects the
    per-entity discounted accruals, averages them, and writes one `Outcomes` row.

    Args:
        process: The SimPy process factory, ``fn(env, entity, params, strategy,
            toolkit)`` returning a generator.
        entities: Attribute sampler ``fn(rng, n) -> DataFrame``, an ``int`` count
            for a featureless population, or ``None`` to use ``n_entities`` with
            no attributes.
        strategies: Map of strategy name to a parameter-override dict merged into
            ``params`` for that strategy. Order is preserved in `Outcomes`.
        seed_manager: Root of all randomness. Each iteration draws a stream keyed
            by its index; entity streams derive from it.
        resources: ``fn(env, params, strategy) -> dict[str, simpy.Resource]``,
            built fresh for each run and shared by every entity in it. ``None``
            for a model with no constrained resources.
        horizon: Time horizon in the environment's unit; the run stops here.
        discount_cost: Annual (per-unit-time) discount rate for costs.
        discount_effect: Discount rate for effects.
        n_entities: Population size when ``entities`` is a sampler or ``None``.
        effect: Name of the effect column (QALYs by default).
        independent_streams: Give each strategy its own entities and streams
            instead of common random numbers.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heval.models import DESEngine
        >>> from heval.run import SeedManager
        >>> def process(env, entity, params, strategy, toolkit):
        ...     wait = toolkit.rng.exponential(params["los"])
        ...     toolkit.accrue_rate(params["day_cost"], 1.0, wait)
        ...     yield env.timeout(wait)
        >>> engine = DESEngine(
        ...     process=process, entities=200, strategies={"ward": {}},
        ...     horizon=30.0, seed_manager=SeedManager(0))
        >>> draws = pd.DataFrame({"los": [3.0], "day_cost": [500.0]},
        ...                      index=pd.RangeIndex(1, name="iteration"))
        >>> engine.evaluate(draws).strategies
        ['ward']
    """

    def __init__(
        self,
        *,
        process: ProcessFn,
        entities: EntitySpec = None,
        strategies: StrategySpec,
        seed_manager: SeedManager,
        resources: ResourceFn | None = None,
        horizon: float,
        discount_cost: float = 0.03,
        discount_effect: float = 0.03,
        n_entities: int = 1_000,
        effect: str = "qaly",
        independent_streams: bool = False,
    ) -> None:
        _require_simpy()
        if not strategies:
            raise ValueError("Provide at least one strategy.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self._process = process
        self._resources_fn = resources
        self._strategies = {name: dict(overrides) for name, overrides in strategies.items()}
        self._seed_manager = seed_manager
        self._horizon = float(horizon)
        self._discount_cost = float(discount_cost)
        self._discount_effect = float(discount_effect)
        self._effect = effect
        self._independent_streams = bool(independent_streams)
        if isinstance(entities, bool):  # bool is an int subclass; reject it explicitly
            raise TypeError("entities must be an int, a callable, or None.")
        if isinstance(entities, int):
            self._n = entities
            self._entities_fn: Callable[..., pd.DataFrame] | None = None
        elif entities is None:
            self._n = n_entities
            self._entities_fn = None
        elif callable(entities):
            self._n = n_entities
            self._entities_fn = entities
        else:
            raise TypeError("entities must be an int, a callable, or None.")
        if self._n <= 0:
            raise ValueError("Population size must be positive.")

    def _sample_entities(self, rng: np.random.Generator) -> pd.DataFrame:
        if self._entities_fn is None:
            return pd.DataFrame(index=pd.RangeIndex(self._n))
        attrs = self._entities_fn(rng, self._n)
        if not isinstance(attrs, pd.DataFrame):
            raise TypeError("entities sampler must return a DataFrame.")
        if len(attrs) != self._n:
            raise ValueError(
                f"entities sampler returned {len(attrs)} rows, expected {self._n}."
            )
        return attrs.reset_index(drop=True)

    def _merge_overrides(self, params: pd.Series, overrides: Mapping[str, Any]) -> pd.Series:
        if not overrides:
            return params
        merged = params.copy()
        for key, value in overrides.items():
            merged[key] = value
        return merged

    def _run_once(
        self,
        params: pd.Series,
        attrs: pd.DataFrame,
        entity_seqs: list[np.random.SeedSequence],
        strategy: str,
        log: list[dict[str, Any]] | None,
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate one (iteration, strategy) and return per-entity accruals."""
        simpy = _require_simpy()
        env = simpy.Environment()
        resources = dict(self._resources_fn(env, params, strategy)) if self._resources_fn else {}
        toolkits: list[_DESToolkit] = []
        for i in range(self._n):
            toolkit = _DESToolkit(
                env=env,
                rng=np.random.default_rng(entity_seqs[i]),
                resources=resources,
                entity_id=i,
                horizon=self._horizon,
                discount_cost=self._discount_cost,
                discount_effect=self._discount_effect,
                log=log,
            )
            toolkits.append(toolkit)
            env.process(self._process(env, attrs.iloc[i], params, strategy, toolkit))
        env.run(until=self._horizon)
        cost = np.array([tk.cost for tk in toolkits], dtype=np.float64)
        eff = np.array([tk.effect for tk in toolkits], dtype=np.float64)
        result: dict[str, NDArray[np.float64]] = {"cost": cost, self._effect: eff}
        component_names = sorted({name for tk in toolkits for name in tk.components})
        for name in component_names:
            result[name] = np.array([tk.components.get(name, 0.0) for tk in toolkits])
        return result

    def evaluate(
        self, draws: pd.DataFrame, *, trace: bool = False
    ) -> Outcomes | tuple[Outcomes, pd.DataFrame]:
        """Simulate every draw and strategy, averaging entities to `Outcomes`.

        Args:
            draws: Parameter draw matrix (rows = iterations). Its index becomes
                the outcome iteration index.
            trace: Also return the per-entity event log as a long ``DataFrame``
                with columns ``strategy, iteration, entity, t, event, state,
                resource``, the optional individual-level side channel.

        Returns:
            `Outcomes`, or ``(Outcomes, trace)`` when ``trace`` is set.

        Example:
            >>> import pandas as pd
            >>> from heval.models import DESEngine
            >>> from heval.run import SeedManager
            >>> def process(env, entity, params, strategy, toolkit):
            ...     toolkit.accrue_cost(params["visit"])
            ...     yield env.timeout(1.0)
            >>> engine = DESEngine(
            ...     process=process, entities=50, strategies={"clinic": {}},
            ...     horizon=5.0, seed_manager=SeedManager(1))
            >>> draws = pd.DataFrame({"visit": [200.0, 210.0]},
            ...                      index=pd.RangeIndex(2, name="iteration"))
            >>> engine.evaluate(draws).n_iterations
            2
        """
        if draws.empty:
            raise ValueError("draws is empty.")
        strategy_names = list(self._strategies)
        rows: list[pd.DataFrame] = []
        traces: list[pd.DataFrame] = []
        for label, (_, raw_params) in zip(draws.index, draws.iterrows(), strict=True):
            iter_seq = self._seed_manager.child_sequence(_iteration_key(label))
            shared_attrs: pd.DataFrame | None = None
            shared_entity_seqs: list[np.random.SeedSequence] | None = None
            if self._independent_streams:
                sub = iter_seq.spawn(2 * len(strategy_names))
            else:
                pop_seq, entity_root = iter_seq.spawn(2)
                shared_attrs = self._sample_entities(np.random.default_rng(pop_seq))
                shared_entity_seqs = list(entity_root.spawn(self._n))
            for j, (name, overrides) in enumerate(self._strategies.items()):
                params = self._merge_overrides(raw_params, overrides)
                if self._independent_streams:
                    attrs = self._sample_entities(np.random.default_rng(sub[2 * j]))
                    entity_seqs = list(sub[2 * j + 1].spawn(self._n))
                else:
                    assert shared_attrs is not None and shared_entity_seqs is not None
                    attrs = shared_attrs.copy()
                    entity_seqs = shared_entity_seqs  # common random numbers
                log: list[dict[str, Any]] | None = [] if trace else None
                accruals = self._run_once(params, attrs, entity_seqs, name, log)
                rows.append(aggregate(accruals, name, label))
                if trace and log is not None:
                    frame = pd.DataFrame(log, columns=list(_LOG_COLS[2:]))
                    frame.insert(0, ITERATION_LEVEL, label)
                    frame.insert(0, STRATEGY_LEVEL, name)
                    traces.append(frame)
        data = pd.concat(rows).fillna(0.0)
        full_index = pd.MultiIndex.from_product(
            [strategy_names, draws.index], names=[STRATEGY_LEVEL, ITERATION_LEVEL]
        )
        outcomes = Outcomes(data.reindex(full_index), effect=self._effect)
        if trace:
            trace_df = (
                pd.concat(traces, ignore_index=True)
                if traces
                else pd.DataFrame(columns=list(_LOG_COLS))
            )
            return outcomes, trace_df
        return outcomes


def queue_waits(trace: pd.DataFrame) -> pd.DataFrame:
    """Per-request waiting times, derived from a `DESEngine` trace.

    Pairs each ``request`` event with its ``grant`` for the same entity, strategy,
    iteration, and resource, and reports the wait between them. This is the
    pattern the design intends: queueing reports come from the event log, so
    analysis code never reaches into the engine.

    Args:
        trace: The event log returned by ``DESEngine.evaluate(..., trace=True)``.

    Returns:
        One row per served request with a ``wait`` column.

    Example:
        >>> import pandas as pd
        >>> from heval.models.des import queue_waits
        >>> trace = pd.DataFrame({
        ...     "strategy": ["s", "s"], "iteration": [0, 0], "entity": [0, 0],
        ...     "t": [1.0, 4.0], "event": ["request", "grant"],
        ...     "state": [None, None], "resource": ["clinic", "clinic"]})
        >>> float(queue_waits(trace)["wait"].iloc[0])
        3.0
    """
    keys = [STRATEGY_LEVEL, ITERATION_LEVEL, "entity", "resource"]
    reqs = trace[trace["event"] == "request"].copy()
    grants = trace[trace["event"] == "grant"].copy()
    # Pair the k-th request with the k-th grant, so repeat requests do not cross.
    reqs["_k"] = reqs.groupby(keys).cumcount()
    grants["_k"] = grants.groupby(keys).cumcount()
    merged = reqs.merge(grants, on=[*keys, "_k"], suffixes=("_request", "_grant"))
    merged["wait"] = merged["t_grant"] - merged["t_request"]
    return merged[[*keys, "wait"]].reset_index(drop=True)
