"""Discrete-event simulation engine, a thin wrapper around SimPy.

`DESModel` runs a SimPy model once per parameter draw and intervention and returns
the standard `Outcomes` structure. It is not a new discrete-event kernel: the SimPy
``Environment``, the process functions, and the ``Resource`` objects stay the
user's own code. `heormodel` adds only what SimPy leaves out for a health economic
model:

- a per-entity `_DESToolkit` for cost and utility accrual with discounting,
  reusing `heormodel.models._accrual` between events exactly as the continuous-time
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
(``exp(-rate * t)``), matching `MicrosimModel(clock="continuous")`, because a DES runs
in continuous time.

``simpy`` is an optional dependency: install with ``uv pip install 'heormodel[des]'``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from heormodel._util import require_optional
from heormodel.models._accrual import discount_factor, integrate_flow
from heormodel.models._engine import PopulationSampler, PopulationSpec, StreamingEngine
from heormodel.models._interventions import InterventionSpec
from heormodel.models.outcomes import INTERVENTION_LEVEL, ITERATION_LEVEL

if TYPE_CHECKING:
    import simpy

ResourceFn = Callable[[Any, pd.Series, str], Mapping[str, Any]]
ProcessFn = Callable[..., Any]

# Event-log column names, the columns of the optional trace side channel.
_LOG_COLS = ("intervention", "iteration", "entity", "t", "event", "state", "resource")


def _require_simpy() -> Any:
    return require_optional("simpy", feature="The discrete-event engine", extra="des")


class _DESToolkit:
    """What `heormodel` adds on top of SimPy, one instance per entity per run.

    Handed to each process as ``toolkit``. It accrues discounted cost and effect
    for its entity, marks trajectory segments and resource events in the shared
    event log, exposes the entity's derived generator as ``rng``, and exposes the
    run's time horizon as ``horizon`` so a process need not duplicate it as a
    module constant.
    """

    def __init__(
        self,
        *,
        env: simpy.Environment,
        rng: np.random.Generator,
        resources: Mapping[str, Any],
        entity_id: int,
        horizon: float,
        discount_rate: float,
        log: list[dict[str, Any]] | None,
    ) -> None:
        self.env = env
        self.rng = rng
        self.horizon = horizon
        self._resources = resources
        self._entity_id = entity_id
        self._rate = discount_rate
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
        factor = float(discount_factor(self.env.now, self._rate, continuous=True))
        discounted = float(amount) * factor
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
        b = min(self.horizon, float(end))
        dur = max(0.0, b - a)
        cost = float(cost_rate) * float(integrate_flow(a, dur, self._rate))
        self.effect += float(effect_rate) * float(integrate_flow(a, dur, self._rate))
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


class DESModel(StreamingEngine):
    """Discrete-event simulation engine wrapping SimPy.

    Each process is the user's own SimPy code with signature
    ``process(env, entity, params, intervention, toolkit)``. ``entity`` is that
    individual's attribute row, ``params`` the iteration's draw merged with the
    intervention decision levers, ``intervention`` the intervention name, and ``toolkit`` the
    `_DESToolkit` that accrues cost and effect and logs the trajectory. Per
    iteration and intervention the engine builds the environment, samples entities,
    creates the shared resources, runs every process to ``horizon``, collects the
    per-entity discounted accruals, averages them, and writes one `Outcomes` row.

    Args:
        process: The SimPy process factory, ``fn(env, entity, params, intervention,
            toolkit)`` returning a generator.
        population: Attribute sampler ``fn(rng, n) -> DataFrame``, an ``int``
            count for a featureless population, or ``None`` to use
            ``n_individuals`` with no attributes.
        interventions: A sequence of intervention names or `heormodel.models.Intervention`
            objects; an `Intervention` may carry parameter decision levers merged into
            ``params`` for that intervention. Order is preserved in `Outcomes`.
        resources: ``fn(env, params, intervention) -> dict[str, simpy.Resource]``,
            built fresh for each run and shared by every entity in it. ``None``
            for a model with no constrained resources.
        horizon: Time horizon in the environment's unit; the run stops here. Each
            process reads it back as ``toolkit.horizon``.
        discount_rate: Annual (per-unit-time) discount rate for costs and
            effects (0.03 by default). Discounting is continuous
            (``exp(-rate * t)``).
        n_individuals: Population size when ``population`` is a sampler or
            ``None``.
        effect: Name of the effect column (QALYs by default).
        independent_streams: Give each intervention its own population and streams
            instead of common random numbers.

    Randomness is supplied by `heormodel.run.run_psa` at run time, not at
    construction; the engine holds no seed of its own.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heormodel.models import DESModel
        >>> def process(env, entity, params, intervention, toolkit):
        ...     wait = toolkit.rng.exponential(params["los"])
        ...     toolkit.accrue_rate(params["day_cost"], 1.0, wait)
        ...     yield env.timeout(wait)
        >>> engine = DESModel(
        ...     process=process, population=200, interventions=["ward"],
        ...     horizon=30.0)
        >>> draws = pd.DataFrame({"los": [3.0], "day_cost": [500.0]},
        ...                      index=pd.RangeIndex(1, name="iteration"))
        >>> engine.evaluate(draws).interventions
        ['ward']
    """

    def __init__(
        self,
        *,
        process: ProcessFn,
        population: PopulationSpec = None,
        interventions: InterventionSpec,
        resources: ResourceFn | None = None,
        horizon: float,
        discount_rate: float = 0.03,
        n_individuals: int = 1_000,
        effect: str = "qaly",
        independent_streams: bool = False,
    ) -> None:
        _require_simpy()
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self._configure(interventions, discount_rate, effect)
        self._process = process
        self._resources_fn = resources
        self._horizon = float(horizon)
        self._independent_streams = bool(independent_streams)
        self._population = PopulationSampler(population, n_individuals)
        self._n = self._population.n

    def _simulation_randomness(
        self, source: np.random.SeedSequence
    ) -> list[np.random.SeedSequence]:
        """Each arm's per-entity seed sequences, spawned once from ``source``."""
        return list(source.spawn(self._n))

    def _run_arm(
        self,
        params: pd.Series,
        intervention: str,
        attrs: pd.DataFrame,
        sim_randomness: Any,
        *,
        collect_events: bool,
    ) -> tuple[dict[str, NDArray[np.float64]], pd.DataFrame | None]:
        log: list[dict[str, Any]] | None = [] if collect_events else None
        accruals = self._run_once(params, attrs, sim_randomness, intervention, log)
        events = pd.DataFrame(log, columns=list(_LOG_COLS[2:])) if collect_events else None
        return accruals, events

    def _run_once(
        self,
        params: pd.Series,
        attrs: pd.DataFrame,
        entity_seqs: list[np.random.SeedSequence],
        intervention: str,
        log: list[dict[str, Any]] | None,
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate one (iteration, intervention) and return per-entity accruals."""
        simpy = _require_simpy()
        env = simpy.Environment()
        resources = (
            dict(self._resources_fn(env, params, intervention)) if self._resources_fn else {}
        )
        toolkits: list[_DESToolkit] = []
        for i in range(self._n):
            toolkit = _DESToolkit(
                env=env,
                rng=np.random.default_rng(entity_seqs[i]),
                resources=resources,
                entity_id=i,
                horizon=self._horizon,
                discount_rate=self._discount_rate,
                log=log,
            )
            toolkits.append(toolkit)
            env.process(self._process(env, attrs.iloc[i], params, intervention, toolkit))
        env.run(until=self._horizon)
        cost = np.array([tk.cost for tk in toolkits], dtype=np.float64)
        eff = np.array([tk.effect for tk in toolkits], dtype=np.float64)
        result: dict[str, NDArray[np.float64]] = {"cost": cost, self._effect: eff}
        component_names = sorted({name for tk in toolkits for name in tk.components})
        for name in component_names:
            result[name] = np.array([tk.components.get(name, 0.0) for tk in toolkits])
        return result



def queue_waits(trace: pd.DataFrame) -> pd.DataFrame:
    """Per-request waiting times, derived from a `DESModel` trace.

    Pairs each ``request`` event with its ``grant`` for the same entity, intervention,
    iteration, and resource, and reports the wait between them. This is the
    pattern the design intends: queueing reports come from the event log, so
    analysis code never reaches into the engine.

    Args:
        trace: The event log from ``run_psa(engine, draws, collect="events").events``.

    Returns:
        One row per served request with a ``wait`` column.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models.des import queue_waits
        >>> trace = pd.DataFrame({
        ...     "intervention": ["s", "s"], "iteration": [0, 0], "entity": [0, 0],
        ...     "t": [1.0, 4.0], "event": ["request", "grant"],
        ...     "state": [None, None], "resource": ["clinic", "clinic"]})
        >>> float(queue_waits(trace)["wait"].iloc[0])
        3.0
    """
    keys = [INTERVENTION_LEVEL, ITERATION_LEVEL, "entity", "resource"]
    reqs = trace[trace["event"] == "request"].copy()
    grants = trace[trace["event"] == "grant"].copy()
    # Pair the k-th request with the k-th grant, so repeat requests do not cross.
    reqs["_k"] = reqs.groupby(keys).cumcount()
    grants["_k"] = grants.groupby(keys).cumcount()
    merged = reqs.merge(grants, on=[*keys, "_k"], suffixes=("_request", "_grant"))
    merged["wait"] = merged["t_grant"] - merged["t_request"]
    return merged[[*keys, "wait"]].reset_index(drop=True)
