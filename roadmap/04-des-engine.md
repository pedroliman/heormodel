# 4. Discrete-event simulation engine (SimPy wrapper)

Implement `DESEngine`, replacing the stub in `heval/models/des.py`, as a thin wrapper around SimPy. Not a new DES kernel. It stays coherent with the microsimulation architecture in item 3.

## Guardrails

- Do not reimplement an event loop, queues, or resources. SimPy's `Environment`, `Process`, and `Resource` remain the user's own code. `heval` adds trajectory recording, resource-constraint helpers, cost and utility accrual, seeding, and aggregation to `Outcomes`.
- The engine shares the output contract and the `_accrual` layer with the microsim engines, never an implementation API. A DES model is not forced to look like a microsimulation.

## Coherence with the microsim architecture

The same three commitments, the same shapes:

1. Configure once, evaluate on draws: `DESEngine(...)` takes the model (a process factory); `evaluate(draws)` returns `Outcomes` indexed by `draws.index`.
2. Seeding: a `SeedManager` at construction, one child generator per iteration, entity-level streams derived from the iteration stream. Results do not depend on `n_jobs`.
3. Accrual: reuse `heval/models/_accrual.py` from item 3, extended with accrual between events (which the continuous-time microsim also uses). One implementation, two engines. If DES starts first, `_accrual` is pulled forward.

## Sketch

```python
DESEngine(
    process=fn,           # fn(env, entity, params, strategy, toolkit) -> SimPy process
    entities=fn | int,    # attribute sampler fn(rng, n) -> DataFrame, or a count
    resources=fn,         # fn(env, params, strategy) -> dict[str, simpy.Resource]
    strategies={"SoC": {...}, "Fast track": {...}},
    horizon=10.0,
    discount_cost=0.03,
    discount_effect=0.03,
    seed_manager=SeedManager(...),
)
```

The `toolkit` handed to each process is what `heval` adds on top of SimPy:

- `toolkit.accrue_cost(amount)` and `toolkit.accrue_rate(cost_rate, utility)`: point and continuous accruals, discounted at the current `env.now` through `_accrual`.
- `toolkit.state(name)`: marks trajectory segments in the per-entity event log (`entity, t, event, state, resource`). The log is the optional trace side channel, the same pattern as the microsim `trace=`.
- `toolkit.request(resource_name)`: a context manager around `resource.request()` that logs queueing time. Waiting-time and utilization reports come from the event log, so analysis code never touches engine internals.
- `toolkit.rng`: the entity's derived generator.

Per iteration and strategy, `evaluate` builds the environment, resources, and entities, runs processes to `horizon`, collects per-entity discounted accruals, averages within the iteration, and emits `Outcomes` rows. Disaggregated costs (per resource, for example) map onto the schema's component columns.

## Dependencies

- `simpy` as an optional extra, `heval[des]`, mirroring the `pyabc` pattern: lazy import with an actionable error message.
- `_accrual` from item 3.

## Validation (acceptance)

- A single-resource clinic with known analytic waiting time and throughput (M/M/1): simulated means converge within statistical tolerance.
- A no-resource DES with exponential event times reproduces the same analytic cohort solution used to validate the continuous-time microsim. The two engines must agree with each other and with the closed form.
- Contract and reproducibility tests identical in shape to the microsim ones, with shared test helpers.
