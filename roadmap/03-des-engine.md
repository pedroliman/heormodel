# 3 — Discrete-event simulation engine (SimPy wrapper)

**Goal:** implement `DESEngine` (replacing the stub in
`heval/models/des.py`) as a **thin wrapper around SimPy** — not a new DES
kernel — that stays coherent with the microsimulation architecture in
design note 02.

## Guardrails (restated from the package charter)

- Do not reimplement an event loop, queues, or resources: SimPy's
  `Environment`, `Process`, and `Resource` remain the user's own code and
  remain visible. `heval` adds only trajectory recording,
  resource-constraint helpers, cost/utility accrual, seeding, and
  aggregation to `Outcomes`.
- The engine shares the *output contract* and the `_accrual` layer with
  the microsim engines — never an implementation API. A DES model is not
  forced to look like a microsim model.

## Coherence with the microsim architecture

Same three commitments, same shapes:

1. **Configure once, evaluate on draws**: `DESEngine(...)` takes the model
   (a process factory), `evaluate(draws)` returns `Outcomes` indexed by
   `draws.index`.
2. **Seeding**: a `SeedManager` at construction; one child generator per
   PSA iteration; entity-level streams derived from the iteration stream.
   Same reproducibility guarantees (invariant to `n_jobs`).
3. **Accrual**: reuse `heval/models/_accrual.py` (built in the microsim
   phase) for discounting and aggregation, extended with
   continuous accrual-between-events (which the continuous-time microsim
   also uses) — one implementation, two engines.

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

The `toolkit` handed to each process is the value `heval` adds on top of
SimPy:

- `toolkit.accrue_cost(amount)` / `toolkit.accrue_rate(cost_rate, utility)`
  — point and continuous accruals, discounted at the entity's current
  `env.now` using `_accrual`.
- `toolkit.state(name)` — marks trajectory segments; enters the per-entity
  **event log** (`entity, t, event, state, resource`) that becomes the
  optional trajectory side channel (same side-channel pattern as microsim
  `trace=`).
- `toolkit.request(resource_name)` — a context manager around
  `resource.request()` that logs queueing time (waiting-time outcomes and
  resource-utilisation reports come from the event log, not from analysis
  code touching engine internals).
- `toolkit.rng` — the entity's derived generator.

`evaluate` runs, per PSA iteration and strategy: build `env`, resources,
entities; run processes to `horizon`; collect per-entity discounted
accruals; average within iteration; emit `Outcomes` rows. Disaggregated
cost components (e.g. per-resource cost) map onto the schema's optional
component columns.

## Dependencies

- `simpy` as an optional extra: `heval[des]` (mirror the `pyabc` pattern —
  lazy import with an actionable error message).
- `_accrual` module from the microsim phase (build order matters; if DES
  is started first, `_accrual` is pulled forward).

## Validation (acceptance bar)

- An M/M/1-style single-resource clinic where waiting time and throughput
  have known analytic values: simulated means converge within statistical
  tolerance.
- A no-resource DES with exponential event times reproduces the same
  analytic cohort solution used to validate the continuous-time microsim —
  the two engines must agree with each other and with the closed form.
- Contract + reproducibility tests identical in shape to the microsim
  ones (shared test helpers).
