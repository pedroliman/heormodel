# 3. Microsimulation engine

Implement `DiscreteTimeMicrosimEngine` first, then `ContinuousTimeMicrosimEngine`, replacing the stubs in `heormodel/models/microsim.py`. Both simulate an individual-level population per PSA iteration and emit the standard `Outcomes` schema.

## Architectural commitments

The DES engine (item 4) must stay coherent with these decisions, so they are fixed here:

1. Configure once, evaluate on draws. The constructor takes the model structure; `evaluate(draws)` takes only the parameter draw matrix and returns `Outcomes` with `draws.index` as the iteration index. The existing protocol, unchanged.
2. Randomness comes from a `SeedManager` injected at construction. `evaluate` spawns one child generator per iteration, so iteration i is reproducible in isolation and results do not depend on `n_jobs`. Individual-level streams derive from the iteration stream, never from a global RNG.
3. A shared accrual layer, not a shared engine API. Cost and utility accrual, discounting, and aggregation to `Outcomes` live in a new internal module `heormodel/models/_accrual.py`, used by microsim and DES:
   - `discount_factor(t, rate)`, continuous and per-cycle variants
   - `accrue(occupancy_or_events, payoffs, rate)` per individual
   - `aggregate(per_individual, intervention, iteration)` to `Outcomes` rows

   The engines share these helpers and the output contract, nothing else.
4. Population averaging happens inside the engine. `Outcomes` stays (intervention, iteration). Individual-level detail is an optional side channel (a `trace=` flag), never part of the analysis contract.

## Discrete-time engine (first)

```python
DiscreteTimeMicrosimEngine(
    states=("H", "S", "D"),
    transition=fn,        # fn(params, state_history, attrs, rng) -> probs over states
    payoffs=fn,           # fn(params, state, attrs) -> (cost, qaly) per cycle
    population=fn | int,  # attribute sampler fn(rng, n) -> DataFrame, or a count
    cycle_length=1.0,
    horizon=60,
    discount_cost=0.03,
    discount_effect=0.03,
    interventions={"SoC": {...}, "Tx": {...}},   # intervention-specific decision levers
    seed_manager=SeedManager(...),
    half_cycle_correction=True,
)
```

- Vectorize over individuals, loop over cycles. State is an integer vector; transitions are sampled with one `rng.random(n)` and a cumulative-probability comparison per cycle. History dependence enters through `attrs` columns (time in state, prior events), updated vectorized.
- Parallelism reuses `run_psa` unchanged. Children are spawned by iteration position, so chunking does not change streams; document this.
- Common random numbers across interventions by default (variance reduction for incremental results); `independent_streams=True` disables it.

## Continuous-time engine (second)

Same constructor shape. Instead of per-cycle `transition` probabilities, `hazards` returns competing time-to-event samplers. The engine takes the minimum, advances, and accrues continuously between events with the same `_accrual` helpers (discounting integrated between event times). No cycle grid; `horizon` truncates.

## Validation (acceptance)

- A three-state discrete-time microsimulation parameterized to match a cohort model with a closed-form solution: mean costs and QALYs converge to the analytic values as the population grows.
- The continuous-time engine on constant hazards reproduces the exponential cohort solution.
- Reproducibility: same seed gives identical `Outcomes` at any `n_jobs`; different iterations differ.
- Contract tests: iteration index preserved, balanced panel, interventions in declared order.
