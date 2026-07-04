# 2 — Microsimulation engine

**Goal:** implement `DiscreteTimeMicrosimEngine` first, then
`ContinuousTimeMicrosimEngine`, replacing the stubs in
`heval/models/microsim.py`. Both simulate an individual-level population
per PSA iteration and emit the standard `Outcomes` schema.

## Architectural commitments

These are the decisions the DES engine (design note 03) must stay
coherent with, so they are fixed here:

1. **An engine is configured once, then evaluated on draws.** The
   constructor takes the model structure; `evaluate(draws)` takes only the
   parameter draw matrix and returns `Outcomes` with `draws.index` as the
   iteration index (the existing protocol, unchanged).
2. **Randomness comes from a `SeedManager` injected at construction.**
   `evaluate` spawns one child generator per PSA iteration
   (`seed_manager.spawn(len(draws))`), so iteration i is reproducible in
   isolation and results are invariant to `n_jobs`. Individual-level
   streams are derived from the iteration stream, never from a global RNG.
3. **A shared accrual layer, not a shared engine API.** Cost/utility
   accrual, discounting, and aggregation to `Outcomes` live in a new
   internal module `heval/models/_accrual.py` used by microsim *and* DES:
   - `discount_factor(t, rate)` (continuous and per-cycle variants)
   - `accrue(events_or_occupancy, payoffs, rate) -> per-individual totals`
   - `aggregate(per_individual, strategy, iteration) -> Outcomes rows`
   The engines share these helpers and the output contract — nothing else.
4. **Per-iteration population averaging happens inside the engine.**
   `Outcomes` stays (strategy, iteration); individual-level detail is an
   optional side channel (`engine.last_trajectories` or a `trace=` flag),
   never part of the analysis contract.

## Discrete-time engine (first)

```python
DiscreteTimeMicrosimEngine(
    states=("H", "S", "D"),
    transition=fn,        # fn(params: pd.Series, state_history, attrs, rng) -> probs over states
    payoffs=fn,           # fn(params, state, attrs) -> (cost, qaly) per cycle
    population=fn | int,  # attribute sampler fn(rng, n) -> DataFrame, or a plain count
    cycle_length=1.0,
    horizon=60,
    discount_cost=0.03,
    discount_effect=0.03,
    strategies={"SoC": {...}, "Tx": {...}},   # strategy-specific overrides passed to fns
    seed_manager=SeedManager(...),
    half_cycle_correction=True,
)
```

- **Vectorise over individuals, loop over cycles.** State is an integer
  vector of length n_individuals; transitions are sampled with one
  `rng.random(n)` + cumulative-probability comparison per cycle. History
  dependence enters through `attrs` columns (time-in-state, prior events)
  updated vectorised.
- **Parallelism** reuses `run_psa`: the engine's `evaluate` is cheap to
  call on a chunk of draws, so joblib chunking works unchanged. Document
  the seeding interaction (children are spawned by iteration *position*,
  so chunking does not change streams).
- **Common random numbers across strategies** (variance reduction for
  incremental results): evaluate all strategies for one individual from
  the same stream by default; expose `independent_streams=True` to
  disable.

## Continuous-time engine (second)

Same constructor shape; instead of `transition` probabilities per cycle,
`hazards` returns competing time-to-event samplers
(`fn(params, state, attrs, rng) -> dict[event, time]`); the engine takes
the minimum, advances, and accrues continuously between events using the
same `_accrual` helpers (integrated discounting between event times).
No cycle grid; `horizon` still truncates.

## Validation (acceptance bar)

- A three-state discrete-time microsimulation whose parameters make it
  equivalent to a cohort model with a known closed-form solution: mean
  costs/QALYs converge to the analytic values as the population grows
  (statistical tolerance scaled by 1/sqrt(n)).
- Continuous-time engine on constant hazards reproduces the exponential
  cohort solution.
- Reproducibility: same `SeedManager` seed → identical `Outcomes`
  regardless of `n_jobs`; different iterations differ.
- Contract tests: iteration index preserved; balanced panel; strategies
  in declared order.
