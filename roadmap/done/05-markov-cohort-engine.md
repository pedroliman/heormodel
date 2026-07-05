# 5. Markov cohort engine

Implement `MarkovCohortEngine` in `heval/models/markov.py`, a cohort state-transition engine that sweeps a cohort trace across PSA iterations and emits the standard `Outcomes` schema. It shares the `heval.models._accrual` layer with the microsimulation and discrete-event engines, so discounting and aggregation stay one implementation.

## Coherence with the engine architecture

The same three commitments as items 3 and 4, the same shapes:

1. Configure once, evaluate on draws. `MarkovCohortEngine(...)` takes the states, strategies, and a `build` callback; `evaluate(draws)` returns `Outcomes` indexed by `draws.index`.
2. No hidden randomness. A cohort trace is deterministic given a parameter row, so the engine draws no random numbers; reproducibility follows from the draw matrix alone.
3. Accrual reuse. Per-cycle discount factors and the reduction to `Outcomes` rows come from `heval/models/_accrual.py`.

## Sketch

```python
MarkovCohortEngine(
    states=("H", "S", "D"),
    strategies=("SoC", "Tx"),
    build=fn,             # fn(params, strategy) -> CohortSpec
    n_cycles=60,
    start="H",
    cycle_length=1.0,
    discount_cost=0.03,
    discount_effect=0.03,
    half_cycle_correction="simpson",
)
```

`CohortSpec` carries one strategy's transition matrix (one array, or a per-cycle stack of arrays for age-varying rates) and its reward arrays: a per-state cost and effect, and optional per-transition rewards for one-time events such as the cost of dying or the disutility of onset. `build` returns one `CohortSpec` per (params, strategy) pair.

Per iteration and strategy, `evaluate` advances the state occupancy vector across `n_cycles`, accrues discounted per-state and per-transition rewards with the within-cycle correction weights, and writes one `Outcomes` row.

## Within-cycle correction

`gen_wcc(n_cycles, method)` builds the correction weight vector. Support `"simpson"` (Simpson's 1/3 rule, the default), `"half-cycle"` (the trapezoidal half-cycle correction), and `"none"`.

## Companion feature: duration groups on the microsimulation engine

Add `duration_groups` to `DiscreteTimeMicrosimEngine`: a per-individual counter of consecutive cycles spent in a named set of states. A sojourn that progresses within the group (Sick to Sicker) keeps counting, where `time_in_state` resets on the state change. This lets a microsimulation replicate a semi-Markov cohort model.

## Validation (acceptance)

- Three replications of published Sick-Sicker cost-effectiveness tutorials (cohort, time-dependent, microsimulation) reproduce the source's deterministic results, shipping as `examples/mdm_*.py` with website tutorials and a replication gallery.
- Contract and reproducibility tests identical in shape to the microsim ones.
