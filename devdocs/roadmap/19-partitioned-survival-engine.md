# 19. Partitioned survival model engine

Implement `PartSurvModel` in `heormodel/models/partsurv.py`, a partitioned
survival engine that derives health-state occupancy from a set of survival curves
and emits the standard `Outcomes` structure. It is the standard oncology model
type both `hesim` and `heemod` provide and heormodel lacks, and it is the last
engine needed to reproduce the survival-driven `hesim` tutorials. It depends on
the survival layer of item 18 for its curves and reuses `heormodel.models._accrual`
for discounting.

## Why a dedicated engine

A partitioned survival model does not simulate transitions. It reads state
occupancy straight off survival curves: for an N-state model ordered from best to
worst health, curve `k` is the probability of being in state `k` or better. State
1 occupancy is `S_1(t)`, each middle state is `S_k(t) - S_{k-1}(t)`, and the worst
state (death) is `1 - S_{N-1}(t)`. Costs and utilities accrue on occupancy, and
the discounted area under each curve gives the state's contribution to life-years,
quality-adjusted life-years, and cost.

This is different enough from a transition model to be its own engine. It needs no
transition matrix and no simulation, only the curves and an integration. Expressing
it as a Markov model would require inverting the curves into transition
probabilities, which is exactly the modeling assumption a partitioned survival
model declines to make.

## Coherence with the engine architecture

The same three commitments as the other engines, in the deterministic shape
`MarkovModel` and `ODEModel` take:

1. Configure once, evaluate on draws. `PartSurvModel(states, interventions, curves, ...)`
   takes a `curves` callback returning one `PartSurvSpec` per (params, intervention)
   pair; `evaluate(draws)` returns `Outcomes` indexed by `draws.index`.
2. No hidden randomness. State occupancy is a deterministic function of the curves,
   so the engine draws no random numbers and satisfies `ModelEngine` alone.
   Parameter uncertainty enters through the draws, including the sampled survival
   parameters from item 18.
3. Accrual reuse. Discounting and the reduction to `Outcomes` rows come from
   `heormodel.models._accrual`; the continuous accrual (`integrate_flow`) already
   integrates a piecewise trajectory, which is what the occupancy curves are on a
   time grid.

## Sketch

```python
PartSurvModel(
    states=("PF", "PD", "Death"),   # ordered best to worst
    interventions=("SoC", "New"),
    curves=fn,                       # fn(params, intervention) -> PartSurvSpec
    horizon=30.0,
    discount_rate=0.03,
)
```

`PartSurvSpec` carries the ordered survival curves, the `N - 1` of them from item
18's survival layer (progression-free survival and overall survival for the
three-state case), the per-state cost and utility rates accrued on occupancy, and
optional one-time costs charged on entry to a state. `evaluate` builds each state's
occupancy from the successive-curve differences on a time grid over the horizon,
accrues the discounted rewards, and writes one `Outcomes` row per intervention and
iteration.

## Crossing curves

The construction requires the curves not to cross: `S_{k}(t) <= S_{k-1}(t)`. Fitted
curves can cross, giving a negative occupancy. Clamp each middle state to be
non-negative and report the largest clamp so the user sees when it happened, the
standard handling for this model type. Document that a large clamp is a sign the
curves are mis-specified.

## Validation (acceptance)

The acceptance test is reproducing the `hesim` partitioned survival tutorial
within heormodel.

- Partitioned survival models (the `hesim` "Partitioned survival models" tutorial).
  A four-state oncology model of a two-line sequential treatment strategy, three
  strategies across patient profiles that vary by age and sex, three Weibull
  curves (progression on first line, progression on second line, and mortality)
  with age, sex, and strategy covariates, state utilities, drug costs fixed by
  strategy, and medical costs by state. Fit the curves through item 18, build the
  occupancy, and reproduce the state probabilities and the cost-effectiveness
  results.

Closed-form cross-checks anchor the engine:

- Exponential curves. With exponential progression-free and overall survival at
  constant rates, the discounted life-years in each state have a closed form (for
  a single exponential curve at rate `r` with discount `d`, the discounted area is
  `1 / (r + d)`). The engine must match this, exercising the occupancy
  construction and the discounted integration.
- The occupancy curves sum to one at every time on the grid, and each is
  non-negative after clamping.
- Contract tests identical in shape to the `MarkovModel` ones: the returned
  iteration index matches the draws.

As with the other replications, sampled-population numbers match the source within
Monte Carlo error, and the tutorial prose states where the replication sits
relative to the source.

## Deliverables

- `heormodel.models.PartSurvModel` and `PartSurvSpec`, exported from
  `heormodel.models`, with docstring worked examples and tests, and a `trace`
  parallel to `MarkovModel.trace` and `ODEModel.trajectory` returning the
  occupancy curves.
- `examples/hesim_psm.py` and a website replication tutorial reproducing the
  partitioned survival model at reduced sizes.
- A replication gallery entry, changelog entry, `docs/concepts/engines.qmd` update
  describing the `curves` callback the engine expects, and API reference entries.

## Relationship to item 18 and the hesim parity gallery

This item depends on item 18: its curves are that layer's survival distributions,
and its parameter uncertainty is that layer's `sample_params`. Together the two
items reproduce the survival-driven `hesim` tutorials (multi-state and partitioned
survival), and the cohort, time-inhomogeneous cohort, and cost-effectiveness
tutorials already reproduce with the shipped `MarkovModel` and `heormodel.cea`. A
"hesim parity" section of the replication gallery collects these, giving a
concrete, checkable statement of functional parity. The excluded tutorials
(performance benchmarks and multinomial-logistic-regression transitions) are noted
in item 18's reasonable-extent boundaries.
