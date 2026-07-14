# 20. Multi-state and semi-Markov models from fitted transition hazards

Add first-class support for multi-state models whose transitions are parameterized
by fitted survival hazards, one distribution per allowed transition, simulated on
the continuous clock. This is the individual-level modeling style at the center of
survival-based cost-effectiveness analysis: a transition structure, a survival model
per transition, and a choice between the clock-forward (Markov) and clock-reset
(semi-Markov) timing of those hazards. It builds on the survival layer of item 18
and runs on the existing `MicrosimModel.continuous`.

## Why a multi-state layer, given the continuous engine already exists

`MicrosimModel.continuous` already samples a time to each competing destination,
takes the earliest, and redraws at state entry, so a semi-Markov process is
expressible today. What a multi-state model adds is structure that recurs and is
error-prone to hand-assemble each time:

1. An allowed-transition matrix over the states, so the model function samples only
   permitted destinations and the state probabilities are read off a known
   structure.
2. One survival distribution per transition, with per-transition covariates, from
   item 18's layer.
3. An explicit clock choice. Clock-forward measures each hazard from time zero
   (model time); clock-reset measures it from the last state entry. The two give
   materially different results in a state a patient can dwell in, so the choice
   must be a named, tested option rather than an implementation detail.
4. State probabilities over time, the multi-state analog of a cohort trace, which
   `heormodel.models.state_occupancy` already derives from an event log.

## Sequencing: reproduce the reference model first, then settle the architecture

As with items 18 and 19, the bespoke replication comes first. Reproduce the
reference model below with example-local functions on `MicrosimModel.continuous`
using item 18's bespoke survival helpers, confirm the numbers for both clocks, and
only then decide whether the structure above deserves a dedicated `MultiStateModel`
convenience or is better left as a documented pattern plus a few helpers.

## The reference model and its expected results

A reversible illness-death model with three states, healthy, sick, and dead, and
four transitions.

```
Transitions and baseline hazards (per year):
  healthy -> sick    : exponential, rate 0.15
  healthy -> dead    : exponential, rate 0.02
  sick    -> healthy : exponential, rate 0.30
  sick    -> dead    : exponential, rate 0.10
```

Two strategies: the comparator, and a new intervention that multiplies the two
hazards out of healthy (to sick and to dead) by a hazard ratio of 0.70. State
utilities are 1.0 healthy and 0.7 sick; state costs are 2000 and 8000 per year; the
annual discount rate is 0.03; every patient starts healthy.

With all transitions exponential the process is a continuous-time Markov chain, so
the clock-forward and clock-reset timings coincide and the expected discounted
rewards have a closed form: `(discount * I - Q) v = r` for generator `Q` and reward
rate vector `r`. The expected results:

| Strategy | Discounted quality-adjusted life-years | Discounted cost |
|---|---|---|
| Comparator | 13.049 | 50244 |
| New | 15.459 | 52195 |

The semi-Markov variant replaces the sick-to-dead transition with a Weibull
clock-reset hazard (shape 1.5, scale 8.0 years), measured from entry to the sick
state, holding the other three transitions and all rewards fixed. There is no closed
form; the acceptance target is that the two clocks now diverge, which is the reason
the clock choice exists:

| Variant | Comparator quality-adjusted life-years | New quality-adjusted life-years |
|---|---|---|
| Clock-reset (semi-Markov) | 13.57 | 15.94 |
| Clock-forward (Markov) | 10.51 | 12.57 |

## Parameter recovery exercise

The reference results use the true transition hazards directly. Every implementation
must also carry a parameter recovery exercise that fits the per-transition hazards
from simulated multi-state trial data, so the estimation and parameter-uncertainty
path is exercised alongside the forward model. The steps:

1. Data-generating process. Simulate a typical trial-size cohort, on the order of
   300 patients, through the reference illness-death model, recording each patient's
   transition times and the states occupied, with administrative censoring at a
   follow-up horizon, for example 30 years. This yields, per transition, a risk set
   (the time each patient spends at risk in the origin state) and the observed
   events.
2. Estimation. Fit one survival model per transition by maximum likelihood on that
   transition's risk set: an exponential for the three exponential transitions and a
   Weibull, measured from entry to the sick state, for the sick-to-dead transition in
   the semi-Markov variant. For an exponential transition the maximum-likelihood
   rate is the number of observed events divided by the total time at risk in the
   origin state, with standard error the rate over the square root of the event
   count. Recover the rates or the shape and scale and their standard errors.
3. Probabilistic analysis. Draw parameter sets from the fitted distributions,
   reassemble the per-transition hazards, simulate, and report the discounted
   quality-adjusted life-years and costs for both strategies with credible intervals.
4. Convergence, asserted as tests:
   - Parameter recovery. As the simulated sample size grows, the recovered rates
     converge to the data-generating 0.15, 0.02, 0.30, and 0.10, with standard
     errors shrinking like one over the square root of the event count.
   - Analytic convergence. At a large sample size, on the order of 100,000 or more,
     the exponential-variant discounted quality-adjusted life-years converge to the
     closed form, 13.049 for the comparator and 15.459 for the new intervention,
     and the probabilistic spread toward zero. At trial size the closed-form values
     lie inside the credible intervals.

## Phase 1: bespoke replication (do this first)

Reproduce both tables and the parameter recovery exercise with example-local
functions under `examples/`: assemble the per-transition hazards with item 18's
bespoke helpers, sample competing times on `MicrosimModel.continuous`, and compute
the discounted quality-adjusted life-years and costs for both strategies. Reproduce
the exponential case against its closed form, then the semi-Markov case for both
clocks, showing the clock-forward and clock-reset results differ as the second table
gives, then run the recovery exercise (simulate the censored multi-state cohort, fit
one model per transition by maximum likelihood, and propagate the fitted uncertainty
through a probabilistic analysis).

## Phase 2: extract the architecture (after parity)

Once phase 1 passes (with items 18 and 19), decide the shape. Candidate: a
`MultiStateModel` in `heormodel/models/multistate.py` that takes the states, an
allowed-transition matrix, an `interventions` sequence, a `transitions` callback
returning one survival distribution per allowed transition, and a `clock` argument
(`"forward"` or `"reset"`). It assembles the competing-times sampler and reuses
`MicrosimModel.continuous` underneath rather than being a separate simulation
kernel, so there is one continuous-time engine and this layer is a structured way to
build its model function. State probabilities come from `state_occupancy` on the
event log, already shipped. The alternative, if the replication shows the assembly
is thin, is a documented pattern with a transition-structure helper rather than a
new class. The replication decides.

## Coherence with the engine architecture

No new simulation kernel and no new output structure. The layer builds a model
function for `MicrosimModel.continuous`, whose randomness already comes from the run
loop's per-iteration streams, so results stay invariant to worker count and batch
size. Survival uncertainty enters through item 18's `sample_params` on the shared
iteration index, and `state_occupancy` supplies the state-probability trace.

## Validation (acceptance)

- Phase 1 reproduces the exponential table within Monte Carlo error (the closed form
  is exact) and the semi-Markov table for both clocks, with clock-forward and
  clock-reset differing in the sick state.
- The parameter recovery exercise passes its convergence tests: the recovered rates
  converge to 0.15, 0.02, 0.30, and 0.10, and the fitted-model exponential-variant
  quality-adjusted life-years converge to the closed form as the simulated sample
  grows.
- Closed form: the exponential illness-death model solves `(discount * I - Q) v = r`,
  which the simulation must match, exercising the transition assembly and both
  clocks (the clocks coincide when every hazard is constant, a check that clock-reset
  reduces correctly).
- Phase 2 changes no number; whatever shape is chosen reproduces the phase-1 result,
  with contract tests confirming the iteration index matches the draws.

## Deliverables

- Phase 1: `examples/multistate.py` reproducing both reference tables for both
  clocks and the parameter recovery exercise with bespoke functions, its closed-form
  and convergence tests, and a replication gallery entry.
- Phase 2: whichever the replication justifies, a `MultiStateModel` convenience or a
  documented pattern with a transition-structure helper, with docstring worked
  examples, tests, a website tutorial, and API reference and changelog entries.

## Relationship to items 18 and 19

The three items share one survival layer (item 18) and one continuous-time engine.
Item 20 is the individual-level multi-state modeling style, item 19 is the
partitioned survival style, and item 18 is the estimation machinery both rest on.
Their phase-2 architecture is settled together, after the replications confirm what
the abstractions must carry.
