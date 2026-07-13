# 18. Survival analysis bridge

Implement `heormodel.survival`, a layer that turns fitted parametric survival
models into inputs the existing engines accept: time-varying transition
probabilities for `MarkovModel`, competing event-time samplers for
`MicrosimModel.continuous`, and the survival curves the partitioned survival
engine (item 19) integrates. It also carries the uncertainty in the fitted
parameters onto the iteration index, so survival estimates flow through
`run_psa` and into cost-effectiveness and value-of-information analysis unchanged.

This is the single clearest gap the package comparison identified
([`../package-comparison.md`](../package-comparison.md)): a health economist
fitting survival curves to patient data has no way to bring the fit, or its
uncertainty, into a heormodel model. The engines can already simulate the
resulting process. What is missing is the estimation-to-model step that `hesim`
and `heemod` provide.

## Why a survival layer rather than ad hoc code

The continuous microsimulation clock already samples a time to each competing
destination, takes the earliest, and redraws at state entry, so a semi-Markov
process is expressible today. But the sampled times come from whatever a model
author writes by hand. Three things are missing and recur in every survival
model, so they belong in one place:

1. Parametric survival distributions with the standard health-technology-assessment
   families, each able to sample an event time, evaluate a hazard, and convert to
   a transition probability over a cycle.
2. A path from a fit to those distributions, including the fit's uncertainty as
   draws on the iteration index. Without this, survival parameters enter as fixed
   numbers and the probabilistic analysis understates decision uncertainty.
3. Curve algebra: apply a hazard ratio, apply an acceleration factor, mix two
   curves, and splice one curve onto another past a cutpoint. Extrapolation
   beyond the observed follow-up is the reason survival models exist in this
   field, and it is these operations.

## The survival distributions

`heormodel.survival` provides distribution objects for the families the field
relies on: exponential, Weibull (accelerated-failure-time and proportional-hazards
parameterizations), Gompertz, lognormal, log-logistic, generalized gamma,
restricted cubic spline (the flexible parametric form), and piecewise exponential.
Each carries its parameters and answers four questions on a time argument:

- `survival(t)`, the probability of remaining event-free to `t`.
- `hazard(t)` and `cumhazard(t)`, for combining causes and for diagnostics.
- `sample_time(rng, size)`, an event time by inverse-transform sampling from the
  cumulative hazard, the sampler the continuous clock calls.
- `transition_probability(t0, t1)`, the conditional probability of the event in
  `(t0, t1]` given survival to `t0`, the quantity a cohort model needs per cycle.

The parameterizations match the field's conventions so a coefficient set from a
fit is usable directly.

## Fitting and its uncertainty

Fitting itself stays in a dedicated survival package rather than being
reimplemented. The default adapter reads a fit from `lifelines`, chosen because it
covers the parametric families above with interpolation and extrapolation and is
distributed under the same permissive license as heormodel, so it can be an
optional dependency (`heormodel[survival]`) without a license conflict. Bayesian
fitting and machine-learning survival models stay out of scope; a user who wants
them supplies the resulting parameter draws directly.

Two constructors:

- `from_fit(fitter)` reads the fitted coefficients and their estimated covariance
  from a `lifelines` fit and returns a survival distribution at the point
  estimate.
- `sample_params(fitter, n, seed)` draws `n` coefficient vectors from the fit's
  asymptotic multivariate normal (with a bootstrap alternative), transforms them
  to the natural parameterization, and returns a draw matrix on the canonical
  iteration index. These columns join a `ParameterSet` sample so that survival
  uncertainty and the other parameters share one iteration index, the guarantee
  `evppi` depends on.

The model author then reads a per-iteration survival distribution inside the
model function, exactly as they read a scalar parameter today.

## From hazards to transitions

Two directions, matching the two engine styles:

- Cohort (for `MarkovModel`). `to_transition_matrix(dists, cycle_length)` builds a
  per-cycle transition array from a set of cause-specific hazards. For competing
  risks out of a state it uses the transition-intensity matrix and its matrix
  exponential over the cycle, so the probabilities are consistent and sum to one,
  the clock-forward (Markov) construction. Stacking these arrays across cycles is
  the age-varying input `MarkovModel` already accepts, so a fitted survival model
  becomes a time-inhomogeneous cohort model with no engine change.
- Individual (for `MicrosimModel.continuous`). The distributions' `sample_time`
  is the competing-times sampler the continuous clock already calls. Semi-Markov
  (clock-reset) falls out of the engine redrawing at state entry; Markov
  (clock-forward) is expressed by sampling conditional on time already elapsed.

## Curve algebra

`apply_hazard_ratio(dist, hr)`, `apply_acceleration_factor(dist, af)`,
`mix(dists, weights)`, and `splice(dist_early, dist_late, cutpoint)` return new
distributions. `splice` is the extrapolation operation: follow the observed curve
to the cutpoint, then a parametric tail. These compose, so a treatment arm is the
comparator curve under a sampled hazard ratio, and a long-term model is a spline
fit spliced onto a background-mortality tail.

## Coherence with the engine architecture

The layer adds no new engine and touches none of the three commitments. It
produces two things the engines already consume: per-cycle transition arrays and
per-iteration draws. Uncertainty enters through the shared iteration index, so
`run_psa`, `icer_table`, `ceac`, `evpi`, and `evppi` work on a survival-driven
model with no special case.

## Validation (acceptance)

The acceptance test is reproducing the survival-driven `hesim` tutorials within
heormodel, to the extent reasonable. This item owns two of them; item 19 owns the
partitioned survival tutorial, and the cohort and cost-effectiveness tutorials
already reproduce with the shipped engines (see the gallery note below).

- Multi-state models (the `hesim` "Markov and semi-Markov multi-state models"
  tutorial). The reversible illness-death model with three states (healthy, sick,
  death) and four transitions, a Weibull distribution fitted per transition with a
  treatment covariate, simulated over a heterogeneous population. Reproduce both
  the clock-reset (semi-Markov) and clock-forward (Markov) variants through
  `MicrosimModel.continuous`, sampling transition times from the survival layer,
  and compute the same state probabilities, quality-adjusted life-years, costs,
  and cost-effectiveness summary. The clock-reset and clock-forward results must
  differ in the sick state, the point the source tutorial makes.
- Time-inhomogeneous individual-level models (the `hesim` tutorial of that name).
  The same machinery with model-time hazards, reproduced through the continuous
  clock.

Closed-form cross-checks anchor the layer independent of the comparison:

- An all-exponential multi-state model is a continuous-time Markov chain whose
  expected discounted costs and effects solve `(discount * I - Q) v = r` for
  generator `Q`. The survival layer plus `MicrosimModel.continuous` must match
  this within Monte Carlo error, exercising the sampler and the intensity-matrix
  path. This reuses the check already anchoring the discrete-event replication
  (item 13).
- `transition_probability` integrates to the analytic transition matrix, and
  `sample_time` reproduces the analytic mean event time, per distribution.
- `sample_params` recovers the fitted mean and covariance as the draw count grows.

Numbers from a replicated tutorial are expected to match the source within Monte
Carlo error, not to the digit, because the population is sampled. The tutorial
prose states where the replication's numbers sit relative to the source, the
standard the replication gallery already follows.

## Deliverables

- `heormodel.survival`: the distribution families, the `lifelines` adapter
  (`from_fit`, `sample_params`), `to_transition_matrix`, and the curve algebra,
  each with a docstring worked example and tests, and `survival` as an optional
  dependency extra.
- `examples/hesim_mstate.py` (or a folder if it grows) and a website replication
  tutorial reproducing the multi-state model at a reduced population size, plus a
  time-inhomogeneous individual-level example.
- A "survival cost-effectiveness" replication gallery entry, changelog entry, and
  API reference entries.

## Reasonable-extent boundaries

Three parts of the `hesim` tutorial set sit outside this item, stated so the
"to the extent reasonable" scope is explicit:

- Performance. `hesim`'s benchmarks tutorial measures its C++ throughput.
  heormodel reproduces the models and their numbers on vectorized NumPy, not the
  raw speed; parity means the same answers, not the same wall-clock time.
- Multinomial-logistic-regression transitions (the `hesim` "Markov models with
  multinomial logistic regression" tutorial) need a regression-to-transition
  path for a different fitting family. It is a sibling of this survival bridge and
  a natural follow-on, deferred rather than folded in here.
- The website replication tutorials cite the published clinical model each example
  is drawn from, not the R package, matching how the existing replications cite
  their source articles and keeping the user-facing documentation free of external
  package names.
