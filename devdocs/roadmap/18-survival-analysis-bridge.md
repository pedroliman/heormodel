# 18. Survival analysis bridge

Add the machinery that turns fitted parametric survival models into inputs the
existing engines accept: time-varying transition probabilities for `MarkovModel`,
competing event-time samplers for `MicrosimModel.continuous`, and the survival
curves the partitioned survival engine (item 19) integrates. It also carries the
uncertainty in the fitted parameters onto the iteration index, so survival
estimates flow through `run_psa` into cost-effectiveness and value-of-information
analysis unchanged.

This closes the clearest functional gap in the package comparison
([`../package-comparison.md`](../package-comparison.md)). A health economist
fitting survival curves to patient data has no way to bring the fit, or its
uncertainty, into a heormodel model. The engines can already simulate the
resulting process; the estimation-to-model step is what is missing.

## Sequencing: reproduce the reference model first, then settle the architecture

The first deliverable is a bespoke replication that reproduces the reference model
specified below with example-local functions, validated against the closed forms
also given below. Only once the numbers match do we extract a public
`heormodel.survival` layer, letting the replication code determine its shape. The
API sketches later in this note are candidates to inform that second step, not
commitments. This ordering applies across items 18, 19, and 20: reproduce all
three reference models with bespoke code, confirm the numbers, then design the
shared abstractions once, knowing what they have to carry.

## The reference model and its expected results

The acceptance target is a single overall-survival curve and the two ways a model
consumes it. It is fully specified here so the replication needs nothing external.

Survival function, Weibull in the accelerated-failure-time parameterization:

```
S(t) = exp(-(t / scale) ** shape),   shape = 1.2, scale = 6.0 years
```

A two-state alive-and-dead model in model time (the clock-forward case), annual
discount rate 0.03, and a utility of 0.85 while alive. The expected results, which
the replication must reproduce:

| Quantity | Expected value |
|---|---|
| Undiscounted life expectancy, integral of `S(t)` | 5.64394 (equals `scale * gamma(1 + 1/shape)`) |
| Discounted life expectancy, integral of `exp(-0.03 t) S(t)` | 4.92709 |
| Discounted quality-adjusted life-years, utility 0.85 | 4.18803 |
| Annual transition-to-death probabilities `1 - S(k+1)/S(k)`, cycles 0 to 4 | 0.10994, 0.14025, 0.15439, 0.16428, 0.17201 |
| Discrete annual cohort discounted life-years, trapezoidal correction | 4.93604 (matches the continuous 4.92709 within the correction error) |

The two derivations that must agree are the point of the item: the continuous
sampler integrates the curve directly, and the cohort built from the per-cycle
transition probabilities reproduces the same discounted life-years under the
half-cycle correction. A model author gets the same answer whichever engine they
use.

## Parameter recovery exercise

The reference results above use the true parameters directly. Every implementation
must also carry a parameter recovery exercise that adds the estimation step a real
analysis goes through, so the fitting adapter and the parameter-uncertainty path
are exercised end to end, not just the forward model. The steps:

1. Data-generating process. Draw event times from the reference survival curve
   (Weibull shape 1.2, scale 6.0) and apply administrative censoring at a
   follow-up horizon (for example 12 years), giving a right-censored sample of a
   typical trial size, on the order of 300 patients.
2. Estimation. Fit a Weibull survival model to the censored sample by maximum
   likelihood, recovering the shape and scale and their asymptotic covariance on
   the log scale. This is the appropriate estimator because the data-generating
   process is Weibull.
3. Probabilistic analysis. Draw parameter sets from the fitted asymptotic
   distribution (the `sample_params` path), propagate each through the two-state
   model, and report the discounted life expectancy with its credible interval.
4. Convergence, asserted as tests:
   - Parameter recovery. As the simulated sample size grows, the estimates
     converge to the data-generating parameters and their standard errors shrink
     toward zero at the rate one over the square root of the sample size. At a
     large size, on the order of 100,000 or more, the shape and scale are within a
     tight tolerance of 1.2 and 6.0.
   - Analytic convergence. At the large size, the fitted-model discounted life
     expectancy and the probabilistic-analysis mean both converge to the analytic
     value 4.92709, and the probabilistic spread shrinks toward zero. At trial
     size the analytic value lies inside the credible interval.

Illustrative behavior at a fixed seed, administrative censoring at 12 years: at 300
patients the fit recovers a shape near 1.14 and a scale near 6.45 with standard
errors near 0.05 on the log scale, a discounted life expectancy near 5.28, and a
probabilistic standard deviation near 0.25; at 200,000 patients it recovers a shape
of 1.200 and a scale of 6.01 with standard errors near 0.002, a discounted life
expectancy near 4.934, and a probabilistic standard deviation near 0.008,
converging to the analytic 4.92709. These are illustrative, not exact targets at
trial size, since the small-sample estimates depend on the seed; the tests assert
the convergence, not a specific small-sample number.

## Phase 1: bespoke replication (do this first)

Reproduce the reference table and the parameter recovery exercise with throwaway
helpers under `examples/`: evaluate the Weibull survival and hazard, sample event
times by inverse-transform sampling and recover the discounted life expectancy
through `MicrosimModel.continuous`, build the annual transition probabilities and
recover it again through a `MarkovModel` cohort, then run the recovery exercise
(generate a censored sample, fit a Weibull by maximum likelihood, and propagate the
fitted uncertainty through a probabilistic analysis). Nothing here is a public
module; the fitting, sampling, and conversion are example functions, the way the
`examples/mdm_*` replications keep bespoke logic local. Phase 1 also produces the
survival helpers that items 19 and 20 use in their own phase-1 replications, so the
survival curve code is written once and exercised by all three before any of it is
promoted.

## Phase 2: extract the architecture (after parity)

Once the replications of items 18, 19, and 20 all pass, promote the recurring
machinery into `heormodel.survival`. What follows is the candidate shape, to be
confirmed or revised by what the replications needed.

### Survival distributions (candidate)

Distribution objects for the families the field relies on: exponential, Weibull
(accelerated-failure-time and proportional-hazards parameterizations), Gompertz,
lognormal, log-logistic, generalized gamma, restricted cubic spline (the flexible
parametric form), and piecewise exponential. Each answers four questions on a time
argument: `survival(t)`, `hazard(t)` and `cumhazard(t)`, `sample_time(rng, size)`,
and `transition_probability(t0, t1)`, the conditional probability of the event in
`(t0, t1]` given survival to `t0`.

### Fitting and its uncertainty (candidate)

Fitting stays in a dedicated survival package rather than being reimplemented. The
default adapter reads a fit from `lifelines`, chosen because it covers the families
above with extrapolation and carries the same permissive license as heormodel, so
it can be an optional dependency (`heormodel[survival]`). `from_fit(fitter)` returns
a distribution at the point estimate; `sample_params(fitter, n, seed)` draws
coefficient vectors from the fit's asymptotic multivariate normal (with a bootstrap
alternative) onto the canonical iteration index, so survival uncertainty and the
other parameters share one index, the guarantee `evppi` depends on.

### From hazards to transitions (candidate)

`to_transition_matrix(dists, cycle_length)` builds a per-cycle transition array from
cause-specific hazards using the transition-intensity matrix and its matrix
exponential, so probabilities are consistent and sum to one. Stacking these across
cycles is the age-varying input `MarkovModel` already accepts. For the individual
engine, `sample_time` is the competing-times sampler `MicrosimModel.continuous`
already calls, and clock-reset (semi-Markov) falls out of the engine redrawing at
state entry.

### Curve algebra (candidate)

`apply_hazard_ratio`, `apply_acceleration_factor`, `mix(dists, weights)`, and
`splice(dist_early, dist_late, cutpoint)` return new distributions. `splice` is the
extrapolation operation: the observed curve to the cutpoint, then a parametric
tail. These compose, so a treatment arm is the comparator curve under a sampled
hazard ratio, and a long-term model is a spline fit spliced onto a
background-mortality tail.

## Coherence with the engine architecture

The layer adds no engine and touches none of the three commitments. It produces two
things the engines already consume, per-cycle transition arrays and per-iteration
draws, and uncertainty enters through the shared iteration index, so `run_psa`,
`icer_table`, `ceac`, `evpi`, and `evppi` need no special case.

## Validation (acceptance)

- Phase 1 reproduces every row of the reference table: the continuous sampler and
  the cohort both recover the discounted life expectancy, and the two agree within
  the cycle-correction error.
- The parameter recovery exercise passes its two convergence tests: the estimates
  converge to the data-generating shape 1.2 and scale 6.0 with standard errors
  shrinking like one over the square root of the sample size, and the fitted-model
  discounted life expectancy and probabilistic mean converge to 4.92709 as the
  simulated sample grows.
- Closed forms, at fixed seeds: a constant hazard `h` gives discounted life-years
  `1 / (discount + h)`, and applying a hazard ratio `r` gives `1 / (discount + r h)`,
  which the curve algebra must match exactly; an all-exponential competing-risks
  model is a continuous-time Markov chain whose expected discounted rewards solve
  `(discount * I - Q) v = r` for generator `Q` (reused across items 18 and 20);
  `transition_probability` integrates to the analytic transition matrix; and
  `sample_params` recovers the fitted mean and covariance as the draw count grows.
- Phase 2 changes no number; it re-expresses the passing replication through the
  extracted layer, and the same tests pass against the public API.

## Deliverables

- Phase 1: `examples/survival_bridge.py` reproducing the reference table and the
  parameter recovery exercise with bespoke survival helpers, its closed-form and
  convergence tests, and a replication gallery entry.
- Phase 2: `heormodel.survival` (distribution families, `lifelines` adapter,
  `to_transition_matrix`, curve algebra), the `survival` optional extra, docstring
  worked examples, tests, a website tutorial, and API reference and changelog
  entries.

## Reasonable-extent boundaries

- Performance. The reference numbers are reproduced on vectorized NumPy; parity is
  the same answers, not the same wall-clock time as a compiled implementation.
- Transitions parameterized by a fitted multinomial model need a
  regression-to-transition path for a different fitting family. It is a sibling of
  this survival bridge, kept in the backlog rather than folded in here.
- The website tutorials cite the published clinical model each example is drawn
  from, keeping the user-facing documentation free of external package names.
