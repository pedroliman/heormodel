# Reconciling the discrete-event Sick-Sicker replication with the published figures

This note records how the `examples/mdm_des` package and its tutorial reproduce the published figures of Lopez-Mendez, Goldhaber-Fiebert, and Alarid-Escudero (Medical Decision Making 2026;46(5):533-548), and where the replication still differs from the companion R code. The design note [`roadmap/13-des-sicksicker-replication.md`](../roadmap/13-des-sicksicker-replication.md) states the modeling intent; this note is the reconciliation against the companion source code, read from the authors' public repository.

Reproducing the figures means matching the companion code rather than the Table 1 specification, because on two points the companion code departs from its own Table 1, and those two points move the published figures. The replication reproduces both. A third difference, the random-number seeding, is left as the framework's per-iteration reproducibility guarantee; it changes the Monte Carlo error structure, not the expected figures.

The base-case cost-effectiveness analysis and every epidemiological outcome (survival, prevalence, dwell times) depend only on the transition dynamics and the state flows, which the two implementations share, so they matched before any of the adjustments below.

## Matched: two companion behaviors that move the published figures

### 1. Transition amounts accrue over the preceding sojourn

Table 1 defines `ic_HS1`, `ic_D`, and `du_HS1` as one-time amounts ("increase in cost when transitioning"). The companion `cea_fn` does not pay them once. It adds each to the annual flow of the sojourn that ends in the transition, then multiplies by the discounted length of that sojourn:

```r
dt_all[, Total_cost_ep := Annual_Cost_ep + Annual_Cost_trt + Cost_trans]
dt_all[, Disc_total_cost := Total_cost_ep * v_dwc_t12]   # v_dwc_t12 = discounted sojourn length
```

So a $2,000 cost of dying enters as $2,000 per year over the final sojourn, which averages about 14 years in Sicker, and a $1,000 onset cost enters as $1,000 per year over the Healthy sojourn before onset.

The replication reproduces this from the event history rather than through an engine feature, since the convention is specialized to this cost function. After the run, `transition_costs_and_utilities` in `examples/mdm_des/transitions.py` takes `evaluate(trace="events")`, and for every event multiplies the one-time amount by the discounted sojourn integral over `[start, time]`, where the start is the individual's previous event time, then averages per individual and adds the result to the intervention's cost and effect. A continuous-time Markov chain closed form in `tests/test_mdm_des.py` confirms the arithmetic: an amount that accrues over the sojourn enters state `i`'s effective flow as the amount times its transition rate over the state's total exit rate.

Effect: expected costs sit about $20,000 per intervention above a once-at-event accounting (the dominant term is the death cost accrued over the long Sicker sojourn). The base case runs about $128,000 for standard of care up to $285,000 for intervention AB, matching the published figure 4A axis; a once-at-event accounting would run about $107,000 to $265,000.

### 2. Six Table 1 parameters are held at their base case

The companion draws its parameter sets with `generate_psa_params_DES`, then merges each row into the simulation and cost lists with `update_param_list`, which is `modifyList` (exact-name matching). Six drawn columns have names the target lists do not use, so `modifyList` adds them as unread entries and leaves the base-case values in force:

| Drawn column (PSA) | Name the model reads | Base value held |
|---|---|---|
| `r_S1S2_scale` | `r_S1S2_scale_ph` (simulation) | 0.08 |
| `c_trtA` | `dc_trtA` (cost list) | $12,000 |
| `c_trtB` | `dc_trtB` (cost list) | $13,000 |
| `ic_HS1` | `dc_HS1` (cost list) | $1,000 |
| `ic_D` | `dc_D` (cost list) | $2,000 |
| `u_trtA` | `du_trtA` (cost list) | 0.20 increment |

The Weibull scale is the consequential one: it carries the progression uncertainty that the bivariate lognormal in `generate_psa_params_DES` was built to express, correlated with the shape. Because only the shape varies, the progression hazard barely moves across draws, and the treatment-B comparison that hinges on it carries little parameter uncertainty. The treated Sick utility is a subtler case: the companion applies it as `u_S1 + du_trtA` with `du_trtA` fixed at 0.20, so the drawn `u_trtA` is never read and the treated utility varies only through the shared `u_S1`.

The replication holds all six fixed. In `parameter_set`, `r_S1S2_scale`, `c_trtA`, `c_trtB`, `ic_HS1`, and `ic_D` are `Fixed` at their base values, and the `state_costs_and_utilities` function models the treated Sick utility as `u_S1 + 0.20` rather than a drawn `u_trtA`. With the scale fixed, only the Weibull shape varies, so no scale-shape correlation applies.

Effect: the acceptability curves are near step functions and the expected value of perfect information (EVPI) peaks are small (published figure 4D reports about $2,600 and $750 per person), because the intervention comparisons carry little uncertainty. Drawing every Table 1 distribution instead would raise the decision uncertainty and the EVPI several times over.

## The one remaining difference: random-number seeding

The companion resets to one fixed seed (`set.seed(2)`) inside every probabilistic iteration, so all 1,000 parameter sets are driven by the same random-number stream. This is a variance-reduction choice: it correlates the Monte Carlo error across parameter draws, which sharpens the cost-effectiveness scatter and the acceptability curves. `heormodel.run.run_psa` instead seeds each iteration from its index through the `SeedManager`, so iteration results do not depend on how a run is chunked across workers, and applies common random numbers across interventions within an iteration only. The replication keeps the framework's design.

Effect: the two probabilistic analyses have different Monte Carlo error structure, not different expected figures. Neither is biased; the companion's clouds are visually tighter across the parameter draws at a given simulation size. A larger population per draw shrinks the difference, so the replication matches the published figure 4 within Monte Carlo error.

## Method differences that do not materially move the figures

### Mortality sampling: exact inversion, not discrete-year with a uniform jitter

The companion samples the age at death by binning the cumulative hazard to integer ages (`nps_nhppp` with `correction = "uniform"`), then adds a uniform draw within the year. `LifeTable.sample_time_to_death` inverts the piecewise-constant cumulative hazard exactly. Both target the same continuous distribution, since the life table is piecewise constant by integer age; the companion's is a within-year approximation of the exact inversion. The difference in sampled ages at death is under a year and averages out.

### Latin hypercube sampling is in the prose, not the code

The article text describes Latin hypercube sampling for the parameter sets. The companion `generate_psa_params_DES` draws each parameter independently with `rgamma`, `rbeta`, `rlnorm`, and a bivariate-lognormal routine, with no stratification. The replication also uses plain Monte Carlo through `ParameterSet.sample`, so it matches the companion code rather than the article's prose. `heormodel.params` has no Latin hypercube sampler.

### Correlated Weibull draw is moot under the held-fixed scale

Table 1 specifies the Weibull scale and shape as a bivariate lognormal with correlation 0.5. Because the replication holds the scale fixed to match the companion (matched behavior 2), only the shape varies and the correlation never enters. Were the scale drawn, `heormodel.params` would reproduce the bivariate lognormal through two lognormal marginals with a Spearman rank correlation of 0.5, up to the rank-to-linear conversion.
