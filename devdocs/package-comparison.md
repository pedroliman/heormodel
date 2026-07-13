# Feature comparison with health economic modeling packages

This note positions `heormodel` against the established R packages a health
economist would otherwise reach for. It exists to guide the roadmap: it shows
where `heormodel` already matches the field, where it leads (one Python package
spanning model building through value of information), and where the R ecosystem
still does more.

The five comparators split into two groups by what they do. `hesim` and `heemod`
build and simulate decision models, as `heormodel` does; `heemod` focuses on
Markov cohort models and delegates value of information to other tools.
`dampack`, `BCEA`, and `voi` analyze the outputs of a model someone else built:
they consume a probabilistic sensitivity analysis (PSA) sample of costs and
effects and never simulate a state-transition or event model themselves. Reading
each cell against that split explains most of the pattern below.

The comparison reflects the documentation of each package read in July 2026:
the reference index, README, and package description of `dampack`
(version 1.0.2), `hesim`, `BCEA` (version 2.4.83), `voi`, and `heemod`. Where a
package delegates a capability to another (for example `BCEA` calls `voi` for the
expected value of partial perfect information, and `heemod` exports to `SAVI` or
calls `BCEA` for value of information), the cell says so.

## Scope and foundations

| Feature | heormodel | dampack | hesim | BCEA | voi | heemod |
|---|---|---|---|---|---|---|
| Language | Python 3.11+ | R | R | R | R | R |
| Primary role | Build models and analyze their outputs, end to end | Analyze model outputs (PSA and deterministic) | Build and simulate models, then analyze | Analyze Bayesian model outputs | Analyze model outputs (value of information only) | Build and simulate Markov models, then analyze (delegates value of information) |
| Builds decision models | Yes, four engines (below) | No, consumes a user model function or PSA table | Yes, four model classes (below) | No, consumes posterior cost and effect samples | No, consumes PSA samples or a model function | Yes, Markov cohort models, including partitioned survival |
| Performance backend | NumPy and SciPy, parallel over cores | Base R and `ggplot2` | C++ via `Rcpp`, `data.table`, built for large individual-level runs | Base R, vectorized over MCMC draws | Base R, regression backends (`mgcv`, others) | Base R, optional cluster parallelism (`use_cluster`) |
| License | MIT | GPL-3 | GPL-3 | GPL-3 | GPL-3 | GPL-3 |

## Model-building engines

`heormodel`, `hesim`, and `heemod` build models. The other three analyze
whatever costs and effects you hand them, so their cells here are "no" by design,
not by omission.

| Feature | heormodel | dampack | hesim | BCEA | voi | heemod |
|---|---|---|---|---|---|---|
| Markov cohort state-transition | Yes, `MarkovModel`: constant or per-cycle transition arrays, per-state and per-transition rewards, `trace()` for occupancy | No | Yes, `CohortDtstm`: discrete-time cohort transitions, time-homogeneous or time-inhomogeneous | No | No | Yes, `define_transition` / `define_state` / `run_model`: time-homogeneous, time-inhomogeneous, and state-time (semi-Markov) |
| Microsimulation (discrete-time individual) | Yes, `MicrosimModel.discrete`: individual population per iteration, common random numbers, `duration_groups` | No | Partly, individual simulation is continuous-time (`IndivCtstm`) rather than a discrete cycle grid | No | No | No, cohort only |
| Individual continuous-time state transition | Yes, `MicrosimModel.continuous`: continuous clock | No | Yes, `IndivCtstm`: continuous-time, Markov and semi-Markov, the package's flagship engine | No | No | No |
| Discrete-event simulation | Yes, `DESModel`: wraps SimPy, event log, `queue_waits` for queueing reports, per-entity discounted accrual | No | No | No | No | No |
| Compartmental transmission (ordinary differential equations) | Yes, `ODEModel`: integrates a user system with `solve_ivp`, force-of-infection coupling, flow-event costs, susceptible-exposed-infectious-recovered example | No | No | No | No | No |
| Stochastic compartmental (Gillespie) | Planned (roadmap item 16), not yet shipped | No | No | No | No | No |
| Partitioned survival model | No | No | Yes, `Psm` and `PsmCurves`: N-state partitioned survival from fitted survival curves | No | No | Yes, `define_part_surv`: from progression-free and overall survival curves |
| Decision tree | No | No | No | No | No | No |
| Life table / age-dependent mortality | Yes, `LifeTable` samples age-dependent mortality | No | Partly, via time-inhomogeneous transitions | No | No | Partly, `get_who_mr` pulls World Health Organization mortality rates and time-varying transitions |
| Within-cycle correction | Yes, Simpson's 1/3, half-cycle, or none | No | Yes, via time steps and `time_intervals()` | No | No | Yes, half-cycle correction |
| Bring your own model outputs | Yes, `Outcomes.from_tidy` / `from_wide` / `as_outcomes` accept an external results table | Yes, `make_psa_obj` wraps an external PSA table | Partly, model classes expect hesim's own structures | Yes, `bcea()` takes external cost and effect matrices | Yes, `evppi()` and `evsi()` take an external PSA sample | No, it builds the model rather than wrapping external outcomes |

## Parameters and uncertainty

| Feature | heormodel | dampack | hesim | BCEA | voi | heemod |
|---|---|---|---|---|---|---|
| Probability distributions for parameters | Yes, `Beta`, `Gamma`, `LogNormal`, `Normal`, `Uniform`, `Dirichlet`, `Fixed` | Yes, samples via `gen_psa_samp`; parameter helpers `beta_params`, `gamma_params`, `lnorm_params`, `dirichlet_params` | Yes, `define_rng` with `beta_rng`, `gamma_rng`, `dirichlet_rng`, `lognormal_rng`, `multi_normal_rng`, and others | No, consumes posterior draws produced upstream | No, consumes PSA draws produced upstream | Yes, `define_psa` with `normal`, `lognormal`, `gamma`, `beta`, `binomial`, `multinomial`, `logitnormal`, `triangle`, `poisson` |
| Correlated sampling | Yes, correlated draws in `ParameterSet.sample` | Partly, multivariate normal only | Yes, `multi_normal_rng` and bootstrapping | No | No | Yes, `define_correlation` |
| Probabilistic sensitivity analysis execution | Yes, `run_psa` is the single execution point | Yes, `run_psa` runs a user function over a PSA sample | Yes, propagates PSA through the simulation | No, expects the PSA already run | No, expects the PSA already run | Yes, `run_psa` over the defined model |
| Deterministic sensitivity analysis | Yes, `dsa.one_way`, `one_at_a_time`, `grid`, feeding the same runner | Yes, `run_owsa_det`, `run_twsa_det`, `owsa`, `twsa` (also as PSA metamodels) | No dedicated deterministic sensitivity analysis functions | Partly, `struct.psa` for structural uncertainty | No | Yes, `define_dsa`, `run_dsa` (one-way and multi-way) |
| Tornado diagram | Yes, `tornado_data` and `plot_tornado` | Yes, `owsa_tornado` | No | Yes, `info.rank` is an information-value tornado, not a one-way tornado | No | Yes, from `run_dsa` |
| Two-parameter grid / heatmap | Yes, `dsa.grid` and `heatmap_data` | Yes, `twsa` and `plot.twsa` | No | No | No | Partly, multi-way `run_dsa`, no dedicated heatmap |
| Parallel execution | Yes, `run_psa` uses all cores by default, results invariant to worker count | No | Yes, C++ backend built for speed on large runs | No | Partly, some regression methods parallelize | Yes, `use_cluster` distributes the PSA |
| Progress and time-remaining display | Yes, `run_psa` reports completed work and estimated time remaining | No | No | No | No | No |
| Reproducible seeding across parallel runs | Yes, `SeedManager` keys streams by iteration so results do not depend on `n_jobs` | No | Partly, standard R seeding | No | No | Partly, standard R seeding |

## Parameter estimation and survival analysis

`heormodel` takes parameters as given (from distributions, external draws, or its
own calibration) and does not fit survival curves in-package. The R ecosystem
does more here: `hesim` and `heemod` consume fitted survival and regression
models, and the fitting itself is done by `flexsurv` and `survHE` (see the note
after the table). This is the clearest area where R leads.

| Feature | heormodel | dampack | hesim | BCEA | voi | heemod |
|---|---|---|---|---|---|---|
| Distribution parameters from summary statistics (method of moments) | Yes, mean/SE constructors on distributions | Yes, `beta_params`, `gamma_params`, `lnorm_params`, `dirichlet_params` | Yes, `mom_beta`, `mom_gamma` | No | No | Partly, direct distribution specs and probability-conversion helpers (`rate_to_prob`, `or_to_prob`) |
| Fitting parametric survival models from data | No | No | Yes, integrates fitted parametric survival models via `flexsurv` (`partsurvfit`, `flexsurvreg_list`) | No | No | Partly, accepts externally fitted survival models (`define_surv_fit`, `load_surv_models`); fitting done upstream |
| Survival extrapolation beyond observed follow-up | No | No | Yes, parametric extrapolation in the partitioned survival and continuous-time engines | No | No | Yes, `define_part_surv` with `join`, `mix`, `apply_hr`, `apply_af` to project beyond the observed curve |
| Fitted regression models as parameters | No | No | Yes, `params_lm`, `params_surv`, `params_mlogit`, `create_params` from fitted models | No | No | Partly, survival fits and `look_up` tables |
| Multi-state transition-intensity estimation | No | No | Yes, `qmatrix` from multi-state (`msm`) fits, transition data tables | No | No | No |
| Bayesian estimation or posterior inputs | Partly, approximate Bayesian computation returns a posterior draw matrix | No | No, frequentist and bootstrap | Yes, consumes MCMC posteriors fitted upstream | No | No |
| Calibration to observed targets | Yes, `abc_calibrate`: approximate Bayesian computation, posterior returned as a draw matrix that flows into `run_psa` | No | No | No | No | Yes, `calibrate_model`, `define_calibration_fn` |

R users fit the survival curves themselves with `flexsurv` (maximum-likelihood
parametric and restricted cubic spline models) or `survHE`, which wraps
`flexsurv` for maximum-likelihood fitting and adds Bayesian fitting through
integrated nested Laplace approximation and Hamiltonian Monte Carlo, digitisation
of published Kaplan-Meier curves, model comparison, and probabilistic sensitivity
analysis of the fitted survival parameters. `hesim` and `heemod` consume those
fits as model inputs. `heormodel` has no equivalent in-package survival-fitting
step, so survival parameters must be estimated elsewhere and brought in as draws.

## Cost-effectiveness analysis

Every package covers the core of cost-effectiveness analysis. The differences
are at the edges: risk aversion, efficiency frontiers, and expected loss.

| Feature | heormodel | dampack | hesim | BCEA | voi | heemod |
|---|---|---|---|---|---|---|
| Incremental cost-effectiveness ratio table | Yes, `icer_table` | Yes, `calculate_icers` and `calculate_icers_psa` | Yes, `icer` and `cea` | Yes, `compute_ICER`, `ce_table` | No | Yes, `summary` of `run_model` |
| Simple and extended dominance | Yes, marked in the ICER table status column | Yes, `calculate_icers` flags dominated and extendedly dominated | Yes | Yes, via the efficiency frontier | No | Yes, in the model summary |
| Net monetary and net health benefit | Yes, `nmb`, `nhb`, `expected_nmb` | Yes, within the PSA summary and metamodels | Yes, within `cea` | Yes, incremental benefit `compute_IB` and `compute_EIB` | No | Yes, net monetary benefit in the summary |
| Cost-effectiveness frontier | Yes, `frontier` | Yes, from `calculate_icers` | Yes | Yes, `ceef.plot` cost-effectiveness efficiency frontier | No | Yes, efficiency frontier in the summary |
| Cost-effectiveness acceptability curve and frontier | Yes, `ceac`, `ceaf` | Yes, `ceac`, `summary.ceac` | Yes, `cea` output, `plot_ceac`, `plot_ceaf` | Yes, `ceac.plot`, `ceaf.plot`, `multi.ce` | No | Partly, via `run_bcea` (BCEA) |
| Cost-effectiveness plane | Yes, `ce_plane` and `plot_ce_plane` | Yes, `plot.psa` and ICER plots | Yes, `plot_ceplane` | Yes, `ceplane.plot`, `contour`, `contour2` | No | Yes, PSA scatter on the plane; contours via `run_bcea` |
| Expected loss curves | Yes, `expected_loss` and `plot_expected_loss` | Yes, `calc_exp_loss` and `plot.exp_loss` | No | Yes, opportunity loss via `compute_ol` | No | Partly, via `run_bcea` |
| Multiple comparators | Yes, any number of interventions | Yes | Yes | Yes, `multi.ce`, `setComparisons` | No | Yes, any number of strategies |
| Risk aversion | No | No | No | Yes, `CEriskav` adds a risk-aversion parameter | No | No |
| Mixed or portfolio strategies | No | No | No | Yes, `mixedAn` for a mix of interventions in the market | No | No |

## Value of information

This is where the R ecosystem is deepest, and where `voi` is the specialist:
`BCEA` calls it for the expected value of partial perfect information, and
`heemod` has no native value-of-information functions at all, exporting instead to
`SAVI` or `BCEA`. `heormodel` implements the three main quantities natively in
Python.

| Feature | heormodel | dampack | hesim | BCEA | voi | heemod |
|---|---|---|---|---|---|---|
| Expected value of perfect information | Yes, `evpi` | Yes, `calc_evpi` | Yes, from `cea` and `plot_evpi` | Yes, `compute_EVI`, `evi.plot` | Yes, `evpi` | Partly, `export_savi` (SAVI) or `run_bcea` (BCEA) |
| Expected value of partial perfect information | Yes, `evppi` (spline and Gaussian process), `evppi_ranking` | Yes, `calc_evppi` (generalized additive model metamodel) | No | Yes, `evppi` (delegates to `voi`) | Yes, `evppi` with many methods (below) | Partly, via `export_savi` or `run_bcea` |
| EVPPI estimation methods | Two, spline and Gaussian process | One, generalized additive model | Not applicable | Inherited from `voi` | Generalized additive model, Gaussian process, multivariate adaptive regression splines (earth), integrated nested Laplace approximation, Bayesian additive regression trees, and single-parameter methods | Inherited from the delegated tool |
| Expected value of sample information | Yes, `evsi_regression`, `evsi_moment_matching`, `evsi_importance_sampling`, `simulate_summaries` | Yes, `calc_evsi` | No | Yes, via `voi` | Yes, `evsi` | Partly, via `export_savi` |
| EVSI estimation methods | Three, nonparametric regression, moment matching, importance sampling | Nonparametric regression | Not applicable | Inherited from `voi` | Nonparametric regression, moment matching, importance sampling | Inherited from the delegated tool |
| Value of information for an estimation problem | No | No | No | No | Yes, `evppivar`, `evsivar` | No |
| Expected net benefit of sampling and population value | No | No | No | Partly, population EVPI in the report | Yes, `enbs`, `enbs_opt`, `pop_voi` | No |
| Information-rank plot | No | No | No | Yes, `info.rank` | No | Partly, via `run_bcea` |

## Reporting, provenance, and documentation

| Feature | heormodel | dampack | hesim | BCEA | voi | heemod |
|---|---|---|---|---|---|---|
| Publication-style plots | Yes, cost-effectiveness plane, acceptability curve and frontier, frontier, tornado, expected loss, with a shared palette | Yes, `ggplot2` plots for each analysis | Yes, `ggplot2` and `autoplot` methods | Yes, base R, `ggplot2`, and interactive `plotly` engines | Yes, `plot.evppi` and tidy output for `ggplot2` | Yes, base R plots for the model, PSA, and DSA |
| Automated report generation | Partly, `capture_run` and run records | No | No | Yes, a report combining the analysis into one document | No | Partly, tabular runs from files (`run_model_tabular`) |
| Provenance and run records | Yes, `capture_run`, `RunRecord`, records draw sources | No | No | Partly, `sim_table` summarizes simulations | No | No |
| Reproducible seeding as a guarantee | Yes, the shared iteration index ties draws to outcomes for value-of-information regression | No | No | No | No | No |
| Tutorials and worked examples | Yes, executable Quarto tutorials with Colab notebooks for every engine and analysis | Yes, six vignettes | Yes, extensive vignettes and articles | Yes, vignettes and a companion book | Yes, vignettes for each method | Yes, extensive vignettes |

## Reading the table

Three patterns stand out.

`heormodel` is the only package that spans the full workflow in one language.
`hesim` builds models but stops at the expected value of perfect information for
value of information and has no calibration or deterministic sensitivity
analysis. `heemod` builds Markov cohort models with strong survival support and
calibration, but covers only the cohort engine and has no native
value-of-information functions, exporting to `SAVI` or `BCEA` instead. `dampack`,
`BCEA`, and `voi` analyze outputs but build no models. A Python user assembling
the R equivalent of `heormodel` would combine a modeling package, a survival
package, a cost-effectiveness package, and `voi`, then move data between them.

`heormodel` carries engines the R packages do not: discrete-event simulation and
compartmental ordinary-differential-equation transmission models sit outside all
five comparators. The stochastic compartmental engine on the roadmap would widen
that gap.

The R ecosystem still leads in three places, each a candidate roadmap item.
First, survival analysis and parameter estimation: `hesim` and `heemod` consume
fitted survival, multi-state, and regression models, and `flexsurv` and `survHE`
fit them, including extrapolation beyond the observed follow-up, all of which
`heormodel` leaves to upstream tools. Second, `voi` offers more estimation
methods for the expected value of partial perfect information and adds the
expected net benefit of sampling and population value of information. Third,
`BCEA` adds risk aversion, mixed-strategy analysis, and an automated report.
`hesim` also runs individual continuous-time simulation at a scale its C++
backend is built for. These are where the need is demonstrated.
