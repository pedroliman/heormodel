# 13. Continuous-time Sick-Sicker replication (DES tutorial)

Reproduce the discrete-event simulation (DES) tutorial of Lopez-Mendez, Goldhaber-Fiebert, and Alarid-Escudero, "A Tutorial on Discrete Event Simulation Models Using a Cost-Effectiveness Analysis Example in R," Medical Decision Making 2026;46(5):533-548 (PMID 42087677). The article simulates the Sick-Sicker model in continuous time from age 25 to 100 with age-dependent background mortality, a Weibull state-residence-time hazard for progression, recurrence, transition rewards, and four strategies, then computes a cost-effectiveness analysis, epidemiological outcomes, and a probabilistic analysis with acceptability curves, expected loss curves, and the expected value of perfect information. It ships as `examples/mdm_des.py` and a website replication tutorial, the fourth entry in the replication gallery.

This item assumes items 3 through 5 (the engines) and extends the framework in four places. Each extension is additive and plugs into the existing contract: draws in, `Outcomes` out.

## Why the continuous-time microsimulation clock, not `DESModel`

The article's simulation kernel samples a latent arrival time for every permitted transition out of the current state, takes the earliest, moves the individual, and repeats. That is exactly what `MicrosimModel(clock="continuous")` does: `hazards` returns sampled times to each competing destination, the engine takes the earliest and accrues continuously between events. The replication therefore uses the continuous microsimulation clock, which is vectorized across individuals and makes the probabilistic analysis tractable. `DESModel` stays the engine for resource-constrained processes (queues, capacity), which the article treats as an extension beyond its base example; the tutorial page states this framing so readers looking for "DES" land on the right engine.

## Extension 1: `LifeTable` (new `heormodel.models.lifetable`)

Age-dependent background mortality needs a sampler for "years until death given the current age," under a hazard ratio for the disease states. No engine support exists; examples so far hard-code a constant rate or embed an age-indexed array in a cohort model.

`LifeTable(ages, rates)` holds piecewise-constant annual mortality rates by age band, the form life tables take, with the last band extended indefinitely so death is certain. `sample_time_to_death(rng, age, hazard_ratio=1.0)` inverts the cumulative hazard exactly: solve `hr * (H(age + t) - H(age)) = e` with `e` standard exponential. Each draw is conditional on survival to the current age, so a `hazards` function can redraw the death time at every state entry, which is how proportional excess mortality (hazard ratios 3 and 10 here) composes with age dependence. `life_expectancy(age, hazard_ratio=1.0)` integrates the survival function band by band and is the analytic mean of the sampler, used in tests. The article samples discretized annual death probabilities with a uniform within-year correction; exact inversion is simpler and statistically exact for piecewise-constant rates, and the tutorial notes this divergence.

## Extension 2: one-time transition payoffs on the continuous clock

The model pays $1,000 on the Healthy-to-Sick transition, $2,000 on death, and a one-time utility decrement of 0.01 at disease onset. The continuous clock accrues flows only, so it cannot express these. `MicrosimModel` gains an optional `transition_payoffs` function, `fn(params, state_from, state_to, attrs) -> (cost, effect)`, applied when an event fires and discounted at the event time. It is continuous-clock only; supplying it with the discrete clock raises, since the discrete engine's per-cycle `payoffs` sees the state sequence and can already price transitions if extended later. Validation: for a one-time cost `ic` paid at an exponential death time with rate `r` and discount rate `d`, the expected discounted cost is `ic * r / (r + d)` over an unbounded horizon, a closed form a test asserts.

## Extension 3: an event history and epidemiological outcomes (new `heormodel.epi`)

The article's postprocessing module B computes survival probabilities, disease prevalence, and dwell-time distributions from the event history. `Outcomes` averages individuals away, so the engine needs a side channel. `MicrosimModel.evaluate` accepts `trace="events"` and returns a long event log with columns `strategy`, `iteration`, `individual`, `time`, `from_state`, `to_state`, one row per transition, on either clock (the discrete clock logs state changes on the cycle grid).

`heormodel.epi` turns the log into epidemiological outcomes:

- `state_occupancy(events, states=..., initial_state=..., n_individuals=..., times=...)` returns the proportion of individuals in each state at each requested time, indexed by `(strategy, iteration, time)` with one column per state. Individuals appear in the log only when they move, so the initial state and the population size are explicit arguments.
- `survival(occupancy, dead_state=...)` is the proportion not yet dead.
- `prevalence(occupancy, states=..., dead_state=...)` is the proportion in the given disease states among those alive, the epidemiological definition.

Dwell times need no new API; they are differences of consecutive event times within an individual, computed in the tutorial with a groupby.

## Extension 4: expected loss curves in `heormodel.cea`

The article's figure 4C plots expected loss curves, the expected foregone net monetary benefit of choosing each strategy at each willingness-to-pay threshold; the strategy with the lowest expected loss is the frontier choice, and its expected loss equals the expected value of perfect information (EVPI). `expected_loss(outcomes, wtp, effect=None)` returns a DataFrame indexed by the threshold grid with one column per strategy, computed per iteration as the gap to that iteration's best net monetary benefit. A test asserts the identity `expected_loss(...).min(axis=1) == evpi(...)` on the same outcomes. `heormodel.report` gains `plot_expected_loss` beside `plot_ceac`.

## The replication itself

`examples/mdm_des.py` encodes the article's table 1 exactly: onset rate 0.15, recovery rate 0.5, Weibull progression with proportional-hazards scale 0.08 and shape 1.10 on time since entering Sick, US 2015 all-cause mortality rates (the same array `examples/mdm_cohort_timedep.py` embeds) with hazard ratios 3 and 10, strategy A improving the Sick utility 0.75 to 0.95 at $12,000 per year, strategy B scaling the progression hazard by 0.6 at $13,000 per year, AB combining both, treatment costs paid in both disease states, transition rewards as above, and 3% continuous discounting anchored at age 25. The Weibull draw needs no truncation because the engine redraws competing times at every state entry, when time in state is zero; strategy B multiplies the proportional-hazards scale, converted to the sampling parameterization by `scale ** (-1 / shape)`.

The probabilistic analysis uses the article's distributions: gamma for rates and costs, beta for utilities and the onset decrement, lognormal for the three hazard ratios, and the Weibull scale and shape as two lognormals with Spearman correlation 0.5 through the existing Gaussian-copula sampler, which is the article's bivariate lognormal up to the rank-to-linear correlation conversion. The article draws 1,000 parameter sets by Latin hypercube sampling; the replication uses the existing correlated Monte Carlo sampler at the same size and notes the divergence rather than adding a Latin hypercube option for one tutorial.

The article reports no base-case ICER table; its results are figures. Validation is therefore structural plus graphical:

- Cross-validation against a closed form: an all-exponential variant (Weibull shape 1, constant mortality) is a continuous-time Markov chain whose expected discounted costs and QALYs, including transition rewards, solve `(d I - Q) v = r` for generator `Q`. A test asserts the continuous-clock engine matches this within Monte Carlo error, exercising every new engine path.
- `LifeTable` sampled means match `life_expectancy` analytically (extension 1 tests).
- The tutorial reproduces the article's figure 3 (survival and prevalence by strategy, with SoC and A identical and B and AB identical) and figure 4 (acceptability curves and frontier, expected loss curves, EVPI over a $0 to $200,000 grid at $1,000 steps), and states how the replication's numbers sit relative to the published figures.

## Deliverables

- `heormodel.models.LifeTable`, `transition_payoffs` and `trace="events"` on `MicrosimModel`, `heormodel.epi`, `heormodel.cea.expected_loss`, `heormodel.report.plot_expected_loss`, each with docstring examples and tests.
- `examples/mdm_des.py`: base case at 100,000 individuals, epidemiological outcomes, probabilistic analysis at 1,000 draws, plots, and a run report.
- A website replication tutorial walking through the example at reduced sizes, a replication gallery row, sidebar and API reference entries, and a changelog entry.

## Acceptance

- The continuous-time Markov chain cross-validation, the transition-payoff closed form, the life-table analytics, and the expected-loss identity all pass as tests at fixed seeds.
- The example runs under `uv run python examples/mdm_des.py`; the rendered tutorial executes; prose matches the printed outputs and plots.
- Survival and prevalence curves show SoC identical to A and B identical to AB, matching the article's figure 3; the acceptability frontier switches SoC to B to AB as the threshold rises, matching figure 4.
