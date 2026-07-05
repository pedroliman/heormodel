# heval roadmap

What is left to implement, in priority order. Each item has a design note in this folder. The notes are specific enough to start from; the final API is settled in the implementing PR.

One rule governs everything below: new features plug into the existing contract. Parameter draw matrices carry an `iteration` index, engines emit the `Outcomes` schema, and the analysis layer consumes only that schema.

All prose follows [guidance/writing_style.md](../guidance/writing_style.md).

## Phase 1 status (done)

- `params`: distribution specs, mean/SE constructors, correlated sampling
- `models`: `Outcomes` schema and `ModelEngine` protocol (engines stubbed)
- `run`: `SeedManager`, `run_psa`, bring-your-own-outputs, running means
- `cea`: ICERs, dominance, extended dominance, frontier, NMB/NHB, CEAC/CEAF
- `voi`: EVPI, EVPPI (spline/GP), EVSI (nonparametric regression)
- `calibrate`: ABC-SMC via pyabc, posterior returned as a draw matrix
- `report`: CE plane, CEAC/CEAF, frontier, tornado, provenance, run report

## Shipped features

Design notes for completed items move to [`done/`](done/). Items 1 through 6 are shipped.

- Item 1, quartodoc documentation website ([done/01-quartodoc-site.md](done/01-quartodoc-site.md)): the site publishes to GitHub Pages with API reference, tutorials, and concept pages.
- Item 2, full calibration workflow ([done/02-calibration-workflow.md](done/02-calibration-workflow.md)): `heval.params.mix_draws` combines calibrated and literature draw matrices; `capture_run` records `draw_sources`; `examples/calibration_workflow.py` and the calibration workflow tutorial run it end to end.
- Item 3, microsimulation engine ([done/03-microsim-engine.md](done/03-microsim-engine.md)): `MicrosimModel` simulates an individual-level population per iteration and emits `Outcomes`, on a discrete cycle grid (`clock="discrete"`) or in continuous time (`clock="continuous"`). It shares the `heval.models._accrual` layer, seeds each iteration from a `SeedManager` so results do not depend on `n_jobs`, and uses common random numbers across strategies by default. `examples/microsim.py` and the microsimulation tutorial run it end to end.
- Item 4, discrete-event simulation engine ([done/04-des-engine.md](done/04-des-engine.md)): `DESModel` wraps SimPy. The environment, processes, and resources stay the user's code; the engine adds a per-entity toolkit for discounted cost and utility accrual (reusing `heval.models._accrual`), per-iteration seeding, and an event log that `queue_waits` turns into queueing reports. `examples/des.py` and the discrete-event tutorial run it end to end, validated against an M/M/1 queue and the exponential cohort solution the continuous-time microsim also matches.
- Item 5, Markov cohort engine ([done/05-markov-cohort-engine.md](done/05-markov-cohort-engine.md)): `MarkovModel` sweeps a cohort trace across iterations with constant or per-cycle (age-varying) transition arrays, per-state and per-transition rewards, discounting, and Simpson's 1/3, half-cycle, or no within-cycle correction, reusing `heval.models._accrual`. `MicrosimModel` gained `duration_groups` for time spent in a set of states. Three replications of published Sick-Sicker cost-effectiveness tutorials (cohort, time-dependent, and microsimulation) match their deterministic results and ship as `examples/mdm_*.py` with website tutorials and a replication gallery.
- Item 6, resonant engine names and clearer parameters ([done/06-resonant-api.md](done/06-resonant-api.md)): the engines are `MarkovModel`, `MicrosimModel` (one class, `clock="discrete"` or `"continuous"`), and `DESModel`; the Markov structure callback is `model_fn`; and every engine takes one `discount_rate` (annual, default `0.03`) in place of `discount_cost` and `discount_effect`. The pre-0.6 names were removed outright, without deprecation aliases.

## Prioritized next steps

The engine phases are complete for cohort state-transition, microsimulation, and discrete-event simulation, and the public API now reads in model-type names. The next items broaden how parameters enter the run loop, add deterministic sensitivity analysis, and fill gaps in the tutorials. Each ships with a website tutorial where it adds public API, as items 1 and 2 did.

- Item 7, parameter inputs from data ([07-parameter-inputs.md](07-parameter-inputs.md)): `single_draw` for a base-case run, `read_draws` for a draw matrix from a CSV or DataFrame, and `resample_posterior` for a weighted posterior resampled with replacement.
- Item 8, deterministic sensitivity analysis ([08-deterministic-sensitivity.md](08-deterministic-sensitivity.md)): a `heval.dsa` module with one-way, one-at-a-time, and full-factorial grid designs that run through the existing loop and feed tornado and heatmap reports.
- Item 9, parallel runs by default with a time-remaining display ([09-parallel-and-progress.md](09-parallel-and-progress.md)): `run_psa` defaults to all cores (`sequential=True` opts out) and reports completed experiments with an estimate of time remaining from finished work.
- Item 10, Markov vs microsimulation models ([10-markov-vs-microsim-tutorial.md](10-markov-vs-microsim-tutorial.md)): a cross-validation tutorial showing the two engines converge under shared assumptions and diverge under risk heterogeneity, emphasizing what a microsimulation represents that a cohort model averages away.
- Item 11, documentation narrative order ([11-docs-narrative-order.md](11-docs-narrative-order.md)): reorder the tutorials so a reader sees bring-your-own-outputs, then a Markov cohort model, then the microsimulation.
- Item 12, value-of-information tutorial ([12-voi-tutorial.md](12-voi-tutorial.md)): an EVPI, EVPPI, and EVSI walkthrough reproducing a published VoI analysis and checking the numbers against it.

## Backlog

Unscheduled work, no design note yet.

- Remaining EVSI estimators (`voi/evsi.py` stubs): moment matching and importance sampling, sharing `simulate_summaries` and the metamodel module.
- Run-loop caching (`heval.run`): cache `Outcomes` keyed on the draws and the model identity, so re-running a notebook does not re-simulate.
- Richer convergence diagnostics (`heval.run.diagnostics`): stability of ICERs, CEAC curves, and EVPI across bootstrap resamples; standard errors for VoI estimates.
- Analyses over multiple effect columns: the schema already carries extra effect columns; add convenience sweeps in `cea` and `voi`.
- Correlation ergonomics: accept a full labelled Spearman matrix (for example, estimated from a calibrated posterior) with validation and nearest-PSD repair reported to the user.
- CI and packaging: GitHub Actions running pytest, doctests, ruff, and mypy on Python 3.11 and 3.12; publish to PyPI once the engine phases stabilize the API.
