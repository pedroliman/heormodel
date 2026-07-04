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

## Shipped since phase 1

- Item 1, quartodoc documentation website ([01-quartodoc-site.md](01-quartodoc-site.md)): the site publishes to GitHub Pages with API reference, tutorials, and concept pages.
- Item 2, full calibration workflow ([02-calibration-workflow.md](02-calibration-workflow.md)): `heval.params.mix_draws` combines calibrated and literature draw matrices; `capture_run` records `draw_sources`; `examples/calibration_workflow.py` and the calibration workflow tutorial run it end to end.
- Item 3, microsimulation engine ([03-microsim-engine.md](03-microsim-engine.md)): `DiscreteTimeMicrosimEngine` and `ContinuousTimeMicrosimEngine` simulate an individual-level population per iteration and emit `Outcomes`. They share the `heval.models._accrual` layer, seed each iteration from a `SeedManager` so results do not depend on `n_jobs`, and use common random numbers across strategies by default. `examples/microsim.py` and the microsimulation tutorial run it end to end.
- Item 4, discrete-event simulation engine ([04-des-engine.md](04-des-engine.md)): `DESEngine` wraps SimPy. The environment, processes, and resources stay the user's code; the engine adds a per-entity toolkit for discounted cost and utility accrual (reusing `heval.models._accrual`), per-iteration seeding, and an event log that `queue_waits` turns into queueing reports. `examples/des.py` and the discrete-event tutorial run it end to end, validated against an M/M/1 queue and the exponential cohort solution the continuous-time microsim also matches.

## Prioritized next steps

The engine phases are complete for microsimulation and discrete-event simulation. The next priorities are in the backlog below. The Markov cohort engine is sequenced first, now that the accrual and discounting utilities it reuses exist. Each new feature ships with a website tutorial, as items 1 and 2 did.

## Backlog

- Markov cohort engine (`models/markov.py` stub): vectorized transition-matrix sweeps across iterations, per-state payoffs, half-cycle correction, discounting. Reuses the `heval.models._accrual` discounting utilities built for the microsimulation and discrete-event engines.
- Remaining EVSI estimators (`voi/evsi.py` stubs): moment matching and importance sampling, sharing `simulate_summaries` and the metamodel module.
- Run-loop caching (`heval.run`): cache `Outcomes` keyed on the draws and the model identity, so re-running a notebook does not re-simulate.
- Richer convergence diagnostics (`heval.run.diagnostics`): stability of ICERs, CEAC curves, and EVPI across bootstrap resamples; standard errors for VoI estimates.
- Analyses over multiple effect columns: the schema already carries extra effect columns; add convenience sweeps in `cea` and `voi`.
- Correlation ergonomics: accept a full labelled Spearman matrix (for example, estimated from a calibrated posterior) with validation and nearest-PSD repair reported to the user.
- CI and packaging: GitHub Actions running pytest, doctests, ruff, and mypy on Python 3.11 and 3.12; publish to PyPI once the engine phases stabilize the API.
