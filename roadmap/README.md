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

## Prioritized next steps

| # | Item | Design note |
|---|------|-------------|
| 3 | Microsimulation engine (discrete-time, then continuous-time) | [03-microsim-engine.md](03-microsim-engine.md) |
| 4 | DES engine wrapping SimPy, coherent with the microsim architecture | [04-des-engine.md](04-des-engine.md) |

Each new feature ships with a website tutorial, as items 1 and 2 did.

## Backlog

- Markov cohort engine (`models/markov.py` stub): vectorized transition-matrix sweeps across iterations, per-state payoffs, half-cycle correction, discounting. Sequenced after the microsim engine so it can reuse the discounting utilities built there.
- Remaining EVSI estimators (`voi/evsi.py` stubs): moment matching and importance sampling, sharing `simulate_summaries` and the metamodel module.
- Run-loop caching (`heval.run`): cache `Outcomes` keyed on the draws and the model identity, so re-running a notebook does not re-simulate.
- Richer convergence diagnostics (`heval.run.diagnostics`): stability of ICERs, CEAC curves, and EVPI across bootstrap resamples; standard errors for VoI estimates.
- Analyses over multiple effect columns: the schema already carries extra effect columns; add convenience sweeps in `cea` and `voi`.
- Correlation ergonomics: accept a full labelled Spearman matrix (for example, estimated from a calibrated posterior) with validation and nearest-PSD repair reported to the user.
- CI and packaging: GitHub Actions running pytest, doctests, ruff, and mypy on Python 3.11 and 3.12; publish to PyPI once the engine phases stabilize the API.
