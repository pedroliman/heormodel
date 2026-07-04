# heval roadmap

What is left to implement, in priority order. Each item has a design note
in this folder; the notes are binding enough to start work from, but the
final API is settled in the PR that implements it.

The one rule that governs everything below: **new features plug into the
existing contract** — parameter draw matrices carry an `iteration` index,
engines emit the `Outcomes` schema, and the analysis layer consumes only
that schema. Nothing in this roadmap changes the contract; everything
targets it.

## Status of phase 1 (done)

- `params`: distribution specs, mean/SE constructors, correlated sampling ✅
- `models`: `Outcomes` schema + `ModelEngine` protocol (engines stubbed) ✅
- `run`: `SeedManager`, `run_psa`, bring-your-own-outputs, running means ✅
- `cea`: ICERs, dominance/extended dominance, frontier, NMB/NHB, CEAC/CEAF ✅
- `voi`: EVPI, EVPPI (spline/GP), EVSI (nonparametric regression) ✅
- `calibrate`: ABC-SMC via pyabc, posterior as draw matrix ✅
- `report`: CE plane, CEAC/CEAF, frontier, tornado, provenance/model card ✅

## Prioritized next steps

| # | Item | Design note | Depends on |
|---|------|-------------|------------|
| 1 | **Full calibration workflow example** — calibrate parameters to targets, mix calibrated draws with literature-derived draws, run PSA, CEA + VoI | [01-calibration-workflow.md](01-calibration-workflow.md) | small `params` addition (`mix_draws`) |
| 2 | **Microsimulation engine** (discrete-time first, continuous-time second) | [02-microsim-engine.md](02-microsim-engine.md) | — |
| 3 | **DES engine wrapping SimPy**, coherent with the microsim architecture | [03-des-engine.md](03-des-engine.md) | shared accrual layer from #2 |
| 4 | **quartodoc documentation website** | [04-quartodoc-site.md](04-quartodoc-site.md) | — (can proceed in parallel) |

## Backlog (after the priorities above)

- **Markov cohort engine** (`models/markov.py` stub): vectorised
  transition-matrix sweeps across PSA iterations, per-state payoffs,
  half-cycle correction, discounting. Deliberately sequenced after the
  microsim engine because the individual-level accrual layer (design note
  02) does not depend on it, while the cohort engine can reuse the
  discounting utilities built there.
- **Remaining EVSI estimators** (`voi/evsi.py` stubs): moment matching and
  importance sampling, sharing `simulate_summaries` and the metamodel
  module with the implemented regression method.
- **Run-loop caching** (`heval.run`): content-addressed caching of
  `Outcomes` keyed on (draws hash, model identity/version), so re-running
  an analysis notebook does not re-simulate.
- **Richer convergence diagnostics** (`heval.run.diagnostics`): stability
  of ICERs, CEAC curves, and EVPI across bootstrap resamples of the PSA;
  standard errors for VoI estimates.
- **Multiple simultaneous effect columns in analyses**: the schema already
  carries extra effect columns; `cea`/`voi` currently analyse one at a
  time. Add convenience sweeps (e.g. CEA by effect column).
- **Correlation ergonomics**: accept a full labelled Spearman matrix
  estimated from data (e.g. from a calibrated posterior) when building a
  `ParameterSet`, with validation and nearest-PSD repair reported to the
  user.
- **Packaging/CI**: GitHub Actions running `uv run pytest`,
  `--doctest-modules`, `ruff`, `mypy` on 3.11/3.12; publish to PyPI once
  the engine phases stabilise the API.
