# Changelog

All notable changes to `heval` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/).

Each entry links to the pull request that introduced it. Add a line under
`[Unreleased]` in the same PR that makes the change; see
[RELEASING.md](RELEASING.md) for how `[Unreleased]` turns into a release.

## [Unreleased]

## [0.3.0] - 2026-07-04

### Added

- Microsimulation engines: `DiscreteTimeMicrosimEngine` advances an
  individual-level population on a cycle grid with history-dependent
  transitions and heterogeneity, and `ContinuousTimeMicrosimEngine` races
  competing time-to-event samplers between events. Both emit the standard
  `Outcomes` schema, seed each iteration from a `SeedManager` so results do not
  depend on `n_jobs`, and use common random numbers across strategies by
  default.
- `heval.models._accrual`: shared cost and utility accrual, discounting, and
  aggregation to `Outcomes`, used by both engines and reserved for the
  discrete-event engine.
- `SeedManager.child_sequence` returns a per-key seed sequence, so
  iteration-indexed streams stay identical however a run is chunked across
  workers.
- Microsimulation example (`examples/microsim.py`) and website tutorial,
  validated against the closed-form cohort solution it mirrors.

## [0.2.0] - 2026-07-04

### Added

- Documentation website built with Quarto and quartodoc, published to GitHub
  Pages from CI on every merge to `main`: executed tutorials, concept pages, a
  generated API reference, the roadmap, and this changelog
  ([#3](https://github.com/pedroliman/heval/pull/3)).
- `heval.params.mix_draws` combines draw matrices from different sources
  (a calibrated posterior and literature draws) into one PSA matrix,
  resampling whole rows so joint correlation survives and sources stay
  independent.
- `capture_run` records a `draw_sources` map, so the run report shows where
  each parameter's draws came from.
- Calibration workflow example (`examples/calibration_workflow.py`) and
  website tutorial: calibrate a natural-history model's rates, mix them with
  literature parameters, and run CEA and VoI on the result.

### Changed

- Docstring cross-references use plain backticks instead of Sphinx roles, so
  they render cleanly on the website and in `help()`
  ([#3](https://github.com/pedroliman/heval/pull/3)).
- Distribution spec strings format floats at 6 significant digits, so
  provenance records and run reports stay readable
  ([#3](https://github.com/pedroliman/heval/pull/3)).
- `RunRecord.model_card()` renamed to `to_markdown()`, titled "Run report":
  "model card" is ML documentation jargon, not a HEOR term
  ([#6](https://github.com/pedroliman/heval/pull/6)).

## [0.1.0] - 2026-07-04

Initial release: parameter sampling (`heval.params`), the `Outcomes` schema
and `ModelEngine` protocol (`heval.models`), the PSA run loop and
bring-your-own-outputs ingestion (`heval.run`), cost-effectiveness analysis
(`heval.cea`), value-of-information analysis (`heval.voi`), optional ABC
calibration (`heval.calibrate`), and reporting plots (`heval.report`).

[Unreleased]: https://github.com/pedroliman/heval/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/pedroliman/heval/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pedroliman/heval/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pedroliman/heval/releases/tag/v0.1.0
