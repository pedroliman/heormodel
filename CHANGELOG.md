# Changelog

All notable changes to `heval` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/).

Each entry links to the pull request that introduced it. Add a line under
`[Unreleased]` in the same PR that makes the change; see
[RELEASING.md](RELEASING.md) for how `[Unreleased]` turns into a release.

## [Unreleased]

## [0.6.0] - 2026-07-05

### Changed

- Resonant engine names and clearer parameters (breaking, no aliases). The
  engines are now `MarkovModel` (was `MarkovCohortEngine`), `MicrosimModel`
  (folds `DiscreteTimeMicrosimEngine` and `ContinuousTimeMicrosimEngine` into
  one class with a `clock` argument, `"discrete"` by default, `"continuous"`
  for the competing-hazards path), and `DESModel` (was `DESEngine`). The Markov
  structure callback is `model_fn` in place of `build`. Every engine takes one
  `discount_rate` (annual, default `0.03`, applied to costs and effects) in
  place of `discount_cost` and `discount_effect`; `cycle_length` scales the
  annual clock. The pre-0.6 names are removed outright rather than deprecated
  ([#11](https://github.com/pedroliman/heval/pull/11)).

## [0.5.0] - 2026-07-05

### Added

- Cohort state-transition engine: `MarkovCohortEngine` sweeps a cohort trace
  across PSA iterations and emits `Outcomes`. Transitions may be one matrix or a
  per-cycle array (age-varying rates); rewards accrue per state and, optionally,
  per transition (a one-time cost of dying or disutility of onset). Supports
  Simpson's 1/3, half-cycle, or no within-cycle correction, reusing
  `heval.models._accrual` for discounting. `CohortSpec` carries one strategy's
  matrices, and `gen_wcc` builds the correction weights
  ([#9](https://github.com/pedroliman/heval/pull/9)).
- `duration_groups` on `DiscreteTimeMicrosimEngine`: a per-individual counter of
  consecutive cycles spent in a set of states, so a sojourn that progresses
  (Sick to Sicker) keeps counting where `time_in_state` would reset
  ([#9](https://github.com/pedroliman/heval/pull/9)).
- Three replications of published Sick-Sicker cost-effectiveness tutorials, each
  matching the source's deterministic results, with runnable scripts and website
  tutorials: cohort state-transition (`examples/mdm_cohort.py`), time-dependent
  cohort with age-varying mortality (`examples/mdm_cohort_timedep.py`), and
  microsimulation (`examples/mdm_microsim.py`). A replication gallery page
  collects them with citations ([#9](https://github.com/pedroliman/heval/pull/9)).

## [0.4.0] - 2026-07-04

### Added

- Discrete-event simulation engine: `DESEngine` wraps SimPy. The environment,
  process functions, and resources stay the user's own code; the engine adds a
  per-entity toolkit for discounted cost and utility accrual, per-iteration
  seeding from a `SeedManager` so results do not depend on `n_jobs`, and an
  optional event log. It reuses `heval.models._accrual` and uses common random
  numbers across strategies by default, staying coherent with the
  microsimulation engines ([#8](https://github.com/pedroliman/heval/pull/8)).
- `heval.models.queue_waits`: derive per-request waiting times from a `DESEngine`
  trace, so queueing reports come from the event log rather than engine
  internals ([#8](https://github.com/pedroliman/heval/pull/8)).
- `simpy` as an optional dependency behind the `des` extra
  (`uv pip install 'heval[des]'`) ([#8](https://github.com/pedroliman/heval/pull/8)).
- Discrete-event example (`examples/des.py`) and website tutorial, validated
  against an M/M/1 queue and the exponential cohort solution
  ([#8](https://github.com/pedroliman/heval/pull/8)).

## [0.3.0] - 2026-07-04

### Added

- Microsimulation engines: `DiscreteTimeMicrosimEngine` advances an
  individual-level population on a cycle grid with history-dependent
  transitions and heterogeneity, and `ContinuousTimeMicrosimEngine` races
  competing time-to-event samplers between events. Both emit the standard
  `Outcomes` schema, seed each iteration from a `SeedManager` so results do not
  depend on `n_jobs`, and use common random numbers across strategies by
  default ([#7](https://github.com/pedroliman/heval/pull/7)).
- `heval.models._accrual`: shared cost and utility accrual, discounting, and
  aggregation to `Outcomes`, used by both engines and reserved for the
  discrete-event engine ([#7](https://github.com/pedroliman/heval/pull/7)).
- `SeedManager.child_sequence` returns a per-key seed sequence, so
  iteration-indexed streams stay identical however a run is chunked across
  workers ([#7](https://github.com/pedroliman/heval/pull/7)).
- Microsimulation example (`examples/microsim.py`) and website tutorial,
  validated against the closed-form cohort solution it mirrors
  ([#7](https://github.com/pedroliman/heval/pull/7)).

## [0.2.0] - 2026-07-04

### Added

- Documentation website built with Quarto and quartodoc, published to GitHub
  Pages from CI on every merge to `main`: executed tutorials, concept pages, a
  generated API reference, the roadmap, and this changelog
  ([#3](https://github.com/pedroliman/heval/pull/3)).
- `heval.params.mix_draws` combines draw matrices from different sources
  (a calibrated posterior and literature draws) into one PSA matrix,
  resampling whole rows so joint correlation survives and sources stay
  independent ([#5](https://github.com/pedroliman/heval/pull/5)).
- `capture_run` records a `draw_sources` map, so the run report shows where
  each parameter's draws came from
  ([#5](https://github.com/pedroliman/heval/pull/5)).
- Calibration workflow example (`examples/calibration_workflow.py`) and
  website tutorial: calibrate a natural-history model's rates, mix them with
  literature parameters, and run CEA and VoI on the result
  ([#5](https://github.com/pedroliman/heval/pull/5)).

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

[Unreleased]: https://github.com/pedroliman/heval/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/pedroliman/heval/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/pedroliman/heval/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/pedroliman/heval/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pedroliman/heval/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pedroliman/heval/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pedroliman/heval/releases/tag/v0.1.0
