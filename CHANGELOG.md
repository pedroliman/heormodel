# Changelog

All notable changes to `heormodel` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/).

Each entry links to the pull request that introduced it. Add a line under
`[Unreleased]` in the same PR that makes the change; see
[RELEASING.md](RELEASING.md) for how `[Unreleased]` turns into a release.

## [Unreleased]

### Added

- An "Open in Colab" badge on every tutorial with executable code, backed by a
  runnable notebook under `docs/_notebooks/` that `docs/build_colab_notebooks.py`
  generates from the tutorial. Each notebook opens with a setup cell that installs
  `heormodel` from the repository; the discrete-event simulation replication
  clones the repository so the model code and mortality table it reads are present
  ([#45](https://github.com/pedroliman/heormodel/issues/45)).
- `MarkovModel.trace(params, intervention)` returns the cohort occupancy trace
  (a `cycle` column and one occupancy column per state), the public parallel to
  `ODEModel.trajectory`, for inspecting and validating a cohort model
  ([#43](https://github.com/pedroliman/heormodel/issues/43)).

## [0.7.1] - 2026-07-11

### Added

- `.zenodo.json` and `CITATION.cff` so every GitHub release archives to
  Zenodo with a DOI once the one-time Zenodo integration is enabled; see
  [RELEASING.md](RELEASING.md#zenodo-archive) ([#39](https://github.com/pedroliman/heormodel/issues/39)).

## [0.7.0] - 2026-07-11

### Changed

- Renamed the package's arm vocabulary to PICOTS terms (Population,
  Intervention, Comparator, Outcomes, Timing, Setting): `Strategy` is now
  `Intervention` everywhere (the `interventions=` constructor argument, the
  `intervention` outcome index level, `Outcomes.interventions`, and every
  model-function callback's second argument), and `Strategy.overrides` is now
  `Intervention.decision_levers`. `Intervention` gained an `is_comparator`
  flag: at most one intervention in a sequence may set it, and
  `heormodel.models._interventions.comparator_of` reads it back. Every
  engine resolves the flag at construction and carries the comparator name on
  the `Outcomes` it returns (`Outcomes.comparator`); `heormodel.cea.ce_plane`
  and `heormodel.report.plot_ce_plane` fall back to it, then to the first
  intervention, when their own `comparator` argument is omitted. No
  deprecated aliases: every call site must switch to the new names.

### Added

- Ordinary differential equation engine ([#33](https://github.com/pedroliman/heormodel/issues/33)):
  `heormodel.models.ODEModel` integrates a user-supplied system of ordinary
  differential equations over population compartments and returns the standard
  `Outcomes` structure, for transmission models where a force of infection
  couples the compartments non-linearly. `ODESpec` carries one intervention's
  right-hand side, initial compartment sizes, per-compartment cost and effect
  rates, and an optional flow-event channel for one-time costs attached to a
  rate of movement (a dose administered, an infection treated). The engine
  augments the system with two accumulator equations and integrates the
  discounted reward flows with `scipy.integrate.solve_ivp`, reusing
  `heormodel.models._accrual` for discounting; it is deterministic, so it
  satisfies `ModelEngine` with no random streams. `ODEModel.trajectory` returns
  compartment occupancy over time for inspection and plotting.
  `examples/seir_vaccination.py` and a website tutorial run a
  susceptible-exposed-infectious-recovered vaccination cost-effectiveness
  analysis end to end: the program nearly eliminates the modelled epidemic and
  buys quality-adjusted life-years at an incremental cost-effectiveness ratio
  around 3,200, and the expected value of perfect information at a 50,000
  threshold is driven almost entirely by the basic reproduction number. A
  stochastic compartmental counterpart is planned in
  [`devdocs/roadmap/16-stochastic-compartmental-engine.md`](https://github.com/pedroliman/heormodel/blob/main/devdocs/roadmap/16-stochastic-compartmental-engine.md).
- Continuous-time Sick-Sicker replication (roadmap item 13):
  `examples/mdm_des/` (a folder of building blocks composed by `run.py`) and a
  website tutorial reproduce the published figures of the discrete-event
  simulation tutorial of Lopez-Mendez, Goldhaber-Fiebert, and Alarid-Escudero
  (Medical Decision Making 2026;46(5):533-548) on the continuous clock of
  `MicrosimModel`, with the base case, epidemiological outcomes, and a 1,000-set
  probabilistic analysis with acceptability, expected loss, and EVPI curves. It
  matches the companion code on the two points that move the numbers: each
  one-time transition amount (onset cost, onset disutility, cost of dying)
  accrues over the sojourn that ends in it (reconstructed from the event
  history), and six parameters the companion draws but never reads are held at
  their base case. Three framework additions carry it:
  `heormodel.models.LifeTable` samples time to death from piecewise-constant
  age-specific mortality rates by exact cumulative-hazard inversion under a
  hazard ratio; `MicrosimModel` returns the event history via
  `evaluate(trace="events")` on either clock, which `heormodel.models.state_occupancy`
  turns into state occupancy over time; and `heormodel.cea.expected_loss` with
  `heormodel.report.plot_expected_loss` adds expected loss curves whose minimum
  equals EVPI. A test cross-validates the replication's all-exponential variant,
  including the sojourn-accrued transition amounts, against the continuous-time
  Markov chain closed form ([#27](https://github.com/pedroliman/heormodel/pull/27)).

- `MicrosimModel` callbacks renamed for clarity: `hazards` is now `event_times`,
  `transition` is now `transition_probabilities`, and `payoffs` is now
  `state_costs_and_utilities`, each named for what the caller supplies
  ([#27](https://github.com/pedroliman/heormodel/pull/27)).

- Value-of-information tutorial: `examples/voi_tutorial.py` and a website tutorial
  run EVPI, EVPPI, and EVSI end to end on the Gaussian linear decision model that
  anchors the regression VoI literature (Strong, Oakley & Brennan, 2014; Strong,
  Oakley, Brennan & Breeze, 2015), framed as a two-intervention cost-effectiveness
  decision at 30,000 per QALY. Its incremental net benefit is Normal, so EVPI,
  per-parameter EVPPI, and the EVSI of a proposed effect study have closed forms
  via the unit normal loss integral. Every estimate lands within about one percent
  of its closed form at 100,000 iterations, and a test asserts the EVPI, the EVPPI
  ranking, and the EVSI against those closed forms as a second published reference
  point ([#20](https://github.com/pedroliman/heormodel/pull/20)).

- Markov vs microsimulation cross-validation: `examples/markov_vs_microsim.py` and
  a website tutorial build one Sick-Sicker-style model twice, as a `MarkovModel`
  cohort trace and a `MicrosimModel` individual simulation from the same rates.
  The homogeneous microsimulation mean converges to the cohort trace within a
  fraction of a percent at 40,000 individuals, cross-validating the two engines.
  A mean-1 frailty on the progression and mortality hazards then raises the
  microsimulation QALYs about 8% above the cohort on unchanged mean rates, the
  risk heterogeneity a cohort averages away, with duration-dependent mortality as
  a short second example. A test asserts both the convergence and the divergence
  ([#18](https://github.com/pedroliman/heormodel/pull/18)).
- Parameter inputs from data in `heormodel.params`: `single_draw` wraps one named
  set of point values as a one-row draw matrix (iteration 0) for a base-case
  run, and `ParameterSet.at_means` is the same call on a distribution set's
  analytic means. `read_draws` validates a CSV path or DataFrame as a draw
  matrix, honouring an explicit `iteration` column and rejecting non-numeric
  columns. `resample_posterior` resamples a weighted parameter table into an
  unweighted draw matrix by drawing whole rows with replacement in proportion
  to the weights, so joint correlation survives. Each result flows through
  `run_psa` unchanged. `examples/parameter_inputs.py` and the parameter-inputs
  tutorial run all three end to end
  ([#13](https://github.com/pedroliman/heormodel/pull/13)).
- Deterministic sensitivity analysis: `heormodel.dsa` builds scenario designs that
  run through `run_psa` unchanged. `one_way` sweeps a single parameter,
  `one_at_a_time` sweeps each parameter in turn (the tornado design), and `grid`
  takes the full factorial of several parameters (the heatmap design). Each
  returns a `(design, descriptor)` pair: the design is a draw matrix of
  scenarios, the descriptor a tidy table naming what each scenario varied.
  `heormodel.report.tornado_data` now reads a one-way or one-at-a-time DSA result as
  well as a PSA, and `heatmap_data` reshapes a two-parameter grid into a matrix.
  `examples/dsa.py` and a website tutorial run all three forms on the
  Sick-Sicker model ([#14](https://github.com/pedroliman/heormodel/pull/14)).

### Changed

- Documentation narrative order: the tutorials now climb from the analysis
  layer to the most detailed engine. The sequence is bring your own outputs,
  the Markov cohort model, the microsimulation engine, Markov vs
  microsimulation models, discrete-event simulation, the full pipeline, and the
  calibration workflow. The Markov cohort tutorial moves from Replications into
  Tutorials; the time-dependent cohort and microsimulation replications stay
  under Replications as validation exhibits. Each tutorial's forward link,
  `get-started.qmd`, and the README follow the same order
  ([#19](https://github.com/pedroliman/heormodel/pull/19)).
- `run_psa` runs in parallel over all cores by default (`n_jobs=-1`). Pass
  `sequential=True` for an in-process run (the readable off switch for
  debugging and reproducibility checks), or `n_jobs` for an explicit worker
  count; a run with one iteration or one available core falls back to
  sequential. The numbers are identical whichever way the run is split,
  because each iteration is seeded by its index. A `progress` argument shows a
  completed-count and time-remaining readout on `stderr` as experiments
  finish, driven by the mean throughput of finished work; it is on when
  `stderr` is a terminal and quiet otherwise, so CI logs and docs builds stay
  silent unless `progress=True` is explicit
  ([#15](https://github.com/pedroliman/heormodel/pull/15)).

### Changed

- The `Intervention` value object, and one canonical spelling for interventions
  ([#28](https://github.com/pedroliman/heormodel/issues/28)). `interventions=` now
  accepts a sequence of names or `Intervention(name, decision_levers={})` objects
  everywhere; a bare string is shorthand for a decision-lever-free intervention.
  The name-to-decision-levers mapping form is removed, so there is exactly one
  canonical spelling plus the string shorthand. The microsimulation examples and
  tutorials now branch on the intervention name for the treatment arm instead of
  encoding it as an `{"on_treatment": 1.0}` pseudo-parameter, which kept a float
  flag out of the `ParameterSet` where `evppi_ranking` and the deterministic
  sensitivity builders could not see it. Decision levers stay for genuine numeric
  scenario knobs such as a discrete-event server count. `Timeline` and
  `Population` value objects are deferred until a new engine lands (noted in
  `devdocs/architecture.md`).

- The runner owns execution, a clean break with no aliases
  ([#28](https://github.com/pedroliman/heormodel/issues/28)). Seeding and the
  event side channel move out of the engines and into `run_psa`:
  - Stochastic engines (`MicrosimModel`, `DESModel`) no longer take
    `seed_manager` at construction, and `evaluate(draws)` no longer takes a
    `trace` flag. `evaluate(draws)` returns `Outcomes` only, matching the
    `ModelEngine` protocol, so the `# type: ignore` the discrete-event example
    needed is gone.
  - `run_psa` gains `seed` and `collect` and returns a `RunResult` with
    `outcomes`, `events`, and `individuals` fields (unset fields are `None`).
    Read the panel with `run_psa(model, draws, seed=...).outcomes`. `RunResult`
    is deliberately not iterable, so it cannot be mistaken for the old
    `(outcomes, trace)` tuple. `collect="events"` gathers the state-change or
    resource history and `collect="individuals"` the per-individual accruals,
    working identically for every engine and in parallel.
  - `run_psa` builds the per-iteration streams from `seed` and hands them to a
    stochastic engine through the wider `StochasticEngine` protocol
    (`evaluate_streamed(draws, streams=..., collect=...) -> EngineResult`);
    deterministic engines keep the plain `ModelEngine` shape and ignore seeding.
    Because streams stay keyed by iteration index, outcomes and the collected
    log are invariant to `batch_size` and `n_jobs`. `SeedManager` stays public
    (it is the record `capture_run` stores) but is no longer a constructor
    argument. Stream derivation reproduces the previous numbers exactly, so the
    replication gallery is unchanged.

- One vocabulary across the engines, a clean break with no aliases
  ([#28](https://github.com/pedroliman/heormodel/issues/28)). Every engine now
  spells a shared concept the same way:
  - `MarkovModel`: `model_fn` is now `transitions_and_rewards`, `start` is now
    `initial_state`, and `half_cycle_correction` is now `cycle_correction` (its
    `"half-cycle"` option is now `"half_cycle"`).
  - `MicrosimModel` no longer takes a `clock` argument. Build the two kernels
    through `MicrosimModel.discrete(...)` and `MicrosimModel.continuous(...)`;
    each constructor carries only its kernel's parameters, so a parameter that
    belongs to the other clock is a `TypeError` at the call site. The discrete
    clock takes `n_cycles` (was `horizon`) and per-cycle `state_rewards` (was
    `state_costs_and_utilities`); the continuous clock takes `horizon` and
    per-year `state_reward_rates`. `half_cycle_correction=True` is now
    `cycle_correction="half_cycle"`, sharing the string with `MarkovModel`.
  - `DESModel`: `entities` is now `population` and `n_entities` is now
    `n_individuals`, matching `MicrosimModel`.
  - Every engine accepts `interventions` as either a sequence of names or a
    name-to-decision-levers mapping, and every user-supplied model function receives
    the intervention name. Microsimulation callbacks gain it as their second
    argument: `transition_probabilities(params, intervention, state, attrs, rng)`,
    `state_rewards(params, intervention, state, attrs)`, and
    `event_times(params, intervention, state, attrs, rng)`.
  - A discrete-event `process` reads the run's horizon back as `toolkit.horizon`
    instead of duplicating it as a module constant.
  The replication examples reproduce their published numbers unchanged, since
  nothing numeric changed.

### Fixed

- `heormodel.__version__` reported `0.0.0` on every install because the package
  read its version from `version("heval")`, the pre-rename distribution name.
  It now reads `version("heormodel")`. The stale `heval` name is also gone from
  every module docstring, cross-reference, and error message in `src/`, so the
  optional-dependency hints now point at `heormodel[des]` and
  `heormodel[calibration]` ([#28](https://github.com/pedroliman/heormodel/issues/28)).

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
  ([#11](https://github.com/pedroliman/heormodel/pull/11)).

## [0.5.0] - 2026-07-05

### Added

- Cohort state-transition engine: `MarkovCohortEngine` sweeps a cohort trace
  across PSA iterations and emits `Outcomes`. Transitions may be one matrix or a
  per-cycle array (age-varying rates); rewards accrue per state and, optionally,
  per transition (a one-time cost of dying or disutility of onset). Supports
  Simpson's 1/3, half-cycle, or no within-cycle correction, reusing
  `heormodel.models._accrual` for discounting. `CohortSpec` carries one intervention's
  matrices, and `gen_wcc` builds the correction weights
  ([#9](https://github.com/pedroliman/heormodel/pull/9)).
- `duration_groups` on `DiscreteTimeMicrosimEngine`: a per-individual counter of
  consecutive cycles spent in a set of states, so a sojourn that progresses
  (Sick to Sicker) keeps counting where `time_in_state` would reset
  ([#9](https://github.com/pedroliman/heormodel/pull/9)).
- Three replications of published Sick-Sicker cost-effectiveness tutorials, each
  matching the source's deterministic results, with runnable scripts and website
  tutorials: cohort state-transition (`examples/mdm_cohort.py`), time-dependent
  cohort with age-varying mortality (`examples/mdm_cohort_timedep.py`), and
  microsimulation (`examples/mdm_microsim.py`). A replication gallery page
  collects them with citations ([#9](https://github.com/pedroliman/heormodel/pull/9)).

## [0.4.0] - 2026-07-04

### Added

- Discrete-event simulation engine: `DESEngine` wraps SimPy. The environment,
  process functions, and resources stay the user's own code; the engine adds a
  per-entity toolkit for discounted cost and utility accrual, per-iteration
  seeding from a `SeedManager` so results do not depend on `n_jobs`, and an
  optional event log. It reuses `heormodel.models._accrual` and uses common random
  numbers across interventions by default, staying coherent with the
  microsimulation engines ([#8](https://github.com/pedroliman/heormodel/pull/8)).
- `heormodel.models.queue_waits`: derive per-request waiting times from a `DESEngine`
  trace, so queueing reports come from the event log rather than engine
  internals ([#8](https://github.com/pedroliman/heormodel/pull/8)).
- `simpy` as an optional dependency behind the `des` extra
  (`uv pip install 'heormodel[des]'`) ([#8](https://github.com/pedroliman/heormodel/pull/8)).
- Discrete-event example (`examples/des.py`) and website tutorial, validated
  against an M/M/1 queue and the exponential cohort solution
  ([#8](https://github.com/pedroliman/heormodel/pull/8)).

## [0.3.0] - 2026-07-04

### Added

- Microsimulation engines: `DiscreteTimeMicrosimEngine` advances an
  individual-level population on a cycle grid with history-dependent
  transitions and heterogeneity, and `ContinuousTimeMicrosimEngine` races
  competing time-to-event samplers between events. Both emit the standard
  `Outcomes` schema, seed each iteration from a `SeedManager` so results do not
  depend on `n_jobs`, and use common random numbers across interventions by
  default ([#7](https://github.com/pedroliman/heormodel/pull/7)).
- `heormodel.models._accrual`: shared cost and utility accrual, discounting, and
  aggregation to `Outcomes`, used by both engines and reserved for the
  discrete-event engine ([#7](https://github.com/pedroliman/heormodel/pull/7)).
- `SeedManager.child_sequence` returns a per-key seed sequence, so
  iteration-indexed streams stay identical however a run is chunked across
  workers ([#7](https://github.com/pedroliman/heormodel/pull/7)).
- Microsimulation example (`examples/microsim.py`) and website tutorial,
  validated against the closed-form cohort solution it mirrors
  ([#7](https://github.com/pedroliman/heormodel/pull/7)).

## [0.2.0] - 2026-07-04

### Added

- Documentation website built with Quarto and quartodoc, published to GitHub
  Pages from CI on every merge to `main`: executed tutorials, concept pages, a
  generated API reference, the roadmap, and this changelog
  ([#3](https://github.com/pedroliman/heormodel/pull/3)).
- `heormodel.params.mix_draws` combines draw matrices from different sources
  (a calibrated posterior and literature draws) into one PSA matrix,
  resampling whole rows so joint correlation survives and sources stay
  independent ([#5](https://github.com/pedroliman/heormodel/pull/5)).
- `capture_run` records a `draw_sources` map, so the run report shows where
  each parameter's draws came from
  ([#5](https://github.com/pedroliman/heormodel/pull/5)).
- Calibration workflow example (`examples/calibration_workflow.py`) and
  website tutorial: calibrate a natural-history model's rates, mix them with
  literature parameters, and run CEA and VoI on the result
  ([#5](https://github.com/pedroliman/heormodel/pull/5)).

### Changed

- Docstring cross-references use plain backticks instead of Sphinx roles, so
  they render cleanly on the website and in `help()`
  ([#3](https://github.com/pedroliman/heormodel/pull/3)).
- Distribution spec strings format floats at 6 significant digits, so
  provenance records and run reports stay readable
  ([#3](https://github.com/pedroliman/heormodel/pull/3)).
- `RunRecord.model_card()` renamed to `to_markdown()`, titled "Run report":
  "model card" is ML documentation jargon, not a HEOR term
  ([#6](https://github.com/pedroliman/heormodel/pull/6)).

## [0.1.0] - 2026-07-04

Initial release: parameter sampling (`heormodel.params`), the `Outcomes` schema
and `ModelEngine` protocol (`heormodel.models`), the PSA run loop and
bring-your-own-outputs ingestion (`heormodel.run`), cost-effectiveness analysis
(`heormodel.cea`), value-of-information analysis (`heormodel.voi`), optional ABC
calibration (`heormodel.calibrate`), and reporting plots (`heormodel.report`).

[Unreleased]: https://github.com/pedroliman/heormodel/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/pedroliman/heormodel/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/pedroliman/heormodel/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/pedroliman/heormodel/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/pedroliman/heormodel/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/pedroliman/heormodel/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pedroliman/heormodel/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pedroliman/heormodel/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pedroliman/heormodel/releases/tag/v0.1.0
