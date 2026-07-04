# Changelog

All notable changes to `heval` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/).

Each entry links to the pull request that introduced it. Add a line under
`[Unreleased]` in the same PR that makes the change; see
[RELEASING.md](RELEASING.md) for how `[Unreleased]` turns into a release.

## [Unreleased]

### Added

- Documentation website built with Quarto and quartodoc, published to GitHub
  Pages from CI on every merge to `main`: executed tutorials, concept pages, a
  generated API reference, the roadmap, and this changelog
  ([#3](https://github.com/pedroliman/heval/pull/3)).

### Changed

- Docstring cross-references use plain backticks instead of Sphinx roles, so
  they render cleanly on the website and in `help()`
  ([#3](https://github.com/pedroliman/heval/pull/3)).
- Distribution spec strings format floats at 6 significant digits, so
  provenance records and model cards stay readable
  ([#3](https://github.com/pedroliman/heval/pull/3)).

## [0.1.0] - 2026-07-04

Initial release: parameter sampling (`heval.params`), the `Outcomes` schema
and `ModelEngine` protocol (`heval.models`), the PSA run loop and
bring-your-own-outputs ingestion (`heval.run`), cost-effectiveness analysis
(`heval.cea`), value-of-information analysis (`heval.voi`), optional ABC
calibration (`heval.calibrate`), and reporting plots (`heval.report`).

[Unreleased]: https://github.com/pedroliman/heval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pedroliman/heval/releases/tag/v0.1.0
