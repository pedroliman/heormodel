# Releasing heormodel

How to cut a release after a PR merges to `main`. Release often: a single
bug fix or small feature is enough to justify a patch release.

## After every merge to main

Every PR that changes behavior adds one line under `## [Unreleased]` in
[CHANGELOG.md](CHANGELOG.md), linking to the PR. Nothing else is required;
`main` is always releasable.

## Cutting a release

1. Pick the version bump under [semantic versioning](https://semver.org/):
   `patch` for fixes, `minor` for backward-compatible features, `major` for
   breaking changes.
2. On `main`, update `version` in `pyproject.toml`.
3. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`,
   add a fresh empty `## [Unreleased]` above it, and update the link
   references at the bottom of the file.
4. Commit as `Release vX.Y.Z`, merge to `main` through a pull request.
5. On merge, the `Tag release` workflow
   (`.github/workflows/tag-release.yml`) reads the version from
   `pyproject.toml`, tags the merge commit `vX.Y.Z`, creates a GitHub release
   from the matching `CHANGELOG.md` section, and (in the same workflow run)
   builds the package with `uv build` and publishes it to PyPI. The publish
   step runs in the same workflow deliberately: a release created with the
   default `GITHUB_TOKEN` does not trigger other workflows, so a separate
   `release: published`-triggered job would silently never run.
6. If a publish is ever stuck for a tag that already has a GitHub release
   (check the `Tag release` and `Release` workflow runs in the `Actions` tab),
   trigger `.github/workflows/release.yml` manually: `Actions` -> `Release` ->
   `Run workflow`, entering the tag (e.g. `v0.7.0`).

## One-time PyPI setup

The release workflow publishes via
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/), no API
token stored in the repo. On the `heormodel` project's PyPI page, under
`Publishing`, add a trusted publisher for this repository:

- Owner: `pedroliman`, repository: `heormodel`
- Workflow file: `release.yml`
- Environment: `pypi`

## Checks before releasing

`CI` (`.github/workflows/ci.yml`) runs `ruff`, `mypy`, `pytest`, and the
doctest suite on every push and PR to `main`. Do not tag a release on top of
a red `main`.
