# Releasing heval

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
4. Commit as `Release vX.Y.Z`, push to `main`.
5. Tag the release commit and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

6. Create a GitHub release from that tag (`Releases` -> `Draft a new
   release`), using the matching CHANGELOG section as the release notes.
   Publishing the release triggers the `Release` workflow
   (`.github/workflows/release.yml`), which builds the package with `uv
   build` and publishes it to PyPI.

## One-time PyPI setup

The release workflow publishes via
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/), no API
token stored in the repo. On the `heval` project's PyPI page, under
`Publishing`, add a trusted publisher for this repository:

- Owner: `pedroliman`, repository: `heval`
- Workflow file: `release.yml`
- Environment: `pypi`

## Checks before releasing

`CI` (`.github/workflows/ci.yml`) runs `ruff`, `mypy`, `pytest`, and the
doctest suite on every push and PR to `main`. Do not tag a release on top of
a red `main`.
