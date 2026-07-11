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

## Zenodo archive

Every GitHub release also archives to [Zenodo](https://zenodo.org) with a
DOI, once the one-time setup below is done. No workflow step is needed:
Zenodo's GitHub integration registers its own repository webhook (under
`Settings` -> `Webhooks`) that fires on the `release: published` event
directly, independent of GitHub Actions. That event still fires for
releases `tag-release.yml` creates with the default `GITHUB_TOKEN`: the
"`GITHUB_TOKEN` doesn't trigger other workflows" restriction (see step 5
above) only suppresses new *Actions* workflow runs, not repository
webhooks, which is what the Zenodo integration relies on.

Zenodo reads archive metadata from [`.zenodo.json`](.zenodo.json); GitHub's
own "Cite this repository" button reads [`CITATION.cff`](CITATION.cff).
When both files are present, Zenodo uses `.zenodo.json` and ignores
`CITATION.cff`. Update `.zenodo.json` if the title, authors, or keywords
change; there is no version field to keep in sync; each archived record
takes its version from the release tag.

One-time setup, done by the repository owner on zenodo.org (cannot be
scripted, requires the owner's GitHub OAuth login):

1. Sign in at [zenodo.org](https://zenodo.org) with the GitHub account that
   owns this repository, and open the
   [GitHub integration settings](https://zenodo.org/account/settings/github/).
2. Click "Sync now" if `heormodel` is not in the repository list yet.
3. Toggle `heormodel` on. This is a one-time step; every release published
   afterward archives automatically.
4. After the first archive completes, copy the "concept DOI" badge Zenodo
   shows for the repository (the one that always resolves to the latest
   version) into the README's badge row.

## Checks before releasing

`CI` (`.github/workflows/ci.yml`) runs `ruff`, `mypy`, `pytest`, and the
doctest suite on every push and PR to `main`. Do not tag a release on top of
a red `main`.
