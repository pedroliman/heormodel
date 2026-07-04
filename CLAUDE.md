# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Environment

- **Always use `uv` for the Python environment**; never bare `pip` or `python -m venv`.
  - Create/sync the environment: `uv venv && uv pip install -e ".[dev]"`
  - Run tools through it: `uv run pytest`, `uv run ruff check .`, `uv run mypy`
- Python 3.11+ is required.

## Conventions

- `src/` layout; the package lives in `src/heval`.
- Follow `guidance/writing_style.md` for all prose: README, roadmap, docstrings, comments, commit messages, and website pages. In short: concise, no em-dashes, HEOR vocabulary.
- Do not mention external existing R packages in code or documentation.
- Lint/format with `ruff`, type-check the public API with `mypy`, test with `pytest`.
- Every public function carries a docstring with a short worked example.

## Implementing a roadmap item

When asked to implement the next roadmap priority (`roadmap/README.md`):

- Pick the top unfinished item, read its design note, and build it to the acceptance criteria stated there.
- Update docs as you go: the changelog, the README, the roadmap status, and a website tutorial or reference entry for any new public API.
- Grill the result. Run the example and any executable docs, and confirm the printed outputs and prose actually match before committing. Do not claim an example works without running it.
- Ship each feature with a website tutorial, as items 1 and 2 did.

## Git identity

- All commits belong to the repo owner's GitHub account, never to Claude. Do not add `Co-Authored-By: Claude ...` or `Claude-Session:` trailers to commit messages.
- A `SessionStart` hook (`.claude/hooks/session-start.sh`) sets `user.name`/`user.email` and installs a `commit-msg` hook that strips any such trailers as a safety net; it reruns every session since `.git/hooks` and local git config do not survive a fresh clone.
