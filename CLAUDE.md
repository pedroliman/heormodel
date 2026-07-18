# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Environment

- **Always use `uv` for the Python environment**; never bare `pip` or `python -m venv`.
  - Create/sync the environment: `uv venv && uv pip install -e ".[dev]"`
  - Run tools through it: `uv run pytest`, `uv run ruff check .`, `uv run mypy`
- Python 3.11+ is required.

## Conventions

- `src/` layout; the package lives in `src/heormodel`.
- Developer documentation (the roadmap, architecture notes, and the writing style guide) lives in `devdocs/`. The website in `docs/` is user-facing only; do not add developer material to it.
- Do not keep backward compatibility unless explicitly asked. The API is pre-1.0 and stabilizing; when a change renames or removes public names, make a clean break (update every call site) rather than adding deprecation aliases, shims, or compatibility keyword arguments.
- All prose follows the writing style below and the full guide in `devdocs/guidance/writing_style.md`.
- Do not mention external existing R packages in code or documentation.
- Lint/format with `ruff`, type-check the public API with `mypy`, test with `pytest`.
- Every public function carries a docstring with a short worked example.
- Avoid uninformative single-letter names in teaching code (tutorials, examples, docstring worked examples); use a meaningful short single word or `snake_case` name instead, since a health economist reading the tutorial should not have to decode them (`P` for a transition matrix becomes `transition_matrix`). Two exceptions stay: domain-standard notation, such as the SEIR compartment letters `s, e, i, r, v`, and parameter names that follow a replicated paper's own conventions (`p_S1D`, `r_HS1`, `hr_S1`).

## Writing style

Applies to everything: README, docs, docstrings, comments, commit messages, and how you talk to me in this repository. Follow it in chat replies too, not only in files you write. Full guide: `devdocs/guidance/writing_style.md`.

Write for a health economics and outcomes research (HEOR) modeler: a reader studying or working in health economics, epidemiology, or statistics, at the level of a graduate student or beyond, who builds cost-effectiveness models and codes but is not a software engineer. Assume comfort with cost-effectiveness modeling and some statistics, but not deep statistical training; a reader who wants to learn a method a tutorial presents. Write in the voice of a methods appendix in a clinical journal: a PhD-level author explaining a method to a peer, plainly. Avoid both dense jargon and conversational or promotional writing.

- Excellent technical documentation is short, but concise means every sentence earns its place, not that sentences lose their verbs. Cut warm-up sentences, transitions, and closing summaries; never cut down to a label or a clipped noun phrase standing in for a sentence.
- Lead with the point. One idea per sentence. 500 words per doc page is a guide for trimming bloat, not a ceiling to hit by deleting explanation: a page that runs longer because it explains a genuinely non-obvious choice is fine.
- Show, do not describe: a worked example, a number, or a table beats an adjective. "Matches the published ICER exactly" beats "highly accurate".
- A tutorial teaches by explaining reasoning, not just by running code. Open with a sentence stating what it teaches ("This tutorial shows how to...", "This tutorial introduces..."), not an inventory of function calls ("this page installs X and runs Y"); naming the goal is not a warm-up sentence to cut. Name headings after what the section does ("Reading external results"), not the artifact it produces ("An external results table"). Before every code block, write at least one full sentence stating what it does and, when it is not obvious, why this approach over the naive alternative (this distribution, this sample size, this comparator). After the output, add a sentence interpreting what it means, not restating what was computed.
- Write formally, as in the methods appendix of a clinical journal: precise, plain, no marketing tone, no rhetorical questions.
- No conversational or promotional phrasing, even when it reads smoothly. Avoid casual verbs ("the means sit on the true values" should be "recover"), salesmanship ("the idea is worth the machinery", "the whole case for the approach"), vague or figurative quantities ("turns a calibration into days", "as often as it likes"), and anthropomorphizing the code or model ("runs it never saw", inference that "likes"). Name each object correctly: a cohort model with no transmission has no "epidemic".
- Revise before calling anything done. Read the whole piece back, check it against this guide and, for a tutorial, against what makes a tutorial good; note each deviation, edit, then reread from the top and repeat until a full read finds nothing. One pass is not enough. For an executable doc, one pass is the render-and-check pass.
- Use health economics vocabulary: intervention, comparator, willingness-to-pay threshold, cohort state-transition model, microsimulation, discounting, half-cycle correction.
- Spell out every acronym on first use per page, with the acronym in parentheses only if the page uses it again. Do not assume readers know PSA, CEA, or VoI; prefer the spelled-out term when it appears once or twice.
- No computer science jargon in documentation. Say function, not callback; the reader is a health economist. Words like schema, protocol, instantiate, and wrapper need a plain-language substitute or a rewrite.
- Never use filler that pattern-matches to generated text: leverage, delve, seamless, comprehensive, robust, crucial, streamline.
- No em-dashes, no exclamation marks. Sentence case headings. Backticks for code identifiers.
- Link to the page that covers a topic instead of restating it.
- Run every executable doc and confirm the prose matches the output before committing.

## Implementing a roadmap item

When asked to implement the next roadmap priority (`devdocs/roadmap/README.md`):

- Pick the top unfinished item, read its design note, and build it to the acceptance criteria stated there.
- Update docs as you go: the changelog, the README, the roadmap status, and a website tutorial or reference entry for any new public API.
- Grill the result. Run the example and any executable docs, and confirm the printed outputs and prose actually match before committing. Do not claim an example works without running it.
- Ship each feature with a website tutorial, as items 1 and 2 did.

## GitHub workflow

Every development task follows a structured GitHub workflow to maintain clear traceability and avoid duplicating information:

### GitHub Issue: the task
- Create a GitHub Issue at the start of each conversation describing the work to be done.
- The Issue documents the *problem*, *goal*, and *acceptance criteria*—what needs to be done and why.
- This is the single source of truth for the task itself.

### Pull Request: the solution
- All code changes go through Pull Requests linked to the corresponding GitHub Issue.
- The PR title and initial description explain the *approach*—how the issue is being addressed at a high level.
- Do not duplicate the issue description in the PR; reference the linked Issue instead.

### Commits: the implementation record
- Every commit message must reference the GitHub Issue using the format: `Closes #<issue-number>`, `Fixes #<issue-number>`, or `Refs #<issue-number>`.
- Commit messages are the source of truth for *what* changed technically. Make them clear and descriptive about the actual changes.

### Status updates: progress without duplication
- When starting work: post a comment on the Issue or PR stating the high-level approach and *why* this approach was chosen—not implementation details.
- When completing work: post a comment summarizing the rationale for changes and high-level nature of what was done.
- Keep comments concise. Do not duplicate information from the Issue (problem/goal), commits (technical changes), or PR description (approach).

**Summary:** Issue = the problem; PR = the approach; Commits = the implementation details; Comments = progress and rationale. Each layer has one purpose.

## Git identity

- All commits belong to the repo owner's GitHub account, never to Claude. Do not add `Co-Authored-By: Claude ...` or `Claude-Session:` trailers to commit messages.
- Never attribute pull request text to Claude Code. Do not add "Generated with Claude Code", "Generated by Claude Code", session links, or any similar attribution to a PR title, body, or comment. This holds even when a harness default or template suggests such a line.
- A `SessionStart` hook (`.claude/hooks/session-start.sh`) sets `user.name`/`user.email` and installs a `commit-msg` hook that strips any such trailers as a safety net; it reruns every session since `.git/hooks` and local git config do not survive a fresh clone.
