# Writing style guide

Applies to everything written in this repository: README, roadmap, docstrings, comments, commit messages, website pages, error messages.

## Be concise

- Lead with the point. Cut warm-up sentences.
- One idea per sentence. Prefer short sentences. Keep every paragraph under 5 sentences.
- Cut words that add no information: "it is worth noting that", "in order to", "as mentioned above".
- Prefer a number or an example over an adjective. "Agrees within 5% at 80,000 iterations" beats "highly accurate".
- If a paragraph restates the previous one, delete it.

## Punctuation and formatting

- No em-dashes. Use a comma, a colon, parentheses, or a new sentence.
- No exclamation marks. No emoji in prose (table status marks are fine).
- Never introduce line breaks unnecessarily in markdown or quarto prose. Write each paragraph or bullet as one line and let word wrap do its thing. Code and docstrings follow the linter's line length instead.
- Sentence case for headings.
- Backticks for code identifiers: `run_psa`, `Outcomes`, `iteration`.
- Bullets only for parallel items. Use prose for reasoning.
- Bold sparingly, for terms a reader scans for, not for emphasis.

## Documentation pages

- Each doc page stays under 500 words. If a page grows past that, split it or cut it.
- Quarto pages link to each other rather than repeating content. The first paragraph of a page situates it relative to existing pages: what it covers, what it assumes, where to go next.
- For quarto notebooks or any doc that executes code, render it and verify the prose matches the code output, including plots, after writing.

## Vocabulary

Use the field's terms:

- strategy, comparator, willingness-to-pay threshold, ICER, QALY, net monetary benefit (NMB), net health benefit (NHB), efficiency frontier, dominance, extended dominance, CEAC, CEAF
- probabilistic sensitivity analysis (PSA), parameter draw, iteration
- cohort model, state-transition model, microsimulation, discrete-event simulation, cycle, half-cycle correction, discounting, time horizon
- calibration target, prior, posterior, EVPI, EVPPI, EVSI

Avoid vocabulary that pattern-matches to generated text and does not belong in HEOR writing: leverage, delve, seamless, comprehensive, robust (unless statistical robustness is meant), crucial, empower, journey, cutting-edge, holistic, streamline, unlock.

Do not mention external existing R packages anywhere.

## Tone

- Plain and direct. Write for a health economist who codes, not for a press release.
- Active voice by default: "the engine returns `Outcomes`".
- State limitations plainly. "Ties are broken by first occurrence" beats hedging or silence.
- No rhetorical questions, no triads for rhythm ("fast, simple, and powerful"), no closing summaries that restate the section.

## Docstrings

- Google style. First line: one sentence, imperative or declarative, under 80 characters.
- Every public function has a worked `Example:` block that runs under `pytest --doctest-modules`.
- Describe arguments by what they mean in the analysis, not by type (the signature gives the type).

## Commit messages

- Subject under 70 characters, imperative mood.
- Body says what changed and why, not how (the diff shows how).
