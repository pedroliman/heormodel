# Writing style guide

Applies to everything written in this repository: README, roadmap, docstrings, comments, commit messages, website pages, error messages.

## Be concise, not clipped

Concise means every sentence earns its place, not that sentences lose their verbs. Write full sentences: a subject and a verb, every time. "Three strategies for a chronic disease: standard of care, a new drug, and drug plus monitoring" is a fragment, a label standing in for a sentence. "Three strategies compete for a chronic disease: standard of care, a new drug, and drug plus monitoring" is a sentence, and no longer than the fragment was. If trimming a sentence would strip its verb, the sentence was doing real work; cut a different sentence instead, or leave it whole.

- Lead with the point. Cut warm-up sentences, not verbs.
- One idea per sentence. Prefer short sentences over long ones, never over incomplete ones. Keep every paragraph under 5 sentences.
- Cut words that add no information: "it is worth noting that", "in order to", "as mentioned above".
- Prefer a number or an example over an adjective. "Agrees within 5% at 80,000 iterations" beats "highly accurate".
- If a paragraph restates the previous one, delete it. If a sentence explains a choice a reader would otherwise have to guess at, keep it. That explanation is content, not filler, and word-count pressure is never a reason to cut it.

## What makes a tutorial good

A tutorial's job is to let a reader replicate the analysis on their own problem, not just to prove the package runs. That means the reasoning behind a choice matters as much as the code: which distribution, which sample size, which comparator, and why that one rather than the obvious alternative.

Open with a sentence that states what the tutorial teaches, in those terms: "This tutorial shows how to compare three strategies with probabilistic sensitivity analysis" or "This tutorial introduces `MicrosimModel`, the engine for representing patient history and heterogeneity that a cohort model averages away." Naming the goal is not a warm-up sentence to cut; it is the one sentence that tells the reader whether this is the page they need, and "lead with the point" means this sentence, stated plainly, rather than an inventory of function calls ("this page installs X and runs Y"). A list of actions is not the same as a statement of purpose: a reader can see the actions in the code below; what they cannot see is why this page exists or what they will be able to do afterward.

After that opening, structure every section the same way:

1. A heading that names what the section does, not the artifact it produces or introduces. "Reading external results" beats "An external results table"; "Ranking parameters by their value of information" beats "One call into the standard structure". A reader scanning the table of contents should be able to tell what happens at each step.
2. At least one full sentence before the first code block in the section. State what the code is about to do and, whenever it is not obvious from the code itself, why: why this distribution, why this sample size, why this comparator, why this threshold. A code block with no lead-in reads as a dump, not a tutorial.
3. A sentence or two after the output that interprets it: what the number or plot means for the decision, not a restatement of what was just computed.

Do not narrate self-evident code line by line ("first we import pandas, then we define a function"), and do not let the drive for brevity turn an explanation, or the opening statement of purpose, into a fragment or a bare action list. A tutorial section that is a few sentences longer because it explains a genuinely non-obvious choice is better than one that hits a word target by cutting that explanation.

## Punctuation and formatting

- No em-dashes. Use a comma, a colon, parentheses, or a new sentence.
- No exclamation marks. No emoji in prose (table status marks are fine).
- Never introduce line breaks unnecessarily in markdown or quarto prose. Write each paragraph or bullet as one line and let word wrap do its thing. Code and docstrings follow the linter's line length instead.
- Sentence case for headings.
- Backticks for code identifiers: `run_psa`, `Outcomes`, `iteration`.
- Bullets only for parallel items. Use prose for reasoning.
- Bold sparingly, for terms a reader scans for, not for emphasis.

## Documentation pages

- 500 words is a guide for trimming bloat, not a hard ceiling to hit by deleting explanation. If a page runs longer because it walks through several worked steps or explains genuinely non-obvious choices, let it; if it runs longer because of throat-clearing or repetition, cut that instead. A page padded with filler to reach 500 words is as wrong as one that cuts reasoning to stay under it.
- Quarto pages link to each other rather than repeating content. The first paragraph of a page situates it relative to existing pages: what it covers and where to go next. Do not state what it "assumes"; either the page is self-contained or it links to the page that supplies the missing piece.
- For quarto notebooks or any doc that executes code, render it and verify the prose matches the code output, including plots, after writing.

## Vocabulary

Use the field's terms:

- strategy, comparator, willingness-to-pay threshold, incremental cost-effectiveness ratio (ICER), quality-adjusted life-year (QALY), net monetary benefit, net health benefit, efficiency frontier, dominance, extended dominance, cost-effectiveness acceptability curve
- probabilistic sensitivity analysis, parameter draw, iteration
- cohort model, state-transition model, microsimulation, discrete-event simulation, cycle, half-cycle correction, discounting, time horizon
- calibration target, prior, posterior, expected value of perfect information (EVPI), expected value of partial perfect information (EVPPI), expected value of sample information (EVSI)

Spell out every acronym on first use per page, with the acronym in parentheses only if the page uses it again. Do not assume readers know PSA, CEA, or VoI; when a term appears once or twice, use the words instead of the acronym.

No computer science jargon in documentation. The reader is a health economist who codes, not a software engineer. Say function, not callback or callable; say table, not schema or frame; describe what a thing does instead of naming the design pattern (protocol, wrapper, decorator, factory). Python names visible in code (`DataFrame`, `dict`) are fine inside backticks.

Avoid vocabulary that pattern-matches to generated text and does not belong in health economics writing: leverage, delve, seamless, comprehensive, robust (unless statistical robustness is meant), crucial, empower, journey, cutting-edge, holistic, streamline, unlock.

Avoid startup and business-strategy jargon for the same reason: wedge, adoption wedge, go-to-market, north star, growth loop, and similar terms describe product strategy, not a cost-effectiveness analysis. If a plain description ("this tutorial covers the same analysis") says the same thing, use that instead.

Do not describe a tutorial or example script as "narrated" (a "narrated version", "narrated analysis," and so on). Say what the page actually does instead: "walks through `examples/x.py` step by step", or link the script and say nothing more.

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
