# Writing style guide

Applies to everything written in this repository: README, roadmap, docstrings, comments, commit messages, website pages, error messages.

## Who reads this, and the voice to write in

Write for a health economics and outcomes research (HEOR) modeler: a reader studying or working in health economics, epidemiology, statistics, or a related quantitative field, at the level of a graduate student or beyond, who builds cost-effectiveness models and writes code to do it, but who is not a software engineer. Assume this reader already knows what a cohort state-transition model, a quality-adjusted life-year, a prior, and a posterior are, and does not need undergraduate probability re-explained. Do not assume this reader knows software-engineering vocabulary or wants it.

Write in the voice of the methods section or statistical appendix of a clinical or health-economics journal: a PhD-level author explaining a method to a peer, in plain and precise language. Two registers are wrong for this repository, in opposite directions. One is dense jargon and formality for its own sake, which the Vocabulary section below rules out. The other is conversational or promotional writing: casual idiom, figures of speech, and salesmanship about the method. A tutorial here is a written scientific methods section, not a blog post, a conference talk, or a product page. When a sentence would sound out of place in a journal's methods appendix, it is out of place here.

## Be concise, not clipped

Concise means every sentence earns its place, not that sentences lose their verbs. Write full sentences: a subject and a verb, every time. "Three interventions for a chronic disease: standard of care, a new drug, and drug plus monitoring" is a fragment, a label standing in for a sentence. "Three interventions compete for a chronic disease: standard of care, a new drug, and drug plus monitoring" is a sentence, and no longer than the fragment was. If trimming a sentence would strip its verb, the sentence was doing real work; cut a different sentence instead, or leave it whole.

- Lead with the point. Cut warm-up sentences, not verbs.
- One idea per sentence. Prefer short sentences over long ones, never over incomplete ones. Keep every paragraph under 5 sentences.
- Cut words that add no information: "it is worth noting that", "in order to", "as mentioned above".
- Prefer a number or an example over an adjective. "Agrees within 5% at 80,000 iterations" beats "highly accurate".
- If a paragraph restates the previous one, delete it. If a sentence explains a choice a reader would otherwise have to guess at, keep it. That explanation is content, not filler, and word-count pressure is never a reason to cut it.

## What makes a tutorial good

A tutorial's job is to let a reader replicate the analysis on their own problem, not just to prove the package runs. That means the reasoning behind a choice matters as much as the code: which distribution, which sample size, which comparator, and why that one rather than the obvious alternative.

Open with a sentence that states what the tutorial teaches, in those terms: "This tutorial shows how to compare three interventions with probabilistic sensitivity analysis" or "This tutorial introduces `MicrosimModel`, the engine for representing patient history and heterogeneity that a cohort model averages away." Naming the goal is not a warm-up sentence to cut; it is the one sentence that tells the reader whether this is the page they need, and "lead with the point" means this sentence, stated plainly, rather than an inventory of function calls ("this page installs X and runs Y"). A list of actions is not the same as a statement of purpose: a reader can see the actions in the code below; what they cannot see is why this page exists or what they will be able to do afterward.

After that opening, structure every section the same way:

1. A heading that names what the section does, not the artifact it produces or introduces. "Reading external results" beats "An external results table"; "Ranking parameters by their value of information" beats "One call into the standard structure". A reader scanning the table of contents should be able to tell what happens at each step.
2. At least one full sentence before the first code block in the section. State what the code is about to do and, whenever it is not obvious from the code itself, why: why this distribution, why this sample size, why this comparator, why this threshold. A code block with no lead-in reads as a dump, not a tutorial.
3. A sentence or two after the output that interprets it: what the number or plot means for the decision, not a restatement of what was just computed.

Keep the headings on a page grammatically parallel. The tutorials name each section with a gerund: "Specifying the model", "Running the analysis", "Analyzing cost-effectiveness". A lone imperative or noun phrase among them ("Run the analysis", "A model as a function") reads as an editing seam; match the form the rest of the page already uses.

Separate the values a reader will change from the logic they copy unchanged. Lift the decision inputs, the willingness-to-pay threshold, the seed, the iteration count, the time horizon, to named constants near the top of the first code block. A reader adapting the tutorial to their own problem then edits those in one place rather than hunting for them inside the model function.

Show the result the reader is working toward, and confirm they reached it. State the number or table the analysis produces ("the frontier runs standard of care, then Intervention B at about 73,000 per QALY"), and when a tutorial reproduces a published result, say plainly whether it matched ("the result should match the published table exactly. It does."). A reader needs both to see where a section is headed and to have a way to tell that their own run landed in the same place.

Do not narrate self-evident code line by line ("first we import pandas, then we define a function"), and do not let the drive for brevity turn an explanation, or the opening statement of purpose, into a fragment or a bare action list. A tutorial section that is a few sentences longer because it explains a genuinely non-obvious choice is better than one that hits a word target by cutting that explanation.

Only explain what you can actually verify. A "why" sentence earns its place when it follows directly from the code, the model's math, or an output you have checked, not from a plausible-sounding guess about the author's intent. Do not invent a rationale for a modeling choice (why this correlation, why this parameter value) unless the source material or the code supports it; if the real reason is not known, describe what the code does and stop. Do not reach for an interpretive aphorism to make a result sound more insightful than it is (a restated inequality dressed up as a conclusion, a clever-sounding turn of phrase you have not derived step by step). If you are not certain a sentence is both correct and useful, cut it rather than leave it in on the chance it reads well.

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

- intervention, comparator, willingness-to-pay threshold, incremental cost-effectiveness ratio (ICER), quality-adjusted life-year (QALY), net monetary benefit, net health benefit, efficiency frontier, dominance, extended dominance, cost-effectiveness acceptability curve
- probabilistic sensitivity analysis, parameter draw, iteration
- cohort model, state-transition model, microsimulation, discrete-event simulation, cycle, half-cycle correction, discounting, time horizon
- calibration target, prior, posterior, expected value of perfect information (EVPI), expected value of partial perfect information (EVPPI), expected value of sample information (EVSI)

Spell out every acronym on first use per page, with the acronym in parentheses only if the page uses it again. Do not assume readers know PSA, CEA, or VoI; when a term appears once or twice, use the words instead of the acronym.

No computer science jargon in documentation. The reader is a health economist who codes, not a software engineer. Say function, not callback or callable; say table, not schema or frame; describe what a thing does instead of naming the design pattern (protocol, wrapper, decorator, factory). Python names visible in code (`DataFrame`, `dict`) are fine inside backticks.

Avoid vocabulary that pattern-matches to generated text and does not belong in health economics writing: leverage, delve, seamless, comprehensive, robust (unless statistical robustness is meant), crucial, empower, journey, cutting-edge, holistic, streamline, unlock.

Avoid startup and business-intervention jargon for the same reason: wedge, adoption wedge, go-to-market, north star, growth loop, and similar terms describe product intervention, not a cost-effectiveness analysis. If a plain description ("this tutorial covers the same analysis") says the same thing, use that instead.

Do not describe a tutorial or example script as "narrated" (a "narrated version", "narrated analysis," and so on). Say what the page actually does instead: "walks through `examples/x.py` step by step", or link the script and say nothing more.

Do not mention external existing R packages anywhere.

## Tone

- Plain and direct. Write for a health economist who codes, not for a press release.
- Active voice by default: "the engine returns `Outcomes`".
- State limitations plainly. "Ties are broken by first occurrence" beats hedging or silence.
- No rhetorical questions, no triads for rhythm ("fast, simple, and powerful"), no closing summaries that restate the section.
- Use the plain, literal word over an idiom or figure of speech chosen to sound polished. Say "the model type to use", not "the engine to reach for"; say "is worth its cost", not "pays off"; say "exceeds the threshold", not "clears the threshold". If a phrase takes a second read to parse, or sounds like it is trying to be memorable rather than clear, replace it with the literal wording.

## No conversational or promotional phrasing

The voice is a written methods section, so casual idiom, figures of speech, and salesmanship do not belong, even when they read smoothly. These are the forms that most often slip in. Each example is taken from a real draft in this repository.

- Casual verbs in place of precise ones. "The posterior means sit on the true values" should read "recover the true values". "The estimates land near the target" should read "are close to the target".
- Salesmanship about the method. "The idea is worth the machinery", "that ratio is the whole case for the approach", and "this is where it pays off" sell rather than describe. State what the method does and what it costs, then let the numbers make the case: "is worth its cost when the model is slow to run", followed by the run counts.
- Vague or figurative quantities. "A model that takes a minute per run turns a calibration into days" should read "requires days to calibrate". "Let the inference call the approximation as often as it likes" should read "the inference then queries the approximation without further model runs". "To show how little the surrogate needs" should read "to show how few runs the surrogate requires".
- Anthropomorphizing the code or the model. A model does not want, a surrogate does not see the points it was trained on, and inference does not like anything. Describe what the code computes: not "model runs it never saw" but "model runs that were not in its design".
- A wrong word dressed as a phrase. Name the object correctly before worrying about how the sentence reads. The prevalence over time in a cohort model without transmission is a prevalence trajectory, not an "epidemic curve": a cohort model has no epidemic. A factual error is not excused by reading well.

## Revising your own writing

Do not treat a first draft as finished. Every time you write prose here, and every time you finish writing a tutorial for the first time, revise it in a systematic pass before considering it done:

1. Read the whole piece back, start to finish, as the intended reader would.
2. Check it against this guide point by point: the reader and voice above, the concision rules, the punctuation rules, the vocabulary list, and the conversational-phrasing list. For a tutorial, also check it against "What makes a tutorial good": does the opening state what the reader will be able to do afterward, does every section open with a full sentence of purpose, does every output get a sentence interpreting it, are the headings grammatically parallel, and is every "why" sentence one you can actually verify.
3. Note each place it deviates, and why.
4. Make the edits.
5. Return to step 1 and reread the whole piece. Repeat until a full read finds nothing left to fix.

One pass is not enough. The first read catches the obvious problems; idiom, tone, and unparallel headings usually surface only on a second or third read. For a page that executes code, one of these passes is the render-and-check pass: run the page and confirm the prose matches the output, including numbers and plots.

## Docstrings

- Google style. First line: one sentence, imperative or declarative, under 80 characters.
- Every public function has a worked `Example:` block that runs under `pytest --doctest-modules`.
- Describe arguments by what they mean in the analysis, not by type (the signature gives the type).

## Commit messages

- Subject under 70 characters, imperative mood.
- Body says what changed and why, not how (the diff shows how).
