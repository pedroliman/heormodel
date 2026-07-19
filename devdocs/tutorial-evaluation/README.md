# Tutorial evaluation report

A critical evaluation of the `heormodel` website tutorials, read from the perspective of the target reader: a health economics and outcomes research (HEOR) modeler at the graduate level or beyond, who builds cost-effectiveness models and writes code but is not a software engineer. The evaluation covers each tutorial on its own and the sequence the tutorials form together.

Each tutorial has its own evaluation file (`eval-<name>.md`) in this folder, written against the repository writing style guide (`devdocs/guidance/writing_style.md`) and the tutorial-quality criteria it sets. This page synthesizes those evaluations and adds the cross-cutting findings that only appear when the tutorials are read as a set: where a concept is introduced relative to where it is first used, whether the pages link together in a coherent path, and where they repeat or contradict each other.

## How the evaluation was run

Each tutorial was reviewed by a separate reviewer that read the writing style guide, the tutorials index, and the tutorial itself, checked the rendered output where it could, and rated the page on purpose, pedagogy, writing, correctness, and sequencing. The cross-cutting analysis below comes from one reading of all twenty pages in order, including the site homepage and the replication gallery.

## Verdicts at a glance

Ratings are Excellent, Good, Needs work, or Weak. This table is filled in as the per-tutorial evaluations land.

| # | Tutorial | Section | Rating |
|---|---|---|---|
| 1 | Full pipeline | Getting started | Needs work |
| 2 | Bring your own outputs | Getting started | Good |
| 3 | Parameter inputs from data | Getting started | Good |
| 4 | Cohort state-transition model | Model engines | pending |
| 5 | Microsimulation engine | Model engines | pending |
| 6 | Markov vs microsimulation models | Model engines | pending |
| 7 | Discrete-event simulation engine | Model engines | pending |
| 8 | Compartmental transmission model | Model engines | pending |
| 9 | Deterministic sensitivity analysis | Sensitivity and VoI | pending |
| 10 | Value of information | Sensitivity and VoI | pending |
| 11 | Calibration workflow | Calibration | pending |
| 12 | Calibrating with ABC | Calibration | pending |
| 13 | Calibrating with simulation-based inference | Calibration | pending |
| 14 | Surrogate-accelerated calibration | Calibration | pending |
| 15 | Calibrating a stochastic microsimulation | Calibration | pending |
| 16 | Time-dependent cohort model | Replications | pending |
| 17 | Microsimulation replication | Replications | pending |
| 18 | Discrete-event simulation replication | Replications | pending |

## The set is strong at the page level and weak at the seams

Read one at a time, most of these tutorials are good. The prose follows the methods-appendix voice the style guide asks for, the mathematics of each estimand is stated before it is used, and results are confirmed against published numbers where a published number exists. The calibration sequence is the high point: four tutorials calibrate one shared model against one shared target with escalating methods, and each ends by naming the limitation the next one removes.

The weaknesses are almost all at the seams between tutorials, not inside them. The forward links contradict the reading order, value-of-information results are interpreted six tutorials before value of information is taught, the getting-started section opens with data plumbing rather than a model, and the replication gallery promises an analysis that two of its four pages do not run. These are the findings that a page-by-page review cannot surface, and they are where the most useful editing is.

## Cross-cutting finding 1: the forward links contradict the reading order

Every tutorial ends with a "Next:" link, and the sidebar lists the tutorials in a fixed order, but the two disagree. A reader who navigates by the "Next:" links does not walk the sidebar order; the links appear to be left over from an earlier arrangement in which the full-pipeline tutorial came after the engines.

The sidebar order is: full pipeline, bring your own outputs, parameter inputs, then the five engine tutorials, then deterministic sensitivity analysis and value of information, then the five calibration tutorials, then the replications. The "Next:" links instead run:

- Full pipeline points to value of information, skipping eight tutorials.
- Bring your own outputs points to the cohort model, skipping parameter inputs.
- Parameter inputs points to deterministic sensitivity analysis, skipping every engine.
- The cohort model points to value of information.
- Discrete-event simulation points backward to the full pipeline, a getting-started page.
- The compartmental model, the last engine, has no "Next:" link at all, and nothing links to it.
- Deterministic sensitivity analysis points to the replication gallery, skipping value of information, the item directly below it.
- Value of information has no "Next:" link and ends on a plot.
- Calibration workflow points backward to parameter inputs, a getting-started page.

Only the microsimulation-to-cohort-comparison-to-discrete-event stretch and the calibration arc link forward correctly. A reader following the links loops between the discrete-event, full-pipeline, and value-of-information pages and never reaches the compartmental model or the sensitivity pages in a natural order. This is the single highest-value, lowest-effort fix in the set: make every "Next:" link point to the next tutorial in the sidebar, give the compartmental model and value-of-information pages a forward link, and remove the backward links.

## Cross-cutting finding 2: value of information is used before it is taught

The expected value of perfect information (EVPI) and the per-parameter ranking by expected value of partial perfect information (EVPPI) appear, with interpreted numbers, in the full-pipeline tutorial (page 1), bring your own outputs (page 2), the cohort model (page 4), the microsimulation (page 5), discrete-event simulation (page 7), and the compartmental model (page 8). The net monetary benefit and EVPI equations are first written down in bring your own outputs, page 2. The dedicated value-of-information tutorial, which defines EVPI, EVPPI, expected value of sample information (EVSI), and expected net benefit of sampling, is page 10.

The reader therefore meets six worked EVPI figures, each interpreted as if the quantity were already understood, before the page that defines it. The early tutorials treat "the expected value of perfect information is positive, so resolving the uncertainty is worth this much" as self-explanatory. For a reader who has not yet done a value-of-information analysis, it is not. Two fixes are compatible: add one or two sentences defining EVPI at its first appearance, or have the early tutorials state plainly that they are previewing a measure the value-of-information tutorial defines, and link to it, rather than interpreting the number in place.

## Cross-cutting finding 3: the getting-started section opens with plumbing, not a model

The first three tutorials are the full pipeline, bring your own outputs, and parameter inputs from data. The full-pipeline tutorial builds a two-branch decision tree through `Outcomes.from_wide`, not one of the package's four model engines. Bring your own outputs ingests a results table computed elsewhere. Parameter inputs ingests draw matrices and a posterior sample. The reader does not build a model with one of the engines until the cohort tutorial, page 4.

Two of the first three tutorials are about getting data into and out of the analysis without building a model, and the first teaches on a synthetic decision tree. The style guide states that a tutorial should teach against a real model engine rather than a synthetic construct. A reader who came to the package to build cost-effectiveness models meets three pages of input and output plumbing before the first engine. This ordering was a deliberate choice recorded in the roadmap, so it is worth revisiting rather than simply an oversight, but from the target reader's point of view the lede is buried: the natural first tutorial is "build a cohort model and analyze it," with the bring-your-own-outputs and parameter-input pages positioned as the two ways to enter the workflow from the side once the reader has seen it run.

## Cross-cutting finding 4: the homepage quickstart overlaps the full-pipeline tutorial

The site homepage runs a complete three-state Markov cohort analysis: parameter sampling, `icer_table`, a cost-effectiveness plane, a cost-effectiveness acceptability curve, and an EVPI curve swept over the willingness-to-pay threshold. The full-pipeline tutorial, the first tutorial, runs a smaller analysis on a decision tree and interprets its results less fully. The homepage quickstart is a more complete pipeline than the page named "full pipeline." The two should be differentiated: either the full-pipeline tutorial should carry the analysis the homepage only previews, on a real engine, or it should be repositioned so it is not the reader's second encounter with the same material.

## Cross-cutting finding 5: the replication gallery over-promises

The replication gallery states that each of its four replications "first matches the source's deterministic results, then runs the same model through `heormodel.cea` and `heormodel.voi` to show how a probabilistic analysis extends it." That is true for the discrete-event replication and for the cohort replication, which is cross-listed under the engine tutorials. It is not true for the time-dependent cohort replication or the microsimulation replication: both stop at the deterministic base-case `icer_table` and run no probabilistic sensitivity analysis, no acceptability curve, and no value-of-information analysis. The gallery's own summary claim holds for two of its four entries. Either extend the time-dependent and microsimulation replications with the short probabilistic analysis the gallery promises, or soften the gallery text to describe what those two pages actually do.

## Cross-cutting finding 6: the first engine tutorial is the heaviest

The cohort tutorial is the reader's first real model engine, and it is also a sixteen-parameter published replication with rate-to-probability conversion, hazard ratios applied to rates, a Simpson within-cycle correction, and densely packed transition-matrix assignments. The microsimulation tutorial that follows uses simpler, plainly illustrative invented parameters. The entry engine is the most demanding page in the engine section, not the least. The published replication is valuable, but a reader meeting `MarkovModel` for the first time has to absorb the engine and a faithful reproduction of a real article at once. Consider whether the first engine tutorial should teach the engine on a small illustrative model and leave the full published replication to the gallery, where the cohort replication is already cross-listed.

## Cross-cutting finding 7: the calibration workflow sits ahead of the method it uses

The calibration section opens with the calibration-workflow tutorial, whose subject is mixing calibrated and literature parameters with `mix_draws` and recording their provenance. To do that it calls `abc_calibrate` and shows a posterior, but approximate Bayesian computation is not explained until the next tutorial. The workflow tutorial also uses a different disease model, a continuous-time generator matrix, from the shared three-state model that the following four calibration tutorials all use. The ABC tutorial then introduces itself as "the first of four calibration tutorials that share one disease model," which reads oddly when it is the second page in the section. The workflow tutorial is really an application of calibration, not an introduction to it. Placing it after the ABC tutorial, so the reader knows what a posterior and a calibration run are before being shown how to mix one into an analysis, would remove the forward dependency.

## Cross-cutting finding 8: repetition and code density

The four-state Sick-Sicker cohort model is defined almost verbatim in three tutorials: the cohort model, deterministic sensitivity analysis, and value of information. The repetition is partly justified, since each page must run on its own and the style guide endorses repeating the shared workflow, but the full twenty-five-line model function copied three times is more than the shared-workflow point requires. The dense transition-matrix assignments, which pack several matrix entries onto one line, are within the style guide's allowance for replicated-paper notation but are hard for a reader who is learning to build the model rather than read it. The discrete-event tutorial carries the steepest coding prerequisite in the set, since it requires fluency with the underlying discrete-event library; that page needs the most care to stay readable for a modeler who is not a software engineer.

## Provisional recommendation on sequence

The individual pages need light editing; the sequence needs a deliberate second pass. In priority order:

1. Repair the forward navigation so the "Next:" links match the sidebar order, and give every page a forward link. This is small and removes the most visible defect.
2. Resolve where value of information is introduced, so the reader is not asked to interpret an EVPI before it is defined.
3. Reconsider the opening: lead the getting-started section with a real engine, and position the bring-your-own-outputs and parameter-input pages as side entrances to the workflow rather than the first thing the reader sees.
4. Align the replication gallery's promise with what its pages do.
5. Move the calibration workflow after the ABC tutorial.

The per-tutorial files carry the page-level detail behind these findings.

## Specific edits for the cross-cutting findings

These are copy-ready changes for the findings above. Each per-tutorial file carries the same level of specificity for its own page.

Finding 1, forward links. Set each tutorial's closing "Next:" link to the next page in the sidebar order, and add the two missing links:

- `full-pipeline.qmd`: change the closing link so "Next:" points to `byo-outputs.qmd`, not `voi.qmd`.
- `byo-outputs.qmd`: point "Next:" to `parameter-inputs.qmd`, not `mdm-cohort.qmd`.
- `parameter-inputs.qmd`: point "Next:" to `mdm-cohort.qmd`, not `dsa.qmd`.
- `mdm-cohort.qmd`: point "Next:" to `microsim.qmd`, not `voi.qmd`.
- `des.qmd`: point "Next:" to `seir-vaccination.qmd`, not `full-pipeline.qmd`.
- `seir-vaccination.qmd`: add a closing line, "Next: [deterministic sensitivity analysis](dsa.qmd) asks which parameters move the result, before [value of information](voi.qmd) asks what resolving the remaining uncertainty is worth."
- `dsa.qmd`: point "Next:" to `voi.qmd`, not `replication-gallery.qmd`.
- `voi.qmd`: add a closing line, "Next: the [calibration workflow](calibration-workflow.qmd) fits model parameters to data and carries the fitted uncertainty into the same analysis."
- `calibration-workflow.qmd`: point "Next:" to `calibrate-abc.qmd`, not `parameter-inputs.qmd` (see finding 7).

Finding 2, value of information used before it is taught. At the first EVPI figure in `full-pipeline.qmd` (the "Analyzing cost-effectiveness and value of information" section), add after the sentence that prints the EVPI: "The expected value of perfect information is the average gain from resolving all parameter uncertainty before deciding, an upper bound on what any study could be worth; the [value of information](voi.qmd) tutorial defines it and the related measures in full." Add the same one-clause pointer at the first EVPI in `byo-outputs.qmd`, `microsim.qmd`, `des.qmd`, and `seir-vaccination.qmd`, so each interprets its number against a named, linked definition rather than assuming it.

Finding 4, replication gallery. In `replication-gallery.qmd`, the intro sentence "Each replication first matches the source's deterministic results, then runs the same model through `heormodel.cea` and `heormodel.voi` to show how a probabilistic analysis extends it" is true only for the cohort and discrete-event pages. Replace it with: "Each replication first matches the source's deterministic results. The cohort and discrete-event pages then run the same model through `heormodel.cea` and `heormodel.voi` to show how a probabilistic analysis extends it; the time-dependent and microsimulation pages stop at the deterministic match." The stronger fix is to add a short probabilistic sensitivity analysis to `mdm-cohort-timedep.qmd` and `mdm-microsim.qmd` so the original sentence holds; the per-tutorial files for those two give the exact code to add.

Finding 7, calibration workflow placement. Move `calibration-workflow.qmd` to follow `calibrate-abc.qmd` in both `docs/_quarto.yml` (the sidebar `contents` list) and `docs/tutorials/index.qmd` (the `calibration` listing). Then `calibrate-abc.qmd`'s self-description, "the first of four calibration tutorials that share one disease model," becomes true, and the workflow page no longer calls `abc_calibrate` before approximate Bayesian computation is defined.
</content>
