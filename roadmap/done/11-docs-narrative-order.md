# 11. Documentation narrative order

The tutorials teach in the wrong order. A reader meets the microsimulation engine before any state-transition model, and the cohort Markov model, the workhorse of applied cost-effectiveness, sits in a "Replications" side menu. Reorder the site so the narrative climbs from the simplest entry point to the most detailed engine. No new content; this item moves and relinks existing pages.

## Target order

The core teaching sequence is: bring your own outputs, then a Markov cohort model, then the microsimulation. Reasoning: bring-your-own-outputs needs no engine and shows the analysis layer first; the Markov cohort model is the standard structural model and the mental anchor; the microsimulation is the step up in detail, and it lands best right after the cohort model it generalizes (which sets up item 10, "Markov vs microsimulation models").

Concretely, the Tutorials menu becomes:

1. Bring your own outputs
2. Markov cohort model (promoted from Replications)
3. Markov vs microsimulation models (item 10, once it lands)
4. Microsimulation engine
5. Discrete-event simulation
6. Full pipeline
7. Calibration workflow

The time-dependent cohort and microsimulation replications stay under Replications as validation exhibits. The replication gallery keeps its overview role and links into the promoted tutorials rather than duplicating them.

## Changes

- Reorder the `website.navbar` Tutorials menu in `docs/_quarto.yml` to the sequence above, and move the Markov cohort tutorial out of the Replications menu into Tutorials.
- Update `page-navigation` next/previous flow implicitly (it follows the navbar order) and fix any in-page "where to go next" links so each tutorial points forward to the next in the new sequence.
- Update `get-started.qmd` so its closing links send a new reader to the Markov cohort tutorial before the microsimulation.
- Update the README narrative so it introduces a cohort state-transition model before the microsimulation, matching the site.

## Acceptance

- `uv run quartodoc build` and `quarto render docs` succeed with no broken cross-links (the build already fails on broken links; keep it clean).
- The Tutorials menu reads in the target order, and every tutorial's forward link matches it.
- The README and the site present the same order.
