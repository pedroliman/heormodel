# 12. Value-of-information tutorial against a published analysis

`heval.voi` ships EVPI, EVPPI, and EVSI, but the site has no tutorial that walks the value-of-information workflow end to end and checks it against an external answer. This item adds one, reproducing a published VoI analysis so a reader sees the numbers land where the source says they should. It ships as `examples/voi_tutorial.py` and a website tutorial, and it reinforces the EVPPI validation already in `tests/test_voi.py` with a second, published reference point.

## What the tutorial reproduces

Pick a published VoI worked example that reports EVPI and per-parameter EVPPI (and, if available, EVSI for a proposed study) with enough detail to rebuild the decision model and priors: the model structure, the parameter distributions, and the willingness-to-pay threshold. Two kinds of source work, and the tutorial uses one of each where possible:

1. An analytic benchmark. A linear or Gaussian decision model whose EVPI and EVPPI have closed forms, so the metamodel estimates can be checked to tight tolerance. This is the correctness anchor and mirrors the synthetic case already tested.
2. An applied published case study. A cost-effectiveness model with reported VoI results, so the tutorial shows the workflow on a realistic model and lands near the published EVPI and EVPPI ranking.

The implementing PR fixes the exact sources and records their citations in the tutorial. Do not name external software packages; cite the analysis, the model, and the reported numbers.

## The workflow the tutorial shows

1. Parameters: a `ParameterSet` matching the source's distributions, or `read_draws` if the source publishes its sample (item 7).
2. Run: `run_psa` over the draws, in parallel by default (item 9).
3. Decision: `icer_table`, and the net-benefit framing VoI needs.
4. EVPI: `evpi` at the source's threshold, compared to the reported value.
5. EVPPI: `evppi_ranking` over the parameters, compared to the source's ranking and magnitudes, with a note on which metamodel (spline or GP) suits which parameter count.
6. EVSI: `evsi_regression` for a proposed study design where the source reports one, framed as the expected value of the study against its cost.

The tutorial closes on interpretation: EVPI as the ceiling on research value, EVPPI as which parameters that value attaches to, EVSI as whether a specific study clears its cost.

## Deliverables

- `examples/voi_tutorial.py` reproducing the chosen analysis, printing the EVPI, the EVPPI ranking, and any EVSI, next to the published values.
- A website tutorial narrating it, placed after the full-pipeline tutorial, following `guidance/writing_style.md`.

## Acceptance

- A test asserts the analytic benchmark's EVPI and EVPPI match the closed form within tolerance.
- The tutorial's reported numbers land within a stated tolerance of the published applied values, and the tutorial states the tolerance and any difference in assumptions.
- The example and rendered tutorial run under `uv run python` and `uv run quartodoc build`, with prose matching the printed numbers and plots.
