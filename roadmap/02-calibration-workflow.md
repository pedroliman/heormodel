# 2. Full calibration workflow example

A runnable example (script now, website tutorial with item 1) showing the workflow most applied models need: some parameters are calibrated to observed targets, the rest come from the literature, and both sources feed one PSA that flows through CEA and VoI.

This item exercises every implemented layer at once and closes the one API gap it needs: mixing draw matrices.

## The workflow to demonstrate

1. Literature parameters: a `ParameterSet` with mean/SE-derived distributions (utilities, unit costs), optionally correlated.
2. Calibrated parameters: a natural-history simulator with unknown transition intensities, priors as `heval` specs, observed targets such as prevalence at two ages. `abc_calibrate` returns an iteration-indexed posterior draw matrix.
3. Mix the two sources into one draw matrix (API below). Calibrated columns keep their joint posterior correlation; literature columns are independent of them; both share one iteration index.
4. Run the decision model over the mixed draws with `run_psa`.
5. CEA: `icer_table`, CEAC, CEAF.
6. VoI: `evppi_ranking` over calibrated and literature parameters together. Once draws share the iteration index, VoI does not care where a parameter came from.
7. Report: plots plus `capture_run` provenance recording the ABC settings and the literature specs.

## API gap: `heval.params.mix_draws`

```python
def mix_draws(
    *sources: pd.DataFrame,
    n: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Combine draw matrices from different sources into one PSA matrix.

    Rules:
    - Column names must be disjoint across sources.
    - Rows within a source are resampled jointly, never column by
      column, which preserves a posterior's joint correlation.
    - Sources are combined independently of each other.
    - If n is None, n = min(len(s) for s in sources). Shorter sources
      are resampled with replacement.
    - The result carries a fresh RangeIndex named "iteration".
    """
```

Notes:

- Resampling a posterior with replacement to a larger `n` is standard practice; document that it adds no information.
- Valid inputs by construction: `ParameterSet.sample` output, `CalibrationResult.posterior`, and any external draw matrix with an `iteration` index (bring-your-own-draws, mirroring bring-your-own-outputs).
- Provenance: `capture_run` gains a `draw_sources` mapping so the model card shows where every parameter came from.

## Deliverables

- `examples/calibration_workflow.py`, structured like `examples/byoo_example.py`, printing the ICER table and EVPPI ranking and saving plots.
- `mix_draws` in `heval.params` with tests: disjoint-column validation, joint-row preservation, reproducibility under seed, index contract.
- A README section pointing at the example.

## Acceptance

- The example runs end to end under `uv run python`.
- A test asserts that the Spearman correlation between two calibrated columns survives mixing within tolerance.
- EVPPI of a calibrated parameter is recovered on a synthetic case with a known answer, reusing the analytic machinery in `tests/test_voi.py`.
