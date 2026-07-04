# 1 — Full calibration workflow example

**Goal:** a runnable, documented end-to-end example (script + future site
tutorial) showing the workflow most real HTA models need: some parameters
are calibrated to observed targets, the rest come from the literature, and
the two sources are combined into one PSA that flows through CEA and VoI.

## Why this is first

It exercises every implemented layer at once (`calibrate` → `params` →
`run` → `cea` → `voi` → `report`), it is the strongest adoption story
after bring-your-own-outputs, and it forces one small API gap to be
closed (mixing draw matrices) before the engine work builds on it.

## The workflow to demonstrate

1. **Literature parameters.** A `ParameterSet` with mean/SE-derived
   distributions (utilities, unit costs), optionally correlated.
2. **Calibrated parameters.** A natural-history simulator with unknown
   transition intensities; priors as `heval` specs; observed targets
   (e.g. prevalence at two ages); `abc_calibrate(...)` returns an
   iteration-indexed posterior draw matrix.
3. **Mixing the two sources** into one draw matrix (see API below): the
   calibrated columns keep their joint posterior correlation; the
   literature columns are sampled independently of them; both share one
   iteration index.
4. **Run the decision model** over the mixed draws with `run_psa`.
5. **CEA**: `icer_table`, CEAC/CEAF.
6. **VoI**: `evppi_ranking` over *both* calibrated and literature
   parameters — showing that once draws share the iteration index, VoI is
   indifferent to where a parameter came from.
7. **Report**: plots + `capture_run` provenance recording both the ABC
   settings and the literature specs.

## API gap to close: `heval.params.mix_draws`

```python
def mix_draws(
    *sources: pd.DataFrame,
    n: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Combine draw matrices from different sources into one PSA matrix.

    Rules:
    - Column names must be disjoint across sources.
    - Row correlation *within* each source is preserved (rows are
      resampled jointly, never column-by-column) — this is what keeps a
      calibrated posterior's joint structure intact.
    - Sources are combined row-wise independently of each other.
    - If n is None, n = min(len(s) for s in sources); sources longer than
      n are resampled (with replacement only if shorter than n).
    - Result carries a fresh RangeIndex named "iteration".
    """
```

Notes:

- Resampling a posterior with replacement up to a larger `n` is standard
  practice; document that it does not add information.
- `ParameterSet.sample(...)` output and `CalibrationResult.posterior` are
  both valid inputs by construction; so is any external draw matrix with
  an `iteration` index (bring-your-own-draws, mirroring
  bring-your-own-outputs).
- Provenance: `capture_run` should accept a `draw_sources` mapping (e.g.
  `{"posterior": "ABC-SMC, 4 populations, eps=0.01", "literature": spec}`)
  so the model card shows where every parameter came from.

## Deliverables

- `examples/calibration_workflow.py` (structured like
  `examples/byoo_example.py`, with printed ICER table, EVPPI ranking, and
  saved plots).
- `mix_draws` in `heval.params` with tests: disjoint-column validation,
  joint-row preservation (correlation of posterior columns survives),
  reproducibility under seed, index contract.
- A short section in the README pointing at the example.

## Acceptance

- The example runs end to end under `uv run python`.
- A test asserts that the Spearman correlation between two calibrated
  columns in the mixed matrix matches the posterior's within tolerance.
- EVPPI of a calibrated parameter is recovered on a synthetic case where
  the answer is known by construction (reuse the analytic-Gaussian
  machinery from `tests/test_voi.py`).
