# 7. Parameter inputs: single sets, CSV draws, and weighted posteriors

Not every analysis starts from `ParameterSet` distributions. Some run at one base-case set of values, some carry a draw matrix exported from another tool, and some carry a posterior sample with weights. All three should flow into the same run loop with no glue code, because the draw matrix is already the package's shared currency. This item adds three entry points to `heval.params`, mirroring how `as_outcomes` is the bring-your-own-outputs entry point on the outcome side.

The rule that governs all three: the result is a valid draw matrix, a tidy `DataFrame` with one row per iteration, index named `iteration`, one column per scalar parameter. Once it is that, `run_psa`, `cea`, and `voi` do not care where it came from.

## Single parameter set

A base-case run evaluates the model at one set of point values, the deterministic analysis that sits next to the PSA. Add `single_draw`:

```python
def single_draw(values: Mapping[str, float]) -> pd.DataFrame:
    """Wrap one named set of parameter values as a one-row draw matrix.

    The row's iteration index is 0, so the result flows straight into
    ``run_psa`` for a base-case (deterministic) run.

    Example:
        >>> from heval.params import single_draw
        >>> single_draw({"p_die": 0.1, "cost": 1000.0}).shape
        (1, 2)
    """
```

`ParameterSet` gains `.at_means()` returning `single_draw(self.means().to_dict())`, so the base case of a probabilistic model is one call.

## Draws from a CSV or DataFrame

A user with a draw matrix from another simulator or a spreadsheet export should not rebuild it. Add `read_draws`:

```python
def read_draws(
    source: pd.DataFrame | str | Path,
    *,
    iteration: str | None = None,
) -> pd.DataFrame:
    """Validate an external parameter sample as a draw matrix.

    Reads a CSV path or takes a DataFrame. If ``iteration`` names a
    column, it becomes the index; otherwise a fresh RangeIndex named
    ``iteration`` is assigned. Non-numeric columns raise.
    """
```

This is the parameter-side analogue of `as_outcomes`: one function that turns any tidy table into the schema the run loop expects.

## Weighted posterior, resampled with replacement

A Bayesian calibration or a bootstrap may hand back a table of parameter rows with a weight column, an importance sample or a posterior on a grid. The standard move is to resample rows with replacement in proportion to the weights, which turns the weighted sample into an equally weighted draw matrix. Add `resample_posterior`:

```python
def resample_posterior(
    source: pd.DataFrame | str | Path,
    *,
    n: int,
    weight: str = "weight",
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Resample a weighted parameter table into an unweighted draw matrix.

    Rows are drawn with replacement with probability proportional to the
    ``weight`` column, jointly (whole rows), so any correlation in the
    posterior survives. The ``weight`` column is dropped from the result,
    which carries a fresh RangeIndex named ``iteration``.
    """
```

Resampling whole rows, never column by column, preserves the joint posterior, the same commitment `mix_draws` makes. Document that resampling to an `n` larger than the input adds no information, only smooths Monte Carlo noise in downstream expectations.

## Deliverables

- `single_draw`, `read_draws`, `resample_posterior`, and `ParameterSet.at_means` in `heval.params`, added to `__all__` and the quartodoc reference.
- `examples/parameter_inputs.py` showing all three: a base-case run, a CSV draw matrix, and a weighted posterior, each fed to `run_psa`.
- A website tutorial narrating the example.

## Acceptance

- Each entry point produces a matrix that `run_psa` accepts without modification.
- A test asserts `resample_posterior` recovers the weighted mean of each column within Monte Carlo tolerance, and that the Spearman correlation between two columns survives resampling.
- A test asserts `read_draws` rejects a non-numeric column with an actionable message and honours an explicit `iteration` column.
- The example and tutorial run under `uv run python`.
