"""Bring parameter values into the run loop from sources other than distributions.

Not every analysis starts from `ParameterSet` distributions. Some run at one
base-case set of point values, some carry a draw matrix exported from another
tool, and some carry a posterior sample with weights. These three entry points
turn each source into a **parameter draw matrix**: a tidy ``pandas.DataFrame``
with one row per iteration (index named ``"iteration"``) and one numeric column
per scalar parameter. Once a table is that, `heormodel.run.run_psa`, `heormodel.cea`,
and `heormodel.voi` do not care where it came from.

This is the parameter-side analogue of `heormodel.run.as_outcomes`, the
bring-your-own-outputs entry point on the outcome side.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


def single_draw(values: Mapping[str, float]) -> pd.DataFrame:
    """Wrap one named set of parameter values as a one-row draw matrix.

    The row's iteration index is 0, so the result flows straight into
    `heormodel.run.run_psa` for a base-case (deterministic) run alongside the
    probabilistic analysis.

    Args:
        values: Parameter names mapped to point values, for example a
            `ParameterSet.means` result or a set of published base-case
            estimates.

    Returns:
        DataFrame with one row, a ``RangeIndex`` named ``"iteration"`` (value
        0), and one column per parameter.

    Example:
        >>> from heormodel.params import single_draw
        >>> single_draw({"p_die": 0.1, "cost": 1000.0}).shape
        (1, 2)
    """
    if not values:
        raise ValueError("single_draw requires at least one parameter value.")
    frame = pd.DataFrame([dict(values)], index=pd.RangeIndex(1, name="iteration"))
    return frame.astype(float)


def read_draws(
    source: pd.DataFrame | str | Path,
    *,
    iteration: str | None = None,
) -> pd.DataFrame:
    """Validate an external parameter sample as a draw matrix.

    Reads a CSV path or takes a DataFrame. If ``iteration`` names a column, it
    becomes the index; otherwise a fresh ``RangeIndex`` named ``"iteration"`` is
    assigned. Every remaining column must be numeric, so the matrix can flow
    through the run loop unchanged.

    Args:
        source: A draw matrix DataFrame, or a path to a CSV file of one.
        iteration: Name of the column holding the iteration index. ``None``
            assigns a fresh ``RangeIndex``.

    Returns:
        DataFrame with an index named ``"iteration"`` and one numeric column per
        parameter.

    Raises:
        ValueError: If the table is empty, ``iteration`` names a missing column,
            or any parameter column is non-numeric.

    Example:
        >>> import pandas as pd
        >>> from heormodel.params import read_draws
        >>> df = pd.DataFrame({"p_die": [0.1, 0.2], "cost": [900.0, 1100.0]})
        >>> read_draws(df).index.name
        'iteration'
    """
    frame = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()
    if len(frame) == 0:
        raise ValueError("read_draws received an empty table.")

    if iteration is not None:
        if iteration not in frame.columns:
            raise ValueError(
                f"iteration column {iteration!r} is not in the table; "
                f"columns are {list(frame.columns)}."
            )
        index = pd.Index(frame[iteration], name="iteration")
        frame = frame.drop(columns=[iteration])
    else:
        index = pd.RangeIndex(len(frame), name="iteration")

    non_numeric = [col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])]
    if non_numeric:
        raise ValueError(
            f"draw matrix columns must be numeric; {non_numeric} are not. "
            "Drop or encode these columns, or name the iteration column via "
            "the iteration argument."
        )
    if frame.shape[1] == 0:
        raise ValueError("read_draws found no parameter columns.")

    frame = frame.astype(float)
    frame.index = index
    return frame


def resample_posterior(
    source: pd.DataFrame | str | Path,
    *,
    n: int,
    weight: str = "weight",
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Resample a weighted parameter table into an unweighted draw matrix.

    Rows are drawn with replacement with probability proportional to the
    ``weight`` column, jointly (whole rows), so any correlation in the posterior
    survives. The ``weight`` column is dropped from the result, which carries a
    fresh ``RangeIndex`` named ``"iteration"``.

    Resampling to an ``n`` larger than the input adds no information; it only
    smooths Monte Carlo noise in downstream expectations.

    Args:
        source: A weighted parameter table DataFrame, or a path to a CSV file of
            one, with one column of weights and the rest parameters.
        n: Number of rows in the resampled draw matrix.
        weight: Name of the weight column. Weights must be non-negative and not
            all zero; they are normalised internally.
        seed: Integer seed or ``numpy`` Generator for the resampling.

    Returns:
        DataFrame with ``n`` rows, a ``RangeIndex`` named ``"iteration"``, and
        one column per parameter (the weight column removed).

    Raises:
        ValueError: If the table is empty, ``weight`` names a missing column,
            ``n`` is not positive, or the weights are negative or sum to zero.

    Example:
        >>> import pandas as pd
        >>> from heormodel.params import resample_posterior
        >>> post = pd.DataFrame({"beta": [0.1, 0.2, 0.3], "weight": [1.0, 2.0, 1.0]})
        >>> resample_posterior(post, n=1000, seed=0).shape
        (1000, 1)
    """
    frame = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()
    if len(frame) == 0:
        raise ValueError("resample_posterior received an empty table.")
    if weight not in frame.columns:
        raise ValueError(
            f"weight column {weight!r} is not in the table; columns are {list(frame.columns)}."
        )
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    w = frame[weight].to_numpy(dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")

    params = read_draws(frame.drop(columns=[weight]))
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    picks = rng.choice(len(params), size=n, replace=True, p=w / total)
    out = params.iloc[picks].reset_index(drop=True)
    out.index = pd.RangeIndex(n, name="iteration")
    return out
