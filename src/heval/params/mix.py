"""Combine parameter draw matrices from different sources into one PSA matrix.

Applied models mix parameter sources: some columns are calibrated to
observed targets (a posterior draw matrix), the rest come from the
literature (a `ParameterSet.sample` matrix). `mix_draws` joins
them into a single matrix that carries the standard ``iteration`` index, so
the mixed draws flow through `heval.run.run_psa`, `heval.cea`, and
`heval.voi` exactly like any other draw matrix.

The join resamples whole rows within each source, never column by column,
so a posterior's joint correlation survives. Sources are resampled
independently of each other, so no correlation is invented across them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mix_draws(
    *sources: pd.DataFrame,
    n: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Combine draw matrices from different sources into one PSA matrix.

    Each source is resampled to ``n`` rows by drawing whole rows, so the
    joint structure within a source (for example, a calibrated posterior's
    correlation) is preserved while sources stay independent of each other.

    Args:
        sources: Two or more draw matrices with disjoint column names. Valid
            inputs are `ParameterSet.sample` output,
            `heval.calibrate.CalibrationResult.posterior`, or any
            external draw matrix indexed by iteration.
        n: Rows in the mixed matrix. Defaults to the shortest source's
            length. Sources shorter than ``n`` are resampled with
            replacement; sources at least as long are subsampled without
            replacement.
        seed: Integer seed or ``numpy`` Generator for the resampling.

    Returns:
        DataFrame with a fresh ``RangeIndex`` named ``"iteration"`` and every
        column from every source.

    Raises:
        ValueError: If no sources are given, a source is empty, ``n`` is not
            positive, or column names collide across sources.

    Example:
        >>> import pandas as pd
        >>> from heval.params import mix_draws
        >>> lit = pd.DataFrame({"u": [0.6, 0.7, 0.8]})
        >>> post = pd.DataFrame({"beta": [0.1, 0.2]})
        >>> mixed = mix_draws(lit, post, n=4, seed=0)
        >>> list(mixed.columns)
        ['u', 'beta']
        >>> mixed.index.name
        'iteration'
        >>> len(mixed)
        4
    """
    if not sources:
        raise ValueError("mix_draws requires at least one source.")
    seen: dict[str, int] = {}
    for i, source in enumerate(sources):
        if len(source) == 0:
            raise ValueError(f"Source {i} is empty.")
        for col in source.columns:
            if col in seen:
                raise ValueError(
                    f"Column {col!r} appears in sources {seen[col]} and {i}; "
                    "source columns must be disjoint."
                )
            seen[col] = i
    if n is None:
        n = min(len(source) for source in sources)
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    blocks: list[pd.DataFrame] = []
    for source in sources:
        m = len(source)
        picks = rng.choice(m, size=n, replace=m < n)
        blocks.append(source.iloc[picks].reset_index(drop=True))
    mixed = pd.concat(blocks, axis=1)
    mixed.index = pd.RangeIndex(n, name="iteration")
    return mixed
