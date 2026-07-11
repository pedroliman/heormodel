"""Convergence diagnostics for probabilistic sensitivity analysis runs.

Phase-1 skeleton: running-mean traces to judge whether the number of
iterations is sufficient. Richer diagnostics (stability of ICERs and
CEAC curves across bootstrap resamples) come with the engine phases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heormodel.models.outcomes import Outcomes


def running_means(outcomes: Outcomes, column: str | None = None) -> pd.DataFrame:
    """Running mean of an outcome column per intervention, by iteration count.

    Flat traces at the right edge indicate the analysis has stabilised for that
    outcome; drifting traces call for more iterations.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        column: Outcome column (default: cost).

    Returns:
        DataFrame indexed by the iteration position (1..n) with one column
        per intervention, holding the mean over the first ``k`` iterations.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.run import running_means
        >>> c = pd.DataFrame({"A": [1.0, 3.0]})
        >>> e = pd.DataFrame({"A": [0.1, 0.1]})
        >>> running_means(Outcomes.from_wide(c, e))["A"].tolist()
        [1.0, 2.0]
    """
    wide = (
        outcomes.costs_wide()
        if column in (None, "cost")
        else (outcomes.data[column].unstack("intervention")[outcomes.interventions])
    )
    values = wide.to_numpy(dtype=np.float64)
    k = np.arange(1, len(values) + 1, dtype=np.float64)
    running = np.cumsum(values, axis=0) / k[:, None]
    return pd.DataFrame(
        running, columns=wide.columns, index=pd.RangeIndex(1, len(values) + 1, name="n")
    )
