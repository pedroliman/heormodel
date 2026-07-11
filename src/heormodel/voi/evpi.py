"""Expected value of perfect information from the probabilistic analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from heormodel.cea.ceac import expected_loss
from heormodel.models.outcomes import Outcomes


def evpi(
    outcomes: Outcomes,
    wtp: float | ArrayLike | Sequence[float],
    *,
    effect: str | None = None,
) -> float | pd.Series:
    """Expected value of perfect information per decision.

    EVPI is the expected NMB gain from resolving **all** uncertainty:
    ``E[max_d NMB_d] - max_d E[NMB_d]``. It equals the smallest expected loss
    across interventions at each willingness-to-pay value, so it reads straight
    off the expected loss curves.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        wtp: A willingness-to-pay value or grid.
        effect: Effect column (default: the primary effect).

    Returns:
        A float for scalar ``wtp``; a Series indexed by ``wtp`` for a grid.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.voi import evpi
        >>> c = pd.DataFrame({"A": [0.0, 0.0], "B": [5.0, 5.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, -1.0]})
        >>> evpi(Outcomes.from_wide(c, e), wtp=10.0)
        2.5
    """
    grid = np.atleast_1d(np.asarray(wtp, dtype=np.float64))
    values = expected_loss(outcomes, grid, effect=effect).min(axis=1).to_numpy(dtype=np.float64)
    if np.isscalar(wtp) or (hasattr(wtp, "ndim") and getattr(wtp, "ndim", 1) == 0):
        return float(values[0])
    return pd.Series(values, index=pd.Index(grid, name="wtp"), name="evpi")
