"""Expected value of sample information.

Implemented now: the nonparametric-regression estimator (Strong, Oakley,
Brennan & Breeze, 2015, Medical Decision Making 35:570-583): simulate a
summary statistic of the proposed study once per parameter draw (from that
iteration's parameter draw), regress net benefit on the summary, and read
EVSI off the fitted conditional means.

Scaffolded for later phases: moment matching and importance sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from heormodel._util import as_rng, require_shared_index
from heormodel.cea.nb import nmb
from heormodel.models.outcomes import Outcomes
from heormodel.voi._metamodel import fitted_conditional_means, voi_from_fitted

#: A study simulator: given one parameter draw (a row of the draw matrix)
#: and a random generator, return the summary statistic(s) the proposed
#: study would report, as a mapping of name -> value.
StudySimulator = Callable[[pd.Series, np.random.Generator], dict[str, float]]


def simulate_summaries(
    draws: pd.DataFrame,
    study: StudySimulator,
    *,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate one study dataset summary per parameter draw.

    Args:
        draws: Parameter draw matrix (rows = iterations).
        study: Callable simulating the proposed study from one draw.
        seed: Seed or generator for the study-data randomness.

    Returns:
        DataFrame of summaries aligned on the draws' iteration index.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heormodel.voi import simulate_summaries
        >>> draws = pd.DataFrame({"p": [0.1, 0.2]},
        ...                      index=pd.RangeIndex(2, name="iteration"))
        >>> def study(row, rng):
        ...     return {"x": float(rng.binomial(100, row["p"]))}
        >>> simulate_summaries(draws, study, seed=1).shape
        (2, 1)
    """
    rng = as_rng(seed)
    records = [study(row, rng) for _, row in draws.iterrows()]
    return pd.DataFrame(records, index=draws.index)


def evsi_regression(
    outcomes: Outcomes,
    summaries: pd.DataFrame,
    wtp: float,
    *,
    effect: str | None = None,
    method: str = "spline",
    n_knots: int = 5,
    degree: int = 3,
    seed: int | None = None,
) -> float:
    """EVSI by nonparametric regression on simulated study summaries.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        summaries: Simulated study summaries, one row per iteration,
            aligned on the outcomes' iteration index (see
            `simulate_summaries`).
        wtp: Willingness to pay per unit of effect.
        effect: Effect column (default: the primary effect).
        method: Metamodel; see `heormodel.voi.evppi.evppi`.
        n_knots: Spline knot count.
        degree: Spline degree.
        seed: Subsample seed for the GP method.

    Returns:
        The EVSI estimate, clipped at zero.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.voi import evsi_regression
        >>> rng = np.random.default_rng(0)
        >>> n = 4000
        >>> e_b = rng.normal(0.0, 1.0, n)
        >>> out = Outcomes.from_wide(
        ...     pd.DataFrame({"A": np.zeros(n), "B": np.zeros(n)}),
        ...     pd.DataFrame({"A": np.zeros(n), "B": e_b}))
        >>> s = pd.DataFrame({"xbar": e_b + rng.normal(0, 0.1, n)})
        >>> 0.0 <= evsi_regression(out, s, wtp=1.0) <= 0.45
        True
    """
    require_shared_index(
        summaries.index,
        outcomes.iterations,
        "summaries",
        detail="each summary must be simulated from the parameter draw of the same iteration.",
    )
    nb = nmb(outcomes, wtp, effect=effect)
    fitted = fitted_conditional_means(
        summaries, nb, method=method, n_knots=n_knots, degree=degree, seed=seed
    )
    return max(0.0, voi_from_fitted(fitted))


def evsi_moment_matching(*args: Any, **kwargs: Any) -> float:
    """EVSI by moment matching (stub, scheduled for a later phase).

    Raises:
        NotImplementedError: Always, in this phase.
    """
    raise NotImplementedError(
        "Moment-matching EVSI is scheduled for a later phase; use evsi_regression."
    )


def evsi_importance_sampling(*args: Any, **kwargs: Any) -> float:
    """EVSI by importance sampling (stub, scheduled for a later phase).

    Raises:
        NotImplementedError: Always, in this phase.
    """
    raise NotImplementedError(
        "Importance-sampling EVSI is scheduled for a later phase; use evsi_regression."
    )
