"""Expected value of partial perfect information via metamodeling.

Uses the nonparametric-regression estimator: regress each intervention's net
benefit on the parameter subset of interest with a flexible metamodel; the
fitted values estimate the conditional expected NB given those parameters,
and EVPPI is ``E[max_d g_d(x)] - max_d E[g_d(x)]`` (Strong, Oakley &
Brennan, 2014, Medical Decision Making 34:311-326).

This is where the shared iteration index earns its keep: the regression
pairs each outcome row with the parameter draw that produced it.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from heormodel._util import require_shared_index
from heormodel.cea.nb import nmb
from heormodel.models.outcomes import Outcomes
from heormodel.voi._metamodel import fitted_conditional_means, voi_from_fitted


def evppi(
    outcomes: Outcomes,
    draws: pd.DataFrame,
    params: str | Sequence[str],
    wtp: float,
    *,
    effect: str | None = None,
    method: str = "spline",
    n_knots: int = 5,
    degree: int = 3,
    seed: int | None = None,
) -> float:
    """EVPPI for a parameter (or parameter group) at one willingness to pay.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        draws: Parameter draw matrix whose index equals the outcomes'
            iteration index (the shared-index contract).
        params: Column name(s) in ``draws`` to value jointly.
        wtp: Willingness to pay per unit of effect.
        effect: Effect column (default: the primary effect).
        method: ``"spline"`` (default) or ``"gp"`` metamodel; see
            `heormodel.voi._metamodel.fitted_conditional_means`.
        n_knots: Spline knot count.
        degree: Spline degree.
        seed: Subsample seed for the GP method.

    Returns:
        The EVPPI estimate (same monetary units as NMB), clipped at zero.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.voi import evppi
        >>> rng = np.random.default_rng(0)
        >>> e_b = rng.normal(0.0, 1.0, 4000)
        >>> draws = pd.DataFrame({"e_b": e_b},
        ...                      index=pd.RangeIndex(4000, name="iteration"))
        >>> costs = pd.DataFrame({"A": np.zeros(4000), "B": np.zeros(4000)})
        >>> effects = pd.DataFrame({"A": np.zeros(4000), "B": e_b})
        >>> out = Outcomes.from_wide(costs, effects)
        >>> v = evppi(out, draws, "e_b", wtp=1.0)
        >>> abs(v - 0.399) < 0.05  # analytic value is 1/sqrt(2*pi)
        True
    """
    cols = [params] if isinstance(params, str) else list(params)
    missing = [c for c in cols if c not in draws.columns]
    if missing:
        raise KeyError(f"Parameters not found in draw matrix: {missing}.")
    require_shared_index(
        draws.index,
        outcomes.iterations,
        "draws",
        detail="EVPPI needs the parameter/outcome linkage preserved by run_psa.",
    )
    nb = nmb(outcomes, wtp, effect=effect)
    fitted = fitted_conditional_means(
        draws[cols], nb, method=method, n_knots=n_knots, degree=degree, seed=seed
    )
    return max(0.0, voi_from_fitted(fitted))


def evppi_ranking(
    outcomes: Outcomes,
    draws: pd.DataFrame,
    wtp: float,
    *,
    params: Sequence[str] | None = None,
    effect: str | None = None,
    method: str = "spline",
    **kwargs: int,
) -> pd.Series:
    """Single-parameter EVPPI for each parameter, sorted descending.

    A convenience sweep for prioritising research targets.

    Example:
        >>> # see evppi() for setup; ranking returns a sorted Series
        >>> # evppi_ranking(out, draws, wtp=1.0).index[0]
    """
    cols = list(params) if params is not None else list(draws.columns)
    values = {
        p: evppi(outcomes, draws, p, wtp, effect=effect, method=method, **kwargs) for p in cols
    }
    return pd.Series(values, name="evppi").sort_values(ascending=False)
