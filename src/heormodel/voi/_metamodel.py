"""Shared metamodel machinery for regression-based VoI estimators.

Both EVPPI and regression-based EVSI reduce to the same computation: for
each intervention, regress net benefit on some conditioning variables (parameter
draws for EVPPI, simulated study summaries for EVSI), then compare the
expected maximum of the fitted conditional means with the maximum of their
expectations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

_GP_MAX_FIT = 500


def _spline_pipeline(n_knots: int, degree: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False),
        LinearRegression(),
    )


def fitted_conditional_means(
    x: pd.DataFrame,
    nb: pd.DataFrame,
    *,
    method: str = "spline",
    n_knots: int = 5,
    degree: int = 3,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Fit a flexible regression of each intervention's NB on ``x``.

    Args:
        x: Conditioning variables, one row per iteration.
        nb: Net benefit (iterations x interventions), aligned with ``x``.
        method: ``"spline"`` (additive cubic-spline basis + linear model,
            fast, default) or ``"gp"`` (Gaussian-process regression fitted
            on a subsample of at most 500 points, then evaluated on all).
        n_knots, degree: Spline basis controls (``method="spline"``).
        seed: Subsample seed (``method="gp"``).

    Returns:
        Array (iterations x interventions) of fitted conditional-mean NB.
    """
    if len(x) != len(nb):
        raise ValueError("x and nb must have the same number of rows.")
    xv = x.to_numpy(dtype=np.float64)
    nbv = nb.to_numpy(dtype=np.float64)
    fitted = np.empty_like(nbv)
    for j in range(nbv.shape[1]):
        y = nbv[:, j]
        if np.ptp(y) == 0.0:  # constant NB needs no regression
            fitted[:, j] = y
            continue
        if method == "spline":
            model = _spline_pipeline(n_knots, degree)
            model.fit(xv, y)
            fitted[:, j] = model.predict(xv)
        elif method == "gp":
            rng = np.random.default_rng(seed)
            if len(xv) > _GP_MAX_FIT:
                ix = rng.choice(len(xv), size=_GP_MAX_FIT, replace=False)
            else:
                ix = np.arange(len(xv))
            kernel = ConstantKernel(1.0) * RBF(np.ones(xv.shape[1])) + WhiteKernel(1.0)
            gp = make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0),
            )
            gp.fit(xv[ix], y[ix])
            fitted[:, j] = gp.predict(xv)
        else:
            raise ValueError(f"Unknown metamodel method: {method!r} (use 'spline' or 'gp').")
    return fitted


def voi_from_fitted(fitted: NDArray[np.float64]) -> float:
    """VoI statistic: ``E[max_d g_d] - max_d E[g_d]`` over fitted values."""
    return float(fitted.max(axis=1).mean() - fitted.mean(axis=0).max())
