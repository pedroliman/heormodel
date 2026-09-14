"""Shared metamodel machinery for regression-based VoI estimators.

Both EVPPI and regression-based EVSI reduce to the same computation: for
each intervention, regress net benefit on some conditioning variables (parameter
draws for EVPPI, simulated study summaries for EVSI), then compare the
expected maximum of the fitted conditional means with the maximum of their
expectations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

_GP_MAX_FIT = 500


class _TensorProductSplineFeatures(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Per-column spline basis, plus pairwise tensor-product interactions.

    Fits one cubic-spline basis per input column and, for every pair of
    columns, appends the row-wise outer product of their two bases (a
    tensor-product spline term). A linear model on the additive bases alone
    can only sum each parameter's marginal effect on the outcome; the
    tensor-product terms let it represent a product between two parameters,
    such as a relative risk multiplying a baseline probability, which the
    additive terms cannot. With one input column there is no pair to form,
    so the output is exactly that column's additive spline basis.

    Example:
        >>> import numpy as np
        >>> features = _TensorProductSplineFeatures(n_knots=4, degree=3)
        >>> x = np.column_stack([np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)])
        >>> basis = features.fit_transform(x)
        >>> basis.shape[1] > 2 * (4 + 3 - 1)  # additive bases plus interactions
        True
    """

    def __init__(self, n_knots: int = 5, degree: int = 3) -> None:
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> Any:
        xv = np.asarray(x, dtype=np.float64)
        self._splines_ = [
            SplineTransformer(n_knots=self.n_knots, degree=self.degree, include_bias=False).fit(
                xv[:, [j]]
            )
            for j in range(xv.shape[1])
        ]
        return self

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xv = np.asarray(x, dtype=np.float64)
        bases = [spline.transform(xv[:, [j]]) for j, spline in enumerate(self._splines_)]
        features = list(bases)
        for j in range(len(bases)):
            for k in range(j + 1, len(bases)):
                interaction = bases[j][:, :, None] * bases[k][:, None, :]
                features.append(interaction.reshape(xv.shape[0], -1))
        return np.concatenate(features, axis=1)


def _spline_pipeline(n_knots: int, degree: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        _TensorProductSplineFeatures(n_knots=n_knots, degree=degree),
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
        method: ``"spline"`` (cubic-spline basis per column, plus pairwise
            tensor-product interaction terms when ``x`` has more than one
            column, fed into a linear model; fast, default) or ``"gp"``
            (Gaussian-process regression fitted on a subsample of at most
            500 points, then evaluated on all). The tensor-product terms let
            ``"spline"`` represent an interaction between two conditioning
            variables, such as a relative risk multiplying a baseline
            probability; with three or more grouped variables it still
            captures every pairwise interaction, but not three-way ones,
            so ``"gp"`` remains the more general, slower alternative.
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
