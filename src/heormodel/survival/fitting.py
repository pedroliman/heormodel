"""Read a fitted parametric survival model and carry its uncertainty.

The adapter is duck-typed against the fitted-model objects a survival-fitting
package produces, so `heormodel.survival` has no hard dependency on one. It reads
three attributes every parametric univariate fit exposes: the estimated
parameters, their asymptotic covariance, and a function that evaluates the
cumulative hazard at a given parameter vector. `from_lifelines` builds a
`SurvivalCurve` at the point estimate or at one sampled parameter vector;
`sample_params` draws parameter vectors from the fit's asymptotic distribution
onto the canonical ``iteration`` index, so survival uncertainty shares one index
with every other parameter and flows through `heormodel.run.run_psa` unchanged.

Install the fitting package with the ``survival`` extra
(``uv pip install 'heormodel[survival]'``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from heormodel.survival.curve import Array, SurvivalCurve


def _parameter_names(fitter: Any) -> list[str]:
    return [str(name) for name in fitter.params_.index]


def from_lifelines(
    fitter: Any, params: pd.Series | None = None, prefix: str = ""
) -> SurvivalCurve:
    """Build a `SurvivalCurve` from a fitted parametric survival model.

    With ``params`` omitted the curve is the point estimate. Passing a parameter
    row (one draw from `sample_params`) returns the curve at that draw, which is
    how a model function turns each iteration's sampled parameters into a curve.

    Args:
        fitter: A fitted parametric univariate model exposing ``params_`` (a
            named parameter series), ``variance_matrix_``, and
            ``_cumulative_hazard(values, times)`` / ``_hazard(values, times)``.
            The ``lifelines`` univariate fitters (Weibull, log-normal,
            log-logistic, generalized gamma, and the rest) satisfy this.
        params: Optional parameter row. When given, its entries named
            ``prefix + parameter name`` supply the parameter vector.
        prefix: Column-name prefix used when a draw matrix carries more than one
            curve's parameters.

    Returns:
        A `SurvivalCurve` evaluating the fit's cumulative hazard and hazard.

    Example:
        >>> from lifelines import WeibullFitter  # doctest: +SKIP
        >>> fit = WeibullFitter().fit(durations, observed)  # doctest: +SKIP
        >>> from heormodel.survival import from_lifelines
        >>> curve = from_lifelines(fit)  # doctest: +SKIP
    """
    names = _parameter_names(fitter)
    if params is None:
        values = np.asarray(fitter.params_.values, dtype=float)
    else:
        values = np.asarray([float(params[prefix + name]) for name in names], dtype=float)

    def cumulative_hazard(t: Array) -> Array:
        cumulative = fitter._cumulative_hazard(values, np.asarray(t, dtype=float))
        return np.asarray(cumulative, dtype=float)

    def hazard(t: Array) -> Array:
        return np.asarray(fitter._hazard(values, np.asarray(t, dtype=float)), dtype=float)

    return SurvivalCurve(
        cumulative_hazard=cumulative_hazard,
        hazard=hazard,
        name=f"{type(fitter).__name__} fit",
    )


def sample_params(
    fitter: Any, n: int, seed: int | None = None, prefix: str = ""
) -> pd.DataFrame:
    """Draw parameter sets from the fit's asymptotic distribution.

    Samples ``n`` parameter vectors from the multivariate normal centered at the
    fitted parameters with their estimated covariance, and returns them on the
    canonical ``iteration`` index. Combine the frame with other parameter draws
    (for example through `heormodel.params.mix_draws`) so one draw matrix carries
    the whole model's uncertainty.

    Args:
        fitter: A fitted model exposing ``params_`` and ``variance_matrix_``.
        n: Number of parameter sets to draw.
        seed: Seed for the draw.
        prefix: Prefix applied to each parameter's column name.

    Returns:
        A ``DataFrame`` of shape ``(n, n_parameters)`` indexed by ``iteration``
        from 1 to ``n``, one column per fitted parameter.

    Example:
        >>> from lifelines import WeibullFitter  # doctest: +SKIP
        >>> fit = WeibullFitter().fit(durations, observed)  # doctest: +SKIP
        >>> from heormodel.survival import sample_params
        >>> draws = sample_params(fit, n=1000, seed=1)  # doctest: +SKIP
    """
    if n < 1:
        raise ValueError("n must be at least one.")
    names = _parameter_names(fitter)
    mean = np.asarray(fitter.params_.values, dtype=float)
    covariance = np.asarray(fitter.variance_matrix_, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.multivariate_normal(mean, covariance, size=n)
    columns = [prefix + name for name in names]
    index = pd.RangeIndex(1, n + 1, name="iteration")
    return pd.DataFrame(draws, columns=columns, index=index)
