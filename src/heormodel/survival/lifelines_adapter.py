"""Reading a fitted survival model into a curve and a draw matrix.

`from_lifelines` and `sample_params` read a fitted parametric survival model
and never import a fitting package themselves. Instead they duck-type against
the interface a fitted model exposes: a `cumulative_hazard_at_times` method
and `params_`/`variance_matrix_` attributes, the public surface every
`lifelines` parametric fitter (`WeibullFitter`, `ExponentialFitter`,
`LogNormalFitter`, and the rest) shares once `.fit(...)` has run. Because
neither function imports `lifelines`, `heormodel.survival` imports cleanly
whether or not it is installed; only the object passed in needs to look like
a fitted model. Install the fitting package with `uv pip install
'heormodel[survival]'`.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from heormodel._util import as_rng
from heormodel.survival.curve import SurvivalCurve, numeric_inverse_cumulative_hazard

_FloatArray = NDArray[np.float64]


class FittedSurvivalModel(Protocol):
    """The fitted-model interface `from_lifelines` and `sample_params` read.

    Every `lifelines` parametric univariate fitter satisfies this once fit:
    `params_` and `variance_matrix_` hold the maximum-likelihood estimate and
    its asymptotic covariance, and `cumulative_hazard_at_times` evaluates the
    fitted cumulative hazard. Nothing here is `heormodel`-specific; it names
    the subset of a fitted model's public attributes these two functions use.
    """

    params_: pd.Series
    variance_matrix_: pd.DataFrame

    def cumulative_hazard_at_times(self, times: Any) -> Any: ...


def _require_fitted_model(fitted_model: Any, needs: tuple[str, ...]) -> None:
    missing = [name for name in needs if not hasattr(fitted_model, name)]
    if missing:
        raise TypeError(
            f"{type(fitted_model).__name__} does not look like a fitted survival model: "
            f"missing {missing}. Pass a `lifelines` parametric fitter after calling "
            "its `.fit(...)` method."
        )


def from_lifelines(fitted_model: FittedSurvivalModel, *, label: str | None = None) -> SurvivalCurve:
    """Build a `SurvivalCurve` from a fitted parametric survival model.

    Reads the model's own `cumulative_hazard_at_times` method, so the curve
    matches the fit exactly whatever family was chosen. Because the resulting
    cumulative hazard has no closed-form inverse in general, sampling from it
    falls back to `heormodel.survival.curve.numeric_inverse_cumulative_hazard`.

    Args:
        fitted_model: A fitted parametric survival model, for example the
            result of `lifelines.WeibullFitter().fit(durations, event_observed)`.
        label: Description of the curve for `repr`. Defaults to the fitted
            model's class name.

    Returns:
        A `SurvivalCurve` at the model's point estimate.

    Example:
        >>> from lifelines import WeibullFitter  # doctest: +SKIP
        >>> fit = WeibullFitter().fit(durations, event_observed)  # doctest: +SKIP
        >>> curve = from_lifelines(fit)  # doctest: +SKIP
    """
    _require_fitted_model(fitted_model, ("cumulative_hazard_at_times",))

    def cumulative_hazard(time: _FloatArray) -> _FloatArray:
        time = np.asarray(time, dtype=np.float64)
        values = fitted_model.cumulative_hazard_at_times(time)
        return np.asarray(values, dtype=np.float64).reshape(time.shape)

    curve_label = label or type(fitted_model).__name__
    return SurvivalCurve(
        cumulative_hazard, numeric_inverse_cumulative_hazard(cumulative_hazard), curve_label
    )


def sample_params(
    fitted_model: FittedSurvivalModel,
    n: int,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw parameter sets from a fit's asymptotic covariance.

    Draws `n` parameter vectors from the multivariate normal approximation to
    the maximum-likelihood estimate, `Normal(params_, variance_matrix_)`, the
    same approximation the fitting package itself uses for confidence
    intervals. The draws land on the canonical `iteration` index, so the
    fitted survival curve's uncertainty flows into `heormodel.run.run_psa`
    alongside every other parameter, keyed by the fitted model's own
    parameter names (for `lifelines.WeibullFitter`, `lambda_` for the scale
    and `rho_` for the shape).

    Args:
        fitted_model: A fitted parametric survival model exposing `params_`
            and `variance_matrix_`.
        n: Number of iterations (rows) to draw.
        seed: Integer seed or `numpy` generator for reproducibility.

    Returns:
        DataFrame with `n` rows, a `RangeIndex` named `"iteration"`, and one
        column per fitted parameter.

    Example:
        >>> from lifelines import WeibullFitter  # doctest: +SKIP
        >>> fit = WeibullFitter().fit(durations, event_observed)  # doctest: +SKIP
        >>> draws = sample_params(fit, n=1000, seed=1)  # doctest: +SKIP
        >>> draws.index.name  # doctest: +SKIP
        'iteration'
    """
    _require_fitted_model(fitted_model, ("params_", "variance_matrix_"))
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    mean = np.asarray(fitted_model.params_, dtype=np.float64)
    covariance = np.asarray(fitted_model.variance_matrix_, dtype=np.float64)
    rng = as_rng(seed)
    draws = rng.multivariate_normal(mean, covariance, size=n)
    columns = list(fitted_model.params_.index)
    return pd.DataFrame(draws, columns=columns, index=pd.RangeIndex(n, name="iteration"))
