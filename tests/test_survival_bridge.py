"""Validate the survival-analysis bridge against closed forms and convergence.

The file exercises the bespoke helpers in ``examples/survival_bridge.py``: the
Weibull survival curve, the two ways an engine consumes it, the Weibull maximum-
likelihood fit, and sampling from the fit's asymptotic distribution. Three groups
of checks anchor it. The reference table rows follow from the analytic curve. The
continuous sampler and the cohort both recover the discounted life expectancy and
agree within the cycle-correction error. As the simulated sample grows, the fit
converges to the data-generating shape 1.2 and scale 6.0 with standard errors
shrinking like one over the square root of the sample size, and both the fitted-
model and probabilistic discounted life expectancies converge to the analytic
4.92709.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import gamma
from survival_bridge import (
    INTERVENTION,
    annual_death_probabilities,
    apply_hazard_ratio,
    cohort_engine,
    cohort_life_years,
    continuous_engine,
    discounted_life_expectancy,
    fit_weibull,
    sample_event_times,
    sample_survival_params,
)

from heormodel.run import run_psa

SHAPE, SCALE, DISCOUNT = 1.2, 6.0, 0.03
ANALYTIC_DLE = 4.927093478597975


def _draws(shape, scale, utility=1.0):
    return pd.DataFrame(
        {"shape": [shape], "scale": [scale], "utility": [utility]},
        index=pd.RangeIndex(1, name="iteration"),
    )


def test_reference_table_rows():
    """Every analytic row of the reference table reproduces to five digits."""
    assert SCALE * gamma(1 + 1 / SHAPE) == pytest.approx(5.64394, abs=1e-5)
    assert discounted_life_expectancy(SHAPE, SCALE) == pytest.approx(4.92709, abs=1e-5)
    qaly = discounted_life_expectancy(SHAPE, SCALE, utility=0.85)
    assert qaly == pytest.approx(4.18803, abs=1e-5)
    death = annual_death_probabilities(SHAPE, SCALE, 5)
    expected = [0.10994, 0.14025, 0.15439, 0.16428, 0.17201]
    assert death == pytest.approx(expected, abs=1e-5)
    assert cohort_life_years(SHAPE, SCALE, 60) == pytest.approx(4.93604, abs=1e-5)


def test_two_engines_recover_discounted_life_expectancy():
    """The continuous sampler and the cohort both recover the discounted value."""
    continuous = run_psa(
        continuous_engine(200_000), _draws(SHAPE, SCALE), seed=1, sequential=True
    ).outcomes.summary().loc[INTERVENTION, "lifeyears"]
    cohort = cohort_engine().evaluate(_draws(SHAPE, SCALE)).summary().loc[INTERVENTION, "lifeyears"]
    # Continuous integration matches within Monte Carlo error; the cohort matches
    # within the cycle-correction error.
    assert continuous == pytest.approx(ANALYTIC_DLE, rel=0.005)
    assert cohort == pytest.approx(ANALYTIC_DLE, rel=0.005)


def test_constant_hazard_closed_form():
    """A constant hazard gives discounted life-years 1 / (discount + hazard)."""
    hazard = 0.2
    shape, scale = 1.0, 1.0 / hazard  # exponential: Weibull with shape one
    expected = 1.0 / (DISCOUNT + hazard)
    assert discounted_life_expectancy(shape, scale) == pytest.approx(expected, abs=1e-9)
    engine = run_psa(
        continuous_engine(200_000, horizon=400.0), _draws(shape, scale), seed=3, sequential=True
    ).outcomes.summary().loc[INTERVENTION, "lifeyears"]
    assert engine == pytest.approx(expected, rel=0.01)


def test_hazard_ratio_closed_form():
    """Applying a hazard ratio r gives 1 / (discount + r * hazard) exactly."""
    hazard, ratio = 0.2, 0.6
    shape, scale = apply_hazard_ratio(1.0, 1.0 / hazard, ratio)
    expected = 1.0 / (DISCOUNT + ratio * hazard)
    assert discounted_life_expectancy(shape, scale) == pytest.approx(expected, abs=1e-9)


def _fit_at(size, censor=12.0, seed=20260714):
    rng = np.random.default_rng(seed)
    event_time = sample_event_times(rng, size, SHAPE, SCALE)
    observed = np.minimum(event_time, censor)
    observed_event = (event_time <= censor).astype(float)
    return fit_weibull(observed, observed_event)


def test_parameter_recovery_converges():
    """The fit converges to the data-generating parameters as the sample grows."""
    _, _, cov_small = _fit_at(300)
    shape_large, scale_large, cov_large = _fit_at(200_000)
    assert shape_large == pytest.approx(SHAPE, abs=0.01)
    assert scale_large == pytest.approx(SCALE, abs=0.05)
    # Standard errors shrink like one over the square root of the sample size:
    # a 667-fold larger sample cuts them by about a factor of 26.
    se_small = np.sqrt(np.diag(cov_small))
    se_large = np.sqrt(np.diag(cov_large))
    assert np.all(se_large < se_small / 15.0)


def _propagate(params):
    """Discounted life expectancy for each sampled parameter set."""
    return params.apply(lambda row: discounted_life_expectancy(row["shape"], row["scale"]), axis=1)


def test_analytic_convergence_and_trial_size_interval():
    """The fitted and probabilistic life expectancies converge to 4.92709."""
    shape_large, scale_large, cov_large = _fit_at(200_000)
    fitted_dle = discounted_life_expectancy(shape_large, scale_large)
    assert fitted_dle == pytest.approx(ANALYTIC_DLE, abs=0.02)
    rng = np.random.default_rng(1)
    dle_large = _propagate(sample_survival_params(shape_large, scale_large, cov_large, 2_000, rng))
    assert dle_large.mean() == pytest.approx(ANALYTIC_DLE, abs=0.02)
    assert dle_large.std() < 0.05  # spread shrinks toward zero at large sample size

    # At trial size the analytic value lies inside the 95% credible interval.
    shape_small, scale_small, cov_small = _fit_at(300)
    dle_small = _propagate(sample_survival_params(shape_small, scale_small, cov_small, 2_000, rng))
    lower, upper = np.percentile(dle_small, [2.5, 97.5])
    assert lower < ANALYTIC_DLE < upper


def test_sample_params_recovers_fit_moments():
    """Sampling recovers the fitted mean and covariance as the draw count grows."""
    shape_hat, scale_hat, cov_log = _fit_at(300)
    rng = np.random.default_rng(0)
    params = sample_survival_params(shape_hat, scale_hat, cov_log, 200_000, rng)
    log_draws = np.log(params[["shape", "scale"]].to_numpy())
    assert log_draws.mean(axis=0) == pytest.approx(np.log([shape_hat, scale_hat]), abs=1e-3)
    assert np.cov(log_draws.T) == pytest.approx(cov_log, abs=5e-4)
