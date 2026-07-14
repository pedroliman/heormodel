"""Validate `heormodel.survival` against closed forms and convergence.

The module is exercised on three levels. The curve, family, algebra, and
transition functions are checked against analytic values a survival curve must
satisfy. The ``lifelines`` adapter is checked against the fitter's own evaluation
and against the moments it samples from. The two engines and the parameter
recovery reproduce the item-18 acceptance values through the public interface,
the same numbers the phase-1 replication produced: the discounted life expectancy
4.92709, and its recovery as a fitted Weibull is refit from a censored sample and
its uncertainty propagated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import gamma
from survival_models import (
    INTERVENTION,
    cohort_engine,
    continuous_engine,
    discounted_life_expectancy,
)

from heormodel import survival as sv
from heormodel.run import run_psa

SHAPE, SCALE, DISCOUNT = 1.2, 6.0, 0.03
ANALYTIC_DLE = 4.927093478597975


# --- Curve, families, and algebra: analytic checks ---


def test_weibull_reference_rows():
    """The Weibull curve reproduces the reference survival and cycle probabilities."""
    curve = sv.weibull(SHAPE, SCALE)
    assert SCALE * gamma(1 + 1 / SHAPE) == pytest.approx(5.64394, abs=1e-5)
    assert discounted_life_expectancy(curve) == pytest.approx(ANALYTIC_DLE, abs=1e-5)
    expected = [0.10994, 0.14025, 0.15439, 0.16428, 0.17201]
    assert curve.cycle_transition_probabilities(5) == pytest.approx(expected, abs=1e-5)


def test_exponential_closed_form():
    """A constant hazard gives discounted life-years 1 / (discount + hazard)."""
    hazard = 0.2
    expected = 1.0 / (DISCOUNT + hazard)
    assert discounted_life_expectancy(sv.exponential(hazard)) == pytest.approx(expected, abs=1e-9)


def test_hazard_ratio_closed_form():
    """Applying a hazard ratio r gives 1 / (discount + r * hazard) exactly."""
    hazard, ratio = 0.2, 0.6
    treated = sv.apply_hazard_ratio(sv.exponential(hazard), ratio)
    expected = 1.0 / (DISCOUNT + ratio * hazard)
    assert discounted_life_expectancy(treated) == pytest.approx(expected, abs=1e-9)
    assert float(treated.hazard_at(5.0)) == pytest.approx(ratio * hazard, abs=1e-12)


def test_acceleration_factor_scales_time():
    """S'(t) = S(t / factor), so survival at factor*t equals the base at t."""
    base = sv.weibull(SHAPE, SCALE)
    slower = sv.apply_acceleration_factor(base, 2.0)
    grid = np.array([1.0, 4.0, 9.0])
    assert slower.survival(2.0 * grid) == pytest.approx(base.survival(grid), abs=1e-12)


def test_mix_is_weighted_survival():
    """A mixture's survival is the weight-average of the component survivals."""
    a, b = sv.exponential(0.1), sv.exponential(0.3)
    blend = sv.mix([a, b], [0.25, 0.75])
    grid = np.array([1.0, 5.0, 10.0])
    expected = 0.25 * a.survival(grid) + 0.75 * b.survival(grid)
    assert blend.survival(grid) == pytest.approx(expected, abs=1e-12)


def test_splice_is_continuous_and_monotone():
    """Splicing is continuous at the cutpoint and keeps survival monotone."""
    early = sv.weibull(SHAPE, SCALE)
    spliced = sv.splice(early, sv.exponential(0.05), cutpoint=5.0)
    assert float(spliced.survival(5.0)) == pytest.approx(float(early.survival(5.0)), abs=1e-12)
    grid = np.linspace(0, 30, 300)
    survival = spliced.survival(grid)
    assert np.all(np.diff(survival) <= 1e-12)


def test_bisection_sampler_matches_quantile():
    """The numeric inverse-transform sampler matches the closed-form quantile."""
    # A curve with no analytic quantile forces the bisection path.
    weibull = sv.weibull(SHAPE, SCALE)
    numeric = sv.SurvivalCurve(cumulative_hazard=weibull.cumulative_hazard)
    threshold = np.random.default_rng(0).exponential(size=50_000)
    exact = SCALE * threshold ** (1 / SHAPE)
    got = numeric._invert(threshold)
    assert got == pytest.approx(exact, rel=1e-9)


# --- Transition matrix ---


def test_transition_matrix_single_decrement():
    """One absorbing cause reproduces 1 - S(k+1) / S(k) and rows sum to one."""
    curve = sv.weibull(SHAPE, SCALE)
    transition = sv.to_transition_matrix({("alive", "dead"): curve}, ("alive", "dead"), 5)
    assert transition[:, 0, 1] == pytest.approx(curve.cycle_transition_probabilities(5), abs=1e-12)
    assert transition.sum(axis=2) == pytest.approx(np.ones((5, 2)), abs=1e-12)


def test_transition_matrix_competing_exponential():
    """Two exponential causes split the exit probability in proportion to their rates."""
    rate_a, rate_b = 0.1, 0.2
    causes = {
        ("well", "cause_a"): sv.exponential(rate_a),
        ("well", "cause_b"): sv.exponential(rate_b),
    }
    transition = sv.to_transition_matrix(causes, ("well", "cause_a", "cause_b"), 1)[0]
    total = rate_a + rate_b
    exit_probability = 1 - np.exp(-total)
    assert transition[0, 0] == pytest.approx(np.exp(-total), abs=1e-12)
    assert transition[0, 1] == pytest.approx(rate_a / total * exit_probability, abs=1e-12)
    assert transition[0, 2] == pytest.approx(rate_b / total * exit_probability, abs=1e-12)


# --- lifelines adapter ---


def _fit(size, censor=12.0, seed=20260714):
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(seed)
    event_time = sv.weibull(SHAPE, SCALE).sample_time(rng, size)
    observed = np.minimum(event_time, censor)
    return lifelines.WeibullFitter().fit(observed, (event_time <= censor).astype(float))


def test_from_lifelines_matches_fitter():
    """The adapted point-estimate curve matches the fitter's own survival function."""
    fit = _fit(2_000)
    times = np.array([1.0, 5.0, 10.0])
    adapted = sv.from_lifelines(fit).survival(times)
    assert adapted == pytest.approx(fit.survival_function_at_times(times).to_numpy(), abs=1e-12)


def test_from_lifelines_at_sampled_params():
    """A curve built from a parameter row uses that row's parameters."""
    fit = _fit(2_000)
    draws = sv.sample_params(fit, 5, seed=0)
    row = draws.iloc[0]
    curve = sv.from_lifelines(fit, row)
    expected = np.exp(-np.asarray(fit._cumulative_hazard(row.to_numpy(), np.array([3.0]))))
    assert float(curve.survival(3.0)) == pytest.approx(float(expected[0]), abs=1e-12)


def test_sample_params_recovers_moments():
    """Sampling recovers the fitted mean and covariance and lands on the iteration index."""
    fit = _fit(2_000)
    draws = sv.sample_params(fit, 200_000, seed=1)
    assert draws.index.name == "iteration"
    assert list(draws.columns) == list(fit.params_.index)
    assert draws.mean().to_numpy() == pytest.approx(fit.params_.to_numpy(), abs=5e-3)
    assert np.cov(draws.to_numpy().T) == pytest.approx(np.asarray(fit.variance_matrix_), abs=5e-4)


# --- The two engines and parameter recovery, through the public module ---


def test_two_engines_recover_discounted_life_expectancy():
    """The continuous sampler and the cohort both recover the discounted value."""

    def weibull_of(params):
        return sv.weibull(params["shape"], params["scale"])

    draws = pd.DataFrame(
        {"shape": [SHAPE], "scale": [SCALE]}, index=pd.RangeIndex(1, name="iteration")
    )
    continuous = run_psa(
        continuous_engine(weibull_of, 200_000), draws, seed=1, sequential=True
    ).outcomes.summary().loc[INTERVENTION, "lifeyears"]
    cohort = cohort_engine(weibull_of).evaluate(draws).summary().loc[INTERVENTION, "lifeyears"]
    assert continuous == pytest.approx(ANALYTIC_DLE, rel=0.005)
    assert cohort == pytest.approx(ANALYTIC_DLE, rel=0.005)


def test_parameter_recovery_converges():
    """As the sample grows, the lifelines fit converges to shape 1.2 and scale 6.0."""
    small = _fit(300)
    large = _fit(200_000)
    assert large.rho_ == pytest.approx(SHAPE, abs=0.01)  # rho is the Weibull shape
    assert large.lambda_ == pytest.approx(SCALE, abs=0.05)  # lambda is the scale
    se_small = np.sqrt(np.diag(np.asarray(small.variance_matrix_)))
    se_large = np.sqrt(np.diag(np.asarray(large.variance_matrix_)))
    assert np.all(se_large < se_small / 15.0)  # standard errors shrink like 1/sqrt(n)


def _propagate(fit, n, rng):
    params = sv.sample_params(fit, n, seed=int(rng.integers(1 << 30)))
    return params.apply(lambda row: discounted_life_expectancy(sv.from_lifelines(fit, row)), axis=1)


def test_analytic_convergence_and_trial_size_interval():
    """The fitted and probabilistic life expectancies converge to 4.92709."""
    large = _fit(200_000)
    fitted_dle = discounted_life_expectancy(sv.from_lifelines(large))
    assert fitted_dle == pytest.approx(ANALYTIC_DLE, abs=0.02)
    rng = np.random.default_rng(1)
    dle_large = _propagate(large, 2_000, rng)
    assert dle_large.mean() == pytest.approx(ANALYTIC_DLE, abs=0.02)
    assert dle_large.std() < 0.05  # spread shrinks toward zero at large sample size

    dle_small = _propagate(_fit(300), 2_000, rng)
    lower, upper = np.percentile(dle_small, [2.5, 97.5])
    assert lower < ANALYTIC_DLE < upper  # analytic value inside the trial-size interval
