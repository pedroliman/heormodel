"""Validate `heormodel.survival` against closed forms and convergence.

Three groups of checks anchor this file. The reference table rows, produced
through `heormodel.survival.weibull` and `heormodel.survival.to_transition_matrix`,
match the values `examples/survival_bridge.py` (phase 1) established. The
curve algebra (`apply_hazard_ratio`, `apply_acceleration_factor`, `mix`,
`splice`) and `to_transition_matrix`'s competing-risks path match their closed
forms exactly. The `lifelines` adapter (`from_lifelines`, `sample_params`,
skipped when `lifelines` is not installed) reproduces the parameter-recovery
and convergence exercise: as the simulated sample grows, the fit converges to
the data-generating shape 1.2 and scale 6.0, and both the fitted-model and
probabilistic discounted life expectancies converge to the analytic 4.92709.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from survival_models import (
    INTERVENTION,
    cohort_engine,
    cohort_life_years,
    continuous_engine,
    discounted_life_expectancy,
)

from heormodel.run import run_psa
from heormodel.survival import (
    apply_acceleration_factor,
    apply_hazard_ratio,
    exponential,
    gompertz,
    mix,
    splice,
    to_transition_matrix,
    weibull,
)
from heormodel.survival.curve import SurvivalCurve, numeric_inverse_cumulative_hazard

SHAPE, SCALE, DISCOUNT = 1.2, 6.0, 0.03
ANALYTIC_DLE = 4.927093478597975


def _draws(shape: float, scale: float, utility: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"shape": [shape], "scale": [scale], "utility": [utility]},
        index=pd.RangeIndex(1, name="iteration"),
    )


class TestReferenceTable:
    def test_reference_table_rows(self):
        """Every reference-table row reproduces to five digits through the public API."""
        curve = weibull(SHAPE, SCALE)
        assert discounted_life_expectancy(curve, rate=0.0) == pytest.approx(5.64394, abs=1e-5)
        assert discounted_life_expectancy(curve) == pytest.approx(4.92709, abs=1e-5)
        qaly = discounted_life_expectancy(curve, utility=0.85)
        assert qaly == pytest.approx(4.18803, abs=1e-5)
        death = to_transition_matrix(curve, n_cycles=5)[:, 0, 1]
        expected = [0.10994, 0.14025, 0.15439, 0.16428, 0.17201]
        assert death == pytest.approx(expected, abs=1e-5)
        assert cohort_life_years(curve, 60) == pytest.approx(4.93604, abs=1e-5)

    def test_two_engines_recover_discounted_life_expectancy(self):
        """The continuous sampler and the cohort both recover the discounted value."""
        continuous = (
            run_psa(continuous_engine(200_000), _draws(SHAPE, SCALE, 1.0), seed=1, sequential=True)
            .outcomes.summary()
            .loc[INTERVENTION, "lifeyears"]
        )
        cohort_summary = cohort_engine().evaluate(_draws(SHAPE, SCALE, 1.0)).summary()
        cohort = cohort_summary.loc[INTERVENTION, "lifeyears"]
        # Continuous integration matches within Monte Carlo error; the cohort
        # matches within the cycle-correction error.
        assert continuous == pytest.approx(ANALYTIC_DLE, rel=0.005)
        assert cohort == pytest.approx(ANALYTIC_DLE, rel=0.005)


class TestClosedForms:
    def test_constant_hazard_closed_form(self):
        """A constant hazard gives discounted life-years 1 / (discount + hazard)."""
        hazard = 0.2
        curve = exponential(hazard)
        expected = 1.0 / (DISCOUNT + hazard)
        assert discounted_life_expectancy(curve) == pytest.approx(expected, abs=1e-9)

    def test_hazard_ratio_closed_form(self):
        """Applying a hazard ratio r gives 1 / (discount + r * hazard) exactly."""
        hazard, ratio = 0.2, 0.6
        curve = apply_hazard_ratio(exponential(hazard), ratio)
        expected = 1.0 / (DISCOUNT + ratio * hazard)
        assert discounted_life_expectancy(curve) == pytest.approx(expected, abs=1e-9)

    def test_hazard_ratio_recovers_engine_life_expectancy(self):
        """The hazard-ratio curve also drives the continuous engine to the closed form."""
        hazard, ratio = 0.2, 0.6
        expected = 1.0 / (DISCOUNT + ratio * hazard)
        # An exponential curve is weibull(shape=1, scale=1 / rate).
        draws = _draws(1.0, 1.0 / (ratio * hazard), 1.0)
        engine = (
            run_psa(continuous_engine(200_000, horizon=400.0), draws, seed=3, sequential=True)
            .outcomes.summary()
            .loc[INTERVENTION, "lifeyears"]
        )
        assert engine == pytest.approx(expected, rel=0.01)

    def test_acceleration_factor_rescales_time(self):
        """An acceleration factor `a` maps `survival(t)` to `curve.survival(t / a)`."""
        base = weibull(SHAPE, SCALE)
        stretched = apply_acceleration_factor(base, acceleration_factor=2.0)
        grid = np.array([1.0, 3.0, 6.0, 10.0])
        assert stretched.survival(2.0 * grid) == pytest.approx(base.survival(grid))


class TestCurveAlgebra:
    def test_mix_matches_weighted_survival(self):
        """`mix`'s survival function is the weighted average of its components."""
        curve_a, curve_b = exponential(0.1), exponential(0.5)
        curve = mix([curve_a, curve_b], [0.3, 0.7])
        grid = np.array([0.5, 2.0, 8.0])
        expected = 0.3 * curve_a.survival(grid) + 0.7 * curve_b.survival(grid)
        assert curve.survival(grid) == pytest.approx(expected)

    def test_mix_requires_valid_weights(self):
        with pytest.raises(ValueError, match="sum to 1"):
            mix([exponential(0.1), exponential(0.5)], [0.3, 0.3])
        with pytest.raises(ValueError, match="non-negative"):
            mix([exponential(0.1), exponential(0.5)], [1.5, -0.5])
        with pytest.raises(ValueError, match="at least two"):
            mix([exponential(0.1)], [1.0])
        with pytest.raises(ValueError, match="same length"):
            mix([exponential(0.1), exponential(0.5)], [1.0])

    def test_mix_sample_time_matches_survival_function(self):
        """Numeric-inverse sampling from a mixture reproduces its own survival curve."""
        curve = mix([exponential(0.1), exponential(0.5)], [0.4, 0.6])
        rng = np.random.default_rng(0)
        draws = curve.sample_time(rng, 50_000)
        for t in (1.0, 3.0, 8.0):
            empirical_survival = float(np.mean(draws > t))
            assert empirical_survival == pytest.approx(float(curve.survival(t)), abs=0.01)

    def test_splice_is_continuous_and_follows_each_segment(self):
        """`splice` matches `early` before the cutpoint and a rescaled `late` after."""
        early, late = exponential(0.1), exponential(0.4)
        cutpoint = 4.0
        curve = splice(early, late, cutpoint)
        assert curve.survival(2.0) == pytest.approx(float(early.survival(2.0)))
        early_at_cut = float(early.survival(cutpoint))
        late_at_cut = float(late.survival(cutpoint))
        expected_at_ten = early_at_cut * float(late.survival(10.0)) / late_at_cut
        assert curve.survival(10.0) == pytest.approx(expected_at_ten)
        # Continuity at the cutpoint: values just below and just above nearly agree.
        assert curve.survival(cutpoint - 1e-6) == pytest.approx(
            curve.survival(cutpoint + 1e-6), abs=1e-5
        )

    def test_numeric_inverse_matches_closed_form(self):
        """The generic bisection inverse matches a family's own closed-form inverse."""
        curve = weibull(SHAPE, SCALE)
        numeric_inverse = numeric_inverse_cumulative_hazard(curve.cumulative_hazard)
        target = np.array([0.1, 0.5, 1.0, 3.0])
        assert numeric_inverse(target) == pytest.approx(
            curve.inverse_cumulative_hazard(target), rel=1e-6
        )

    def test_apply_hazard_ratio_rejects_non_positive(self):
        with pytest.raises(ValueError, match="hazard_ratio"):
            apply_hazard_ratio(exponential(0.2), hazard_ratio=0.0)

    def test_apply_hazard_ratio_sample_time_matches_survival_function(self):
        """The hazard-ratio curve's own inverse cumulative hazard samples correctly."""
        curve = apply_hazard_ratio(exponential(0.2), hazard_ratio=0.5)
        rng = np.random.default_rng(2)
        draws = curve.sample_time(rng, 50_000)
        for t in (1.0, 3.0, 8.0):
            empirical_survival = float(np.mean(draws > t))
            assert empirical_survival == pytest.approx(float(curve.survival(t)), abs=0.01)

    def test_apply_acceleration_factor_rejects_non_positive(self):
        with pytest.raises(ValueError, match="acceleration_factor"):
            apply_acceleration_factor(weibull(SHAPE, SCALE), acceleration_factor=0.0)

    def test_apply_acceleration_factor_sample_time_matches_survival_function(self):
        """The rescaled curve's own inverse cumulative hazard samples correctly."""
        curve = apply_acceleration_factor(weibull(SHAPE, SCALE), acceleration_factor=2.0)
        rng = np.random.default_rng(5)
        draws = curve.sample_time(rng, 50_000)
        for t in (1.0, 3.0, 8.0):
            empirical_survival = float(np.mean(draws > t))
            assert empirical_survival == pytest.approx(float(curve.survival(t)), abs=0.01)

    def test_splice_rejects_non_positive_cutpoint(self):
        with pytest.raises(ValueError, match="cutpoint"):
            splice(exponential(0.1), exponential(0.4), cutpoint=0.0)


class TestToTransitionMatrix:
    def test_single_curve_rows_sum_to_one(self):
        transition = to_transition_matrix(weibull(SHAPE, SCALE), n_cycles=10)
        assert transition.shape == (10, 2, 2)
        assert transition.sum(axis=-1) == pytest.approx(1.0)

    def test_competing_risks_matches_exact_exponential_formula(self):
        """For constant hazards, the split matches the closed-form competing-exponential formula."""
        hazard_a, hazard_b = 0.1, 0.3
        transition = to_transition_matrix(
            [exponential(hazard_a), exponential(hazard_b)], n_cycles=1
        )
        total = hazard_a + hazard_b
        event_probability = 1.0 - np.exp(-total)
        assert transition[0, 0, 0] == pytest.approx(1.0 - event_probability)
        assert transition[0, 0, 1] == pytest.approx(event_probability * hazard_a / total)
        assert transition[0, 0, 2] == pytest.approx(event_probability * hazard_b / total)
        assert transition[0, 1, 1] == pytest.approx(1.0)
        assert transition[0, 2, 2] == pytest.approx(1.0)
        assert transition.sum(axis=-1) == pytest.approx(1.0)

    def test_rejects_non_positive_n_cycles(self):
        with pytest.raises(ValueError, match="n_cycles"):
            to_transition_matrix(weibull(SHAPE, SCALE), n_cycles=0)

    def test_rejects_empty_curve_list(self):
        with pytest.raises(ValueError, match="at least one"):
            to_transition_matrix([], n_cycles=5)

    def test_rejects_decreasing_cumulative_hazard(self):
        """A curve whose cumulative hazard is not non-decreasing is not a survival curve."""
        invalid_curve = SurvivalCurve(
            cumulative_hazard=lambda time: -time,
            inverse_cumulative_hazard=lambda value: value,
        )
        with pytest.raises(ValueError, match="non-decreasing"):
            to_transition_matrix(invalid_curve, n_cycles=3)


class TestFamilies:
    def test_families_reject_non_positive_parameters(self):
        with pytest.raises(ValueError, match="rate"):
            exponential(0.0)
        with pytest.raises(ValueError, match="shape"):
            weibull(-1.0, 6.0)
        with pytest.raises(ValueError, match="scale"):
            weibull(1.0, -6.0)
        with pytest.raises(ValueError, match="shape"):
            gompertz(-0.1, 0.01)
        with pytest.raises(ValueError, match="rate"):
            gompertz(0.1, -0.01)

    def test_gompertz_survival(self):
        curve = gompertz(shape=0.1, rate=0.01)
        assert curve.survival(10.0) == pytest.approx(0.84212, abs=1e-5)

    def test_gompertz_sample_time_matches_survival_function(self):
        curve = gompertz(shape=0.2, rate=0.02)
        rng = np.random.default_rng(1)
        draws = curve.sample_time(rng, 50_000)
        for t in (2.0, 5.0, 10.0):
            empirical_survival = float(np.mean(draws > t))
            assert empirical_survival == pytest.approx(float(curve.survival(t)), abs=0.01)

    def test_exponential_sample_time_matches_survival_function(self):
        curve = exponential(rate=0.3)
        rng = np.random.default_rng(4)
        draws = curve.sample_time(rng, 50_000)
        for t in (1.0, 3.0, 8.0):
            empirical_survival = float(np.mean(draws > t))
            assert empirical_survival == pytest.approx(float(curve.survival(t)), abs=0.01)

    def test_survival_curve_repr_shows_label(self):
        curve = weibull(SHAPE, SCALE)
        assert repr(curve) == f"SurvivalCurve({curve.label!r})"


lifelines = pytest.importorskip("lifelines")

from lifelines import WeibullFitter  # noqa: E402

from heormodel.survival import from_lifelines, sample_params  # noqa: E402

CENSOR = 12.0


def _fit_weibull_at(size: int, censor: float = CENSOR, seed: int = 20260714):
    rng = np.random.default_rng(seed)
    event_time = weibull(SHAPE, SCALE).sample_time(rng, size)
    observed = np.minimum(event_time, censor)
    observed_event = (event_time <= censor).astype(float)
    return WeibullFitter().fit(observed, event_observed=observed_event)


class TestLifelinesAdapter:
    def test_from_lifelines_matches_fitted_model(self):
        """The curve's survival function matches the fitted model's own values."""
        fit = _fit_weibull_at(2_000)
        curve = from_lifelines(fit)
        grid = np.array([1.0, 3.0, 6.0, 9.0])
        expected = fit.survival_function_at_times(grid).to_numpy()
        assert curve.survival(grid) == pytest.approx(expected, rel=1e-6)

    def test_from_lifelines_rejects_unfitted_object(self):
        with pytest.raises(TypeError, match="fitted survival model"):
            from_lifelines(object())

    def test_sample_params_columns_and_index(self):
        fit = _fit_weibull_at(300)
        draws = sample_params(fit, n=25, seed=0)
        assert draws.index.name == "iteration"
        assert list(draws.index) == list(range(25))
        assert list(draws.columns) == list(fit.params_.index)

    def test_sample_params_rejects_non_positive_n(self):
        fit = _fit_weibull_at(300)
        with pytest.raises(ValueError, match="n must be"):
            sample_params(fit, n=0, seed=0)

    def test_sample_params_recovers_fit_moments(self):
        """Sampling recovers the fitted mean and covariance as the draw count grows."""
        fit = _fit_weibull_at(300)
        draws = sample_params(fit, n=200_000, seed=0)
        assert draws.mean().to_numpy() == pytest.approx(fit.params_.to_numpy(), abs=5e-3)
        assert np.cov(draws.to_numpy().T) == pytest.approx(
            fit.variance_matrix_.to_numpy(), abs=5e-3
        )

    def test_parameter_recovery_converges(self):
        """The fit converges to the data-generating parameters as the sample grows."""
        fit_large = _fit_weibull_at(200_000)
        shape_large, scale_large = fit_large.params_["rho_"], fit_large.params_["lambda_"]
        assert shape_large == pytest.approx(SHAPE, abs=0.01)
        assert scale_large == pytest.approx(SCALE, abs=0.05)
        # Standard errors shrink like one over the square root of the sample size:
        # a 667-fold larger sample cuts them by about a factor of 26.
        fit_small = _fit_weibull_at(300)
        se_small = np.sqrt(np.diag(fit_small.variance_matrix_))
        se_large = np.sqrt(np.diag(fit_large.variance_matrix_))
        assert np.all(se_large < se_small / 15.0)

    def test_analytic_convergence_and_trial_size_interval(self):
        """The fitted and probabilistic life expectancies converge to 4.92709."""
        fit_large = _fit_weibull_at(200_000)
        fitted_dle = discounted_life_expectancy(from_lifelines(fit_large))
        assert fitted_dle == pytest.approx(ANALYTIC_DLE, abs=0.02)

        draws_large = sample_params(fit_large, n=2_000, seed=1)
        dle_large = draws_large.apply(
            lambda row: discounted_life_expectancy(weibull(row["rho_"], row["lambda_"])), axis=1
        )
        assert dle_large.mean() == pytest.approx(ANALYTIC_DLE, abs=0.02)
        assert dle_large.std() < 0.05  # spread shrinks toward zero at large sample size

        # At trial size the analytic value lies inside the 95% credible interval.
        fit_small = _fit_weibull_at(300)
        draws_small = sample_params(fit_small, n=2_000, seed=1)
        dle_small = draws_small.apply(
            lambda row: discounted_life_expectancy(weibull(row["rho_"], row["lambda_"])), axis=1
        )
        lower, upper = np.percentile(dle_small, [2.5, 97.5])
        assert lower < ANALYTIC_DLE < upper
