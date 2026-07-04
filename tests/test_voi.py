"""Tests for value-of-information analysis.

Includes the phase-1 validation check: a two-strategy model with Gaussian
costs and effects, for which EVPI, EVPPI, and EVSI have closed-form values
via the unit normal loss integral (the standard analytic benchmark for
regression-based VoI estimators, cf. Strong, Oakley & Brennan 2014). The
estimators must agree with the analytic values within Monte Carlo error.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from heval.models import Outcomes
from heval.params import mix_draws
from heval.voi import (
    evpi,
    evppi,
    evppi_ranking,
    evsi_importance_sampling,
    evsi_moment_matching,
    evsi_regression,
    simulate_summaries,
)

WTP = 1000.0
MU_E, SD_E = 0.1, 0.5
MU_C, SD_C = 60.0, 300.0
TAU = 0.5  # sd of the study summary around the true effect
N = 80_000


def emax0(m: float, s: float) -> float:
    """E[max(0, X)] for X ~ Normal(m, s^2)."""
    return m * norm.cdf(m / s) + s * norm.pdf(m / s)


def analytic_voi(s_conditional: float) -> float:
    """VoI when the conditional mean incremental NB is Normal(m, s^2)."""
    m = WTP * MU_E - MU_C
    return emax0(m, s_conditional) - max(0.0, m)


@pytest.fixture(scope="module")
def gaussian_model():
    rng = np.random.default_rng(2026)
    e = rng.normal(MU_E, SD_E, N)
    c = rng.normal(MU_C, SD_C, N)
    idx = pd.RangeIndex(N, name="iteration")
    draws = pd.DataFrame({"e_b": e, "c_b": c}, index=idx)
    costs = pd.DataFrame({"A": np.zeros(N), "B": c}, index=idx)
    effects = pd.DataFrame({"A": np.zeros(N), "B": e}, index=idx)
    return Outcomes.from_wide(costs, effects), draws


class TestValidationAnalyticGaussian:
    """Phase-1 acceptance check: EVPI/EVPPI vs closed-form values."""

    def test_evpi_matches_analytic(self, gaussian_model):
        outcomes, _ = gaussian_model
        s_total = np.hypot(WTP * SD_E, SD_C)
        assert evpi(outcomes, WTP) == pytest.approx(analytic_voi(s_total), rel=0.05)

    def test_evppi_effect_matches_analytic(self, gaussian_model):
        outcomes, draws = gaussian_model
        expected = analytic_voi(WTP * SD_E)
        assert evppi(outcomes, draws, "e_b", WTP) == pytest.approx(expected, rel=0.05)

    def test_evppi_cost_matches_analytic(self, gaussian_model):
        outcomes, draws = gaussian_model
        expected = analytic_voi(SD_C)
        assert evppi(outcomes, draws, "c_b", WTP) == pytest.approx(expected, rel=0.05)

    def test_joint_evppi_of_all_parameters_equals_evpi(self, gaussian_model):
        outcomes, draws = gaussian_model
        joint = evppi(outcomes, draws, ["e_b", "c_b"], WTP)
        assert joint == pytest.approx(evpi(outcomes, WTP), rel=0.10)

    def test_evsi_matches_analytic(self, gaussian_model):
        outcomes, draws = gaussian_model
        rng = np.random.default_rng(7)
        summaries = pd.DataFrame(
            {"xbar": draws["e_b"] + rng.normal(0.0, TAU, N)}, index=draws.index
        )
        s_evsi = WTP * SD_E**2 / np.hypot(SD_E, TAU)
        expected = analytic_voi(s_evsi)
        assert evsi_regression(outcomes, summaries, WTP) == pytest.approx(expected, rel=0.05)


class TestCalibrationWorkflowVoi:
    """Acceptance check for the calibration workflow: VoI on a calibrated
    parameter survives ``mix_draws``.

    One Gaussian column stands in for a calibrated posterior, another for a
    literature draw. After mixing them into one PSA matrix, single-parameter
    EVPPI must still recover the closed-form value, confirming that mixing
    keeps the parameter/outcome linkage VoI depends on.
    """

    def test_evppi_of_calibrated_parameter_recovered_after_mixing(self):
        calibrated = pd.DataFrame(
            {"e_b": np.random.default_rng(1).normal(MU_E, SD_E, N)},
            index=pd.RangeIndex(N, name="iteration"),
        )
        literature = pd.DataFrame(
            {"c_b": np.random.default_rng(2).normal(MU_C, SD_C, N)},
            index=pd.RangeIndex(N, name="iteration"),
        )
        draws = mix_draws(calibrated, literature, n=N, seed=0)
        costs = pd.DataFrame({"A": np.zeros(N), "B": draws["c_b"]}, index=draws.index)
        effects = pd.DataFrame({"A": np.zeros(N), "B": draws["e_b"]}, index=draws.index)
        outcomes = Outcomes.from_wide(costs, effects)

        assert evppi(outcomes, draws, "e_b", WTP) == pytest.approx(
            analytic_voi(WTP * SD_E), rel=0.05
        )
        assert evppi(outcomes, draws, "c_b", WTP) == pytest.approx(analytic_voi(SD_C), rel=0.05)
        joint = evppi(outcomes, draws, ["e_b", "c_b"], WTP)
        assert joint == pytest.approx(evpi(outcomes, WTP), rel=0.10)


class TestVoiProperties:
    def test_information_ordering(self, gaussian_model):
        """EVSI <= EVPPI(param) <= EVPI, up to estimator noise."""
        outcomes, draws = gaussian_model
        rng = np.random.default_rng(3)
        summaries = pd.DataFrame(
            {"xbar": draws["e_b"] + rng.normal(0.0, TAU, N)}, index=draws.index
        )
        v_evsi = evsi_regression(outcomes, summaries, WTP)
        v_evppi = evppi(outcomes, draws, "e_b", WTP)
        v_evpi = evpi(outcomes, WTP)
        assert v_evsi <= v_evppi * 1.02
        assert v_evppi <= v_evpi * 1.02

    def test_evpi_zero_without_uncertainty(self):
        c = pd.DataFrame({"A": [0.0, 0.0], "B": [10.0, 10.0]})
        e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, 1.0]})
        assert evpi(Outcomes.from_wide(c, e), wtp=100.0) == 0.0

    def test_evpi_grid_returns_series(self, gaussian_model):
        outcomes, _ = gaussian_model
        series = evpi(outcomes, [0.0, WTP])
        assert isinstance(series, pd.Series)
        assert series.index.name == "wtp"
        assert (series >= 0).all()

    def test_evppi_ranking_orders_parameters(self, gaussian_model):
        outcomes, draws = gaussian_model
        ranking = evppi_ranking(outcomes, draws, WTP)
        assert list(ranking.index) == ["e_b", "c_b"]  # effect uncertainty dominates

    def test_gp_method_smoke(self, gaussian_model):
        outcomes, draws = gaussian_model
        small = draws.iloc[:2000]
        sub = Outcomes(outcomes.data[outcomes.data.index.get_level_values(1) < 2000])
        value = evppi(sub, small, "e_b", WTP, method="gp", seed=0)
        assert value == pytest.approx(analytic_voi(WTP * SD_E), rel=0.25)


class TestVoiGuards:
    def test_evppi_requires_shared_iteration_index(self, gaussian_model):
        outcomes, draws = gaussian_model
        with pytest.raises(ValueError, match="iteration index"):
            evppi(outcomes, draws.iloc[:100], "e_b", WTP)

    def test_evppi_unknown_parameter(self, gaussian_model):
        outcomes, draws = gaussian_model
        with pytest.raises(KeyError):
            evppi(outcomes, draws, "nope", WTP)

    def test_unknown_metamodel_method(self, gaussian_model):
        outcomes, draws = gaussian_model
        with pytest.raises(ValueError, match="metamodel"):
            evppi(outcomes, draws, "e_b", WTP, method="cubist")

    def test_evsi_stubs_raise(self):
        with pytest.raises(NotImplementedError):
            evsi_moment_matching()
        with pytest.raises(NotImplementedError):
            evsi_importance_sampling()

    def test_simulate_summaries_aligns_index(self):
        draws = pd.DataFrame({"p": [0.1, 0.9]}, index=pd.RangeIndex(2, name="iteration"))

        def study(row: pd.Series, rng: np.random.Generator) -> dict[str, float]:
            return {"x": float(rng.binomial(50, row["p"]))}

        summaries = simulate_summaries(draws, study, seed=1)
        assert summaries.index.equals(draws.index)
        assert summaries.loc[1, "x"] > summaries.loc[0, "x"]
