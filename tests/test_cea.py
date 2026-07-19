"""Tests for cost-effectiveness analysis.

Includes the phase-1 validation check: a fully hand-verified incremental
analysis with strong and extended dominance, in the style of the standard
decision-analysis teaching examples.
"""

import numpy as np
import pandas as pd
import pytest

from heormodel.cea import (
    STATUS_D,
    STATUS_ED,
    STATUS_ND,
    ce_plane,
    ceac,
    ceaf,
    expected_loss,
    expected_nmb,
    frontier,
    icer_table,
    nhb,
    nmb,
)
from heormodel.models import Outcomes


@pytest.fixture
def reference_means() -> pd.DataFrame:
    """Five interventions with every dominance situation, verified by hand.

    Sorted by cost: A(0, 0), B(100, 0.5), E(250, 0.4), C(300, 0.6), D(400, 1.0).

    - E is strongly dominated by B (costs more, less effective).
    - C: ICER vs B = 200/0.1 = 2000; D: ICER vs C = 100/0.4 = 250 < 2000,
      so C is extendedly dominated.
    - Frontier: A -> B (ICER 200) -> D (ICER vs B = 300/0.5 = 600).
    """
    return pd.DataFrame(
        {
            "cost": [0.0, 100.0, 300.0, 400.0, 250.0],
            "effect": [0.0, 0.5, 0.6, 1.0, 0.4],
        },
        index=["A", "B", "C", "D", "E"],
    )


class TestValidationIncrementalAnalysis:
    """Phase-1 acceptance check: dominance, extended dominance, and ICERs."""

    def test_statuses_match_hand_calculation(self, reference_means):
        table = icer_table(reference_means)
        assert table.loc["A", "status"] == STATUS_ND
        assert table.loc["B", "status"] == STATUS_ND
        assert table.loc["C", "status"] == STATUS_ED
        assert table.loc["D", "status"] == STATUS_ND
        assert table.loc["E", "status"] == STATUS_D

    def test_icers_match_hand_calculation(self, reference_means):
        table = icer_table(reference_means)
        assert np.isnan(table.loc["A", "icer"])
        assert table.loc["B", "icer"] == pytest.approx(200.0)
        assert table.loc["D", "icer"] == pytest.approx(600.0)
        assert np.isnan(table.loc["C", "icer"])
        assert np.isnan(table.loc["E", "icer"])

    def test_frontier_order(self, reference_means):
        assert frontier(reference_means) == ["A", "B", "D"]

    def test_incrementals_computed_between_frontier_neighbours(self, reference_means):
        table = icer_table(reference_means)
        assert table.loc["D", "inc_cost"] == pytest.approx(300.0)
        assert table.loc["D", "inc_effect"] == pytest.approx(0.5)

    def test_table_sorted_by_cost(self, reference_means):
        assert list(icer_table(reference_means).index) == ["A", "B", "E", "C", "D"]


class TestIcerEdgeCases:
    def test_duplicate_interventions_keep_first(self):
        means = pd.DataFrame({"cost": [10.0, 10.0], "effect": [1.0, 1.0]}, index=["A", "A2"])
        table = icer_table(means)
        assert table.loc["A", "status"] == STATUS_ND
        assert table.loc["A2", "status"] == STATUS_D

    def test_two_interventions_no_extended_dominance_possible(self):
        means = pd.DataFrame({"cost": [0.0, 50.0], "effect": [0.0, 1.0]}, index=["A", "B"])
        table = icer_table(means)
        assert list(table["status"]) == [STATUS_ND, STATUS_ND]
        assert table.loc["B", "icer"] == pytest.approx(50.0)

    def test_chain_of_extended_dominance(self):
        # B and C both lie above the A-D segment and must both be removed
        means = pd.DataFrame(
            {"cost": [0.0, 100.0, 200.0, 300.0], "effect": [0.0, 0.1, 0.2, 1.0]},
            index=["A", "B", "C", "D"],
        )
        table = icer_table(means)
        assert table.loc["B", "status"] == STATUS_ED
        assert table.loc["C", "status"] == STATUS_ED
        assert table.loc["D", "icer"] == pytest.approx(300.0)

    def test_accepts_outcomes_object(self):
        c = pd.DataFrame({"A": [0.0, 0.0], "B": [100.0, 100.0]})
        e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, 1.0]})
        table = icer_table(Outcomes.from_wide(c, e))
        assert table.loc["B", "icer"] == pytest.approx(100.0)


class TestIcerUncertaintyIntervals:
    """Intervals from a probabilistic PSA, with paired incremental measures."""

    @pytest.fixture
    def psa_outcomes(self) -> Outcomes:
        rng = np.random.default_rng(0)
        n = 4000
        costs = pd.DataFrame(
            {
                "A": rng.normal(0.0, 1.0, n),
                "B": rng.normal(100.0, 20.0, n),
                "D": rng.normal(400.0, 60.0, n),
            }
        )
        effects = pd.DataFrame(
            {
                "A": rng.normal(0.0, 0.01, n),
                "B": rng.normal(0.5, 0.05, n),
                "D": rng.normal(1.0, 0.08, n),
            }
        )
        return Outcomes.from_wide(costs, effects, effect="qaly")

    def test_interval_columns_present_and_ordered(self, psa_outcomes):
        table = icer_table(psa_outcomes)
        expected = []
        for estimate in ("cost", "effect", "inc_cost", "inc_effect", "icer"):
            expected += [estimate, f"{estimate}_lo", f"{estimate}_hi"]
        expected.append("status")
        assert list(table.columns) == expected

    def test_no_intervals_for_mean_table(self, reference_means):
        assert "cost_lo" not in icer_table(reference_means).columns

    def test_no_intervals_when_suppressed(self, psa_outcomes):
        assert "cost_lo" not in icer_table(psa_outcomes, interval=None).columns

    def test_point_estimates_unchanged_by_intervals(self, psa_outcomes):
        with_ci = icer_table(psa_outcomes)
        without = icer_table(psa_outcomes, interval=None)
        for col in ("cost", "effect", "inc_cost", "inc_effect", "icer"):
            np.testing.assert_allclose(
                with_ci[col].to_numpy(dtype=float), without[col].to_numpy(dtype=float)
            )

    def test_incremental_interval_is_paired_not_marginal(self, psa_outcomes):
        # The paired difference D minus B per draw, then its percentiles, is
        # narrower than differencing the two strategies' separate intervals,
        # because the two costs move together across draws under common draws.
        table = icer_table(psa_outcomes)
        costs = psa_outcomes.costs_wide()
        paired = (costs["D"] - costs["B"]).to_numpy()
        lo, hi = np.percentile(paired, [2.5, 97.5])
        assert table.loc["D", "inc_cost_lo"] == pytest.approx(lo)
        assert table.loc["D", "inc_cost_hi"] == pytest.approx(hi)

    def test_icer_interval_from_paired_ratio(self, psa_outcomes):
        table = icer_table(psa_outcomes)
        costs = psa_outcomes.costs_wide()
        effects = psa_outcomes.effects_wide("qaly")
        ratio = (costs["D"] - costs["B"]).to_numpy() / (effects["D"] - effects["B"]).to_numpy()
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        assert table.loc["D", "icer_lo"] == pytest.approx(lo)
        assert table.loc["D", "icer_hi"] == pytest.approx(hi)

    def test_interval_brackets_point_estimate(self, psa_outcomes):
        table = icer_table(psa_outcomes)
        for estimate in ("cost", "effect", "inc_cost", "inc_effect", "icer"):
            point = table[estimate]
            lo, hi = table[f"{estimate}_lo"], table[f"{estimate}_hi"]
            valid = point.notna()
            assert (lo[valid] <= point[valid] + 1e-9).all()
            assert (point[valid] <= hi[valid] + 1e-9).all()

    def test_cheapest_frontier_intervention_has_no_incremental_interval(self, psa_outcomes):
        table = icer_table(psa_outcomes)
        assert np.isnan(table.loc["A", "inc_cost_lo"])
        assert np.isnan(table.loc["A", "icer_hi"])

    def test_interval_level_widens_the_band(self, psa_outcomes):
        wide = icer_table(psa_outcomes, interval=0.99)
        narrow = icer_table(psa_outcomes, interval=0.5)
        assert wide.loc["B", "cost_lo"] < narrow.loc["B", "cost_lo"]
        assert wide.loc["B", "cost_hi"] > narrow.loc["B", "cost_hi"]

    def test_rejects_out_of_range_interval(self, psa_outcomes):
        with pytest.raises(ValueError):
            icer_table(psa_outcomes, interval=1.5)

    def test_near_efficient_strategy_keeps_a_fixed_comparator(self):
        # The frontier is settled once on the means, then every draw compares an
        # intervention against the same cheaper frontier neighbour. A strategy
        # that is nearly efficient (extendedly dominated in expectation, but on
        # the frontier in many single draws) must not become a comparator: if the
        # frontier were recomputed per draw, the comparison would keep changing
        # and the paired incremental distribution would be meaningless.
        rng = np.random.default_rng(7)
        n = 4000
        # Means A(0, 0) and B(200, 0.70) are on the frontier; N(100, 0.30) sits
        # just above the A-B line, so it is extendedly dominated on average.
        costs = pd.DataFrame(
            {
                "A": rng.normal(0.0, 15.0, n),
                "N": rng.normal(100.0, 15.0, n),
                "B": rng.normal(200.0, 15.0, n),
            }
        )
        effects = pd.DataFrame(
            {
                "A": rng.normal(0.0, 0.12, n),
                "N": rng.normal(0.30, 0.12, n),
                "B": rng.normal(0.70, 0.12, n),
            }
        )
        out = Outcomes.from_wide(costs, effects, effect="qaly")
        table = icer_table(out)

        # N is off the frontier in expectation and carries no incremental interval.
        assert table.loc["N", "status"] == STATUS_ED
        assert np.isnan(table.loc["N", "inc_cost_lo"])
        assert np.isnan(table.loc["N", "icer_hi"])

        # N would be efficient in a sizeable share of single draws, so a per-draw
        # frontier would sometimes compare B against N. Confirm the setup exercises
        # that hazard before asserting the fix holds.
        slope = (costs["B"] - costs["A"]) / (effects["B"] - effects["A"])
        n_below_chord = (costs["N"] < costs["A"] + (effects["N"] - effects["A"]) * slope).mean()
        assert n_below_chord > 0.2

        # B's paired incremental uses A as its comparator in every draw, so its
        # interval equals the percentiles of cost_B minus cost_A, not a mixture
        # that sometimes differences against N.
        lo, hi = np.percentile((costs["B"] - costs["A"]).to_numpy(), [2.5, 97.5])
        assert table.loc["B", "inc_cost_lo"] == pytest.approx(lo)
        assert table.loc["B", "inc_cost_hi"] == pytest.approx(hi)


@pytest.fixture
def simple_outcomes() -> Outcomes:
    costs = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0], "B": [50.0, 50.0, 50.0, 50.0]})
    effects = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0], "B": [1.0, 1.0, 1.0, -1.0]})
    return Outcomes.from_wide(costs, effects)


class TestNetBenefit:
    def test_nmb_values(self, simple_outcomes):
        nb = nmb(simple_outcomes, wtp=100.0)
        assert nb.loc[0, "B"] == pytest.approx(50.0)
        assert nb.loc[3, "B"] == pytest.approx(-150.0)
        assert (nb["A"] == 0).all()

    def test_nhb_values(self, simple_outcomes):
        nh = nhb(simple_outcomes, wtp=100.0)
        assert nh.loc[0, "B"] == pytest.approx(0.5)

    def test_nhb_requires_positive_wtp(self, simple_outcomes):
        with pytest.raises(ValueError):
            nhb(simple_outcomes, wtp=0.0)

    def test_expected_nmb(self, simple_outcomes):
        e_nb = expected_nmb(simple_outcomes, wtp=100.0)
        assert e_nb["B"] == pytest.approx(0.0)  # (50 + 50 + 50 - 150) / 4


class TestCeacCeaf:
    def test_ceac_probabilities(self, simple_outcomes):
        curve = ceac(simple_outcomes, wtp=[0.0, 100.0])
        assert curve.loc[0.0, "A"] == 1.0  # B costs more, effect irrelevant at wtp=0
        assert curve.loc[100.0, "B"] == 0.75
        np.testing.assert_allclose(curve.sum(axis=1), 1.0)

    def test_ceaf_reports_max_expected_nmb_intervention(self, simple_outcomes):
        # At wtp=200: E[NMB_B] = (150*3 - 250)/4 = 50 > 0 so B is optimal,
        # though it wins in only 75% of iterations.
        f = ceaf(simple_outcomes, wtp=[200.0])
        assert f.loc[200.0, "intervention"] == "B"
        assert f.loc[200.0, "prob"] == 0.75

    def test_ce_plane_incrementals(self, simple_outcomes):
        plane = ce_plane(simple_outcomes)
        assert set(plane["intervention"]) == {"B"}
        assert plane["inc_cost"].tolist() == [50.0] * 4

    def test_ce_plane_unknown_comparator(self, simple_outcomes):
        with pytest.raises(KeyError):
            ce_plane(simple_outcomes, comparator="Z")

    def test_ce_plane_defaults_to_outcomes_comparator(self, simple_outcomes):
        flagged = Outcomes(simple_outcomes.data, effect=simple_outcomes.effect, comparator="B")
        plane = ce_plane(flagged)
        assert set(plane["intervention"]) == {"A"}

    def test_ce_plane_argument_overrides_outcomes_comparator(self, simple_outcomes):
        flagged = Outcomes(simple_outcomes.data, effect=simple_outcomes.effect, comparator="B")
        plane = ce_plane(flagged, comparator="A")
        assert set(plane["intervention"]) == {"B"}


class TestExpectedLoss:
    def test_hand_computed_losses(self, simple_outcomes):
        # At wtp=100: NMB_A = 0 in every iteration; NMB_B = 50, 50, 50, -150.
        losses = expected_loss(simple_outcomes, wtp=[100.0])
        assert losses.loc[100.0, "A"] == pytest.approx((50 * 3 + 0) / 4)
        assert losses.loc[100.0, "B"] == pytest.approx(150 / 4)

    def test_minimum_expected_loss_equals_evpi(self, simple_outcomes):
        from heormodel.voi import evpi

        grid = [0.0, 50.0, 100.0, 200.0, 400.0]
        losses = expected_loss(simple_outcomes, wtp=grid)
        pd.testing.assert_series_equal(
            losses.min(axis=1), evpi(simple_outcomes, grid), check_names=False
        )

    def test_zero_loss_without_uncertainty(self):
        costs = pd.DataFrame({"A": [0.0, 0.0], "B": [10.0, 10.0]})
        effects = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, 1.0]})
        losses = expected_loss(Outcomes.from_wide(costs, effects), wtp=[100.0])
        assert losses.loc[100.0, "B"] == 0.0
        assert losses.loc[100.0, "A"] == pytest.approx(90.0)
