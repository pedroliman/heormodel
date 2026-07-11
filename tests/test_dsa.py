"""Tests for deterministic sensitivity analysis designs and reports."""

import numpy as np
import pandas as pd
import pytest

from heormodel.dsa import grid, one_at_a_time, one_way
from heormodel.models import Outcomes
from heormodel.report import heatmap_data, tornado_data
from heormodel.run import run_psa

BASE = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})


def linear_model(draws: pd.DataFrame) -> Outcomes:
    """A closed-form model: cost and effect are linear in the parameters.

    Strategy ``T`` costs ``10*a + 5*b`` and yields ``0.1*a + 0.2*c`` QALYs;
    strategy ``S`` is the fixed comparator at the origin.
    """
    idx = draws.index
    cost = pd.DataFrame(
        {"S": np.zeros(len(idx)), "T": 10.0 * draws["a"] + 5.0 * draws["b"]}, index=idx
    )
    effect = pd.DataFrame(
        {"S": np.zeros(len(idx)), "T": 0.1 * draws["a"] + 0.2 * draws["c"]}, index=idx
    )
    return Outcomes.from_wide(cost, effect)


class TestDesignShape:
    def test_one_way_row_count(self):
        design, descriptor = one_way(BASE, "a", [0.0, 0.5, 1.0, 2.0])
        assert len(design) == 4
        assert len(descriptor) == 4
        assert design.index.name == "iteration"

    def test_one_way_holds_others_at_base(self):
        design, _ = one_way(BASE, "a", [0.0, 5.0])
        assert (design["b"] == BASE["b"]).all()
        assert (design["c"] == BASE["c"]).all()
        assert design["a"].tolist() == [0.0, 5.0]

    def test_one_at_a_time_includes_base_once(self):
        design, descriptor = one_at_a_time(BASE, {"a": (0.0, 2.0), "b": (1.0, 3.0)})
        # base + 2 values for a + 2 values for b
        assert len(design) == 5
        assert descriptor["scenario"].iloc[0] == "(base)"
        base_row = design.iloc[0]
        assert base_row["a"] == BASE["a"] and base_row["b"] == BASE["b"]

    def test_one_at_a_time_varies_one_parameter_per_scenario(self):
        design, descriptor = one_at_a_time(BASE, {"a": (0.0, 2.0), "b": (1.0, 3.0)})
        for i in range(1, len(design)):
            varied = descriptor["parameter"].iloc[i]
            others = [p for p in BASE.index if p != varied]
            for other in others:
                assert design.iloc[i][other] == BASE[other]

    def test_grid_row_count_is_product_plus_base(self):
        design, _ = grid(BASE, {"a": [0.0, 1.0, 2.0], "b": [10.0, 20.0]})
        # 3 * 2 factorial + 1 base
        assert len(design) == 3 * 2 + 1

    def test_grid_unlisted_parameter_stays_at_base(self):
        design, _ = grid(BASE, {"a": [0.0, 1.0], "b": [10.0, 20.0]})
        assert (design["c"] == BASE["c"]).all()

    def test_grid_covers_every_combination(self):
        _, descriptor = grid(BASE, {"a": [0.0, 1.0], "b": [10.0, 20.0]})
        combos = set(zip(descriptor["a"].iloc[1:], descriptor["b"].iloc[1:], strict=True))
        assert combos == {(0.0, 10.0), (0.0, 20.0), (1.0, 10.0), (1.0, 20.0)}

    def test_unknown_parameter_raises(self):
        with pytest.raises(KeyError, match="not in the base case"):
            one_way(BASE, "missing", [1.0])
        with pytest.raises(KeyError):
            grid(BASE, {"missing": [1.0]})


class TestAnalyticSensitivity:
    def test_one_way_matches_linear_slope(self):
        # NMB_T at wtp w is w*(0.1 a + 0.2 c) - (10 a + 5 b), incremental over S.
        # Sweeping a, d(NMB)/da = 0.1 w - 10. Check outcome change equals that slope.
        values = [0.0, 1.0, 2.0, 3.0]
        design, _ = one_way(BASE, "a", values)
        outcomes = run_psa(linear_model, design).outcomes
        wtp = 50.0
        nb = wtp * outcomes.effects_wide()["T"] - outcomes.costs_wide()["T"]
        nb = nb.to_numpy()
        slope = 0.1 * wtp - 10.0
        expected = slope * (np.array(values) - values[0]) + nb[0]
        assert np.allclose(nb, expected)

    def test_grid_matches_bilinear_form(self):
        # Cost of T is 10 a + 5 b: exactly bilinear-separable, so the grid
        # reproduces the closed form at every node.
        design, descriptor = grid(BASE, {"a": [0.0, 1.0, 2.0], "b": [4.0, 8.0]})
        outcomes = run_psa(linear_model, design).outcomes
        cost = outcomes.costs_wide()["T"]
        hm = heatmap_data(cost, descriptor, x="a", y="b")
        for a in (0.0, 1.0, 2.0):
            for b in (4.0, 8.0):
                assert hm.loc[b, a] == pytest.approx(10.0 * a + 5.0 * b)


class TestTornadoFromDsa:
    def test_tornado_reads_swept_extremes(self):
        design, descriptor = one_at_a_time(BASE, {"a": (0.0, 4.0), "b": (0.0, 4.0)})
        outcomes = run_psa(linear_model, design).outcomes
        wtp = 50.0
        td = tornado_data(outcomes, (design, descriptor), wtp=wtp, strategy="T", comparator="S")
        assert set(td.index) == {"a", "b"}
        # d(NMB)/da = 0.1*50 - 10 = -5 per unit; span over [0,4] = 20.
        # d(NMB)/db = -5 per unit; span over [0,4] = 20. Equal spans.
        assert td.loc["a", "span"] == pytest.approx(20.0)
        assert td.loc["b", "span"] == pytest.approx(20.0)

    def test_grid_descriptor_rejected_by_tornado(self):
        design, descriptor = grid(BASE, {"a": [0.0, 1.0], "b": [1.0, 2.0]})
        outcomes = run_psa(linear_model, design).outcomes
        with pytest.raises(ValueError, match="one-way"):
            tornado_data(outcomes, (design, descriptor), wtp=50.0, strategy="T")
