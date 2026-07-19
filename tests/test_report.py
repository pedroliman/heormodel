"""Tests for plots and reproducibility scaffolding."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from heormodel.cea import ceac, ceaf
from heormodel.models import Outcomes
from heormodel.params import Gamma, Normal, ParameterSet
from heormodel.report import (
    PALETTE,
    RunRecord,
    capture_run,
    format_icer_table,
    intervention_colors,
    plot_ce_plane,
    plot_ceac,
    plot_frontier,
    plot_tornado,
    tornado_data,
)
from heormodel.run import SeedManager


@pytest.fixture
def psa() -> tuple[Outcomes, pd.DataFrame]:
    ps = ParameterSet({"c_b": Gamma.from_mean_se(100, 30), "e_b": Normal(0.5, 0.2)})
    draws = ps.sample(300, seed=4)
    idx = draws.index
    costs = pd.DataFrame({"A": np.zeros(300), "B": draws["c_b"]}, index=idx)
    effects = pd.DataFrame({"A": np.zeros(300), "B": draws["e_b"]}, index=idx)
    return Outcomes.from_wide(costs, effects), draws


class TestPlots:
    def test_ce_plane(self, psa):
        outcomes, _ = psa
        ax = plot_ce_plane(outcomes, wtp=500.0)
        assert isinstance(ax, Axes)
        assert "Cost-effectiveness plane" in ax.get_title()
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "B" in labels and "WTP = 500" in labels

    def test_ce_plane_density_fills_regions(self, psa):
        # With enough iterations the default draws filled highest-density regions
        # (a QuadContourSet), not a scatter of points.
        outcomes, _ = psa
        ax = plot_ce_plane(outcomes)
        kinds = {type(c).__name__ for c in ax.collections}
        assert "QuadContourSet" in kinds or any("Contour" in k for k in kinds)

    def test_ce_plane_scatter_kind(self, psa):
        outcomes, _ = psa
        ax = plot_ce_plane(outcomes, kind="scatter")
        assert any(type(c).__name__ == "PathCollection" for c in ax.collections)

    def test_ce_plane_falls_back_to_scatter_when_few_points(self):
        # Too few iterations to estimate a density: the default still draws.
        costs = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [1.0, 2.0, 3.0]})
        effects = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.1, 0.2, 0.3]})
        ax = plot_ce_plane(Outcomes.from_wide(costs, effects))
        assert any(type(c).__name__ == "PathCollection" for c in ax.collections)

    def test_ce_plane_rejects_unknown_kind(self, psa):
        outcomes, _ = psa
        with pytest.raises(ValueError, match="density"):
            plot_ce_plane(outcomes, kind="cloud")

    def test_ceac_with_frontier(self, psa):
        outcomes, _ = psa
        grid = np.linspace(0, 1000, 11)
        ax = plot_ceac(ceac(outcomes, grid), ceaf_df=ceaf(outcomes, grid))
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Frontier (CEAF)" in labels

    def test_frontier_plot(self, psa):
        outcomes, _ = psa
        assert isinstance(plot_frontier(outcomes), Axes)

    def test_tornado(self, psa):
        outcomes, draws = psa
        td = tornado_data(outcomes, draws, wtp=500.0, intervention="B", comparator="A")
        assert set(td.index) == {"c_b", "e_b"}
        # at wtp=500, sd(500*e) = 100 > sd(c) = 30: effect must rank first
        assert td.index[0] == "e_b"
        assert isinstance(plot_tornado(td), Axes)

    def test_tornado_requires_shared_index(self, psa):
        outcomes, draws = psa
        with pytest.raises(ValueError, match="iteration index"):
            tornado_data(outcomes, draws.iloc[:10], wtp=500.0)

    def test_intervention_colors_fixed_order_and_capped(self):
        colors = intervention_colors(["X", "Y"])
        assert colors["X"] == PALETTE[0] and colors["Y"] == PALETTE[1]
        with pytest.raises(ValueError, match="Other"):
            intervention_colors([f"s{i}" for i in range(9)])


class TestProvenance:
    def test_capture_run_records_everything(self, psa):
        outcomes, _ = psa
        ps = ParameterSet({"x": Normal(0, 1)})
        record = capture_run(seed=SeedManager(99), params=ps, outcomes=outcomes, note="t")
        assert record.seed_entropy == 99
        assert record.n_iterations == 300
        assert record.interventions == ["A", "B"]
        assert "Normal" in record.parameters["x"]
        assert "numpy" in record.versions

    def test_json_round_trip(self, tmp_path):
        record = capture_run(seed=1, note="round trip")
        path = tmp_path / "run.json"
        record.to_json(path)
        restored = RunRecord.from_json(path.read_text())
        assert restored == record

    def test_to_markdown_contents(self, psa):
        outcomes, _ = psa
        ps = ParameterSet({"x": Normal(0, 1)})
        report = capture_run(seed=5, params=ps, outcomes=outcomes).to_markdown()
        assert report.startswith("# Run report")
        assert "Root seed entropy:** 5" in report
        assert "| x |" in report

    def test_draw_sources_recorded_and_rendered(self):
        sources = {"beta": "ABC posterior", "u_healthy": "literature"}
        record = capture_run(seed=1, draw_sources=sources)
        assert record.draw_sources == sources
        report = record.to_markdown()
        assert "## Draw sources" in report
        assert "| beta | ABC posterior |" in report
        # round trip preserves the mapping
        assert RunRecord.from_json(record.to_json()).draw_sources == sources

    def test_draw_sources_absent_by_default(self, psa):
        outcomes, _ = psa
        report = capture_run(seed=5, outcomes=outcomes).to_markdown()
        assert "Draw sources" not in report


class TestFormatIcerTable:
    """The reading-oriented ICER table: rounded numbers and point (low, high)."""

    def test_intervals_written_point_low_high(self, psa):
        outcomes, _ = psa
        formatted = format_icer_table(outcomes)
        cell = formatted.loc["B", "ICER"]
        assert cell.count("(") == 1 and cell.endswith(")")
        assert "," in cell.split("(")[1]  # low and high separated by a comma

    def test_mean_table_has_no_parentheses(self):
        means = pd.DataFrame(
            {"cost": [0.0, 100.0, 400.0], "effect": [0.0, 0.5, 1.0]},
            index=["A", "B", "D"],
        )
        formatted = format_icer_table(means)
        assert formatted.loc["D", "ICER"] == "600"
        assert "(" not in formatted.loc["B", "Cost"]

    def test_sentence_case_spelled_out_columns(self, psa):
        outcomes, _ = psa
        formatted = format_icer_table(outcomes)
        assert list(formatted.columns) == [
            "Cost",
            "Effect",
            "Incremental cost",
            "Incremental effect",
            "ICER",
            "Status",
        ]
        assert formatted.index.name == "Intervention"
        # the cheapest frontier intervention has no incremental columns
        assert formatted.loc["A", "Incremental cost"] == ""
        assert formatted.loc["A", "ICER"] == ""

    def test_default_digits_round_costs_and_effects_differently(self):
        means = pd.DataFrame(
            {"cost": [0.0, 12345.678], "effect": [0.0, 1.23456]},
            index=["A", "B"],
        )
        formatted = format_icer_table(means)
        assert formatted.loc["B", "Cost"] == "12,346"  # whole units, separator
        assert formatted.loc["B", "Effect"] == "1.23"  # two decimals

    def test_digits_override_applies_everywhere(self):
        means = pd.DataFrame(
            {"cost": [0.0, 12345.678], "effect": [0.0, 1.23456]},
            index=["A", "B"],
        )
        formatted = format_icer_table(means, digits=1)
        assert formatted.loc["B", "Cost"] == "12,345.7"
        assert formatted.loc["B", "Effect"] == "1.2"

    def test_digits_mapping_sets_one_measure(self):
        means = pd.DataFrame(
            {"cost": [0.0, 12345.678], "effect": [0.0, 1.23456]},
            index=["A", "B"],
        )
        # the mapping still keys on the short measure name, not the display header
        formatted = format_icer_table(means, digits={"cost": 2})
        assert formatted.loc["B", "Cost"] == "12,345.68"
        assert formatted.loc["B", "Effect"] == "1.23"  # default kept for effect

    def test_interval_none_suppresses_bounds(self, psa):
        outcomes, _ = psa
        formatted = format_icer_table(outcomes, interval=None)
        assert "(" not in formatted.loc["B", "ICER"]
