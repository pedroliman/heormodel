"""Tests for the outcome schema, the run loop, and bring-your-own-outputs."""

import numpy as np
import pandas as pd
import pytest

from heormodel.models import (
    ModelEngine,
    Outcomes,
)
from heormodel.params import Normal, ParameterSet, Uniform
from heormodel.run import SeedManager, as_outcomes, run_psa, running_means


def tidy_table(n: int = 4) -> pd.DataFrame:
    rows = []
    for s, base in (("A", 100.0), ("B", 200.0)):
        for i in range(n):
            row = {"intervention": s, "iteration": i, "cost": base + i, "qaly": 1.0 + 0.1 * i}
            rows.append(row)
    return pd.DataFrame(rows)


class TestOutcomes:
    def test_from_tidy_round_trip(self):
        out = Outcomes.from_tidy(tidy_table())
        assert out.interventions == ["A", "B"]
        assert out.n_iterations == 4
        assert out.costs_wide().loc[2, "B"] == 202.0

    def test_from_wide(self):
        c = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        e = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]})
        out = Outcomes.from_wide(c, e)
        pd.testing.assert_frame_equal(out.costs_wide(), c, check_names=False)
        pd.testing.assert_frame_equal(out.effects_wide(), e, check_names=False)

    def test_unbalanced_panel_rejected(self):
        df = tidy_table().iloc[:-1]  # drop one (intervention, iteration) row
        with pytest.raises(ValueError, match="Unbalanced"):
            Outcomes.from_tidy(df)

    def test_duplicate_rows_rejected(self):
        df = pd.concat([tidy_table(), tidy_table().iloc[[0]]])
        with pytest.raises(ValueError, match="Duplicate"):
            Outcomes.from_tidy(df)

    def test_nan_rejected(self):
        df = tidy_table()
        df.loc[0, "cost"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            Outcomes.from_tidy(df)

    def test_missing_columns_rejected(self):
        with pytest.raises(ValueError, match="Missing columns"):
            Outcomes.from_tidy(tidy_table().drop(columns="qaly"))

    def test_components_carried_along(self):
        df = tidy_table()
        df["cost_drug"] = 5.0
        out = Outcomes.from_tidy(df)
        assert out.components == ["cost_drug"]

    def test_select_subsets_interventions(self):
        out = Outcomes.from_tidy(tidy_table())
        assert out.select(["B"]).interventions == ["B"]
        with pytest.raises(KeyError):
            out.select(["Z"])

    def test_summary_means(self):
        out = Outcomes.from_tidy(tidy_table())
        assert out.summary().loc["A", "cost"] == pytest.approx(101.5)

    def test_comparator_defaults_to_none(self):
        assert Outcomes.from_tidy(tidy_table()).comparator is None

    def test_comparator_is_carried_when_given(self):
        out = Outcomes.from_tidy(tidy_table(), comparator="A")
        assert out.comparator == "A"

    def test_unknown_comparator_rejected(self):
        with pytest.raises(KeyError, match="Unknown comparator"):
            Outcomes.from_tidy(tidy_table(), comparator="Z")

    def test_select_drops_comparator_not_in_subset(self):
        out = Outcomes.from_tidy(tidy_table(), comparator="A")
        assert out.select(["B"]).comparator is None
        assert out.select(["A", "B"]).comparator == "A"


class TestByoOutputs:
    def test_as_outcomes_accepts_dataframe_and_custom_columns(self):
        df = tidy_table().rename(
            columns={"intervention": "arm", "iteration": "run", "cost": "tc", "qaly": "qalys"}
        )
        out = as_outcomes(df, intervention="arm", iteration="run", cost="tc", effect="qalys")
        assert out.interventions == ["A", "B"]
        assert out.effect == "qalys"

    def test_as_outcomes_reads_csv(self, tmp_path):
        path = tmp_path / "psa.csv"
        tidy_table().to_csv(path, index=False)
        assert as_outcomes(path).n_iterations == 4

    def test_as_outcomes_passes_through_outcomes(self):
        out = Outcomes.from_tidy(tidy_table())
        assert as_outcomes(out) is out


def dummy_model(draws: pd.DataFrame) -> Outcomes:
    costs = pd.DataFrame({"A": draws["c"], "B": draws["c"] * 2}, index=draws.index)
    effects = pd.DataFrame({"A": draws["e"], "B": draws["e"] + 0.5}, index=draws.index)
    return Outcomes.from_wide(costs, effects)


class DummyEngine:
    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        return dummy_model(draws)


class TestRunPsa:
    def setup_method(self):
        ps = ParameterSet({"c": Uniform(100, 200), "e": Normal(1.0, 0.1)})
        self.draws = ps.sample(40, seed=9)

    def test_callable_model_preserves_iteration_index(self):
        out = run_psa(dummy_model, self.draws).outcomes
        assert out.iterations.equals(self.draws.index)

    def test_engine_object_satisfies_protocol(self):
        assert isinstance(DummyEngine(), ModelEngine)
        out = run_psa(DummyEngine(), self.draws, sequential=True).outcomes
        assert out.interventions == ["A", "B"]

    def test_parallel_matches_serial(self):
        serial = run_psa(dummy_model, self.draws, sequential=True).outcomes
        parallel = run_psa(dummy_model, self.draws, n_jobs=2).outcomes
        pd.testing.assert_frame_equal(serial.data, parallel.data)

    def test_contract_violation_detected(self):
        def bad_model(draws: pd.DataFrame) -> Outcomes:
            return dummy_model(draws.reset_index(drop=True).iloc[:-1])

        with pytest.raises(ValueError, match="contract"):
            run_psa(bad_model, self.draws)

    def test_empty_draws_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            run_psa(dummy_model, self.draws.iloc[:0])


class TestSeedManager:
    def test_spawned_streams_reproducible_and_distinct(self):
        a = SeedManager(123).spawn(2)
        b = SeedManager(123).spawn(2)
        assert a[0].integers(1_000_000) == b[0].integers(1_000_000)
        x = SeedManager(123).spawn(2)
        assert x[0].integers(1_000_000) != x[1].integers(1_000_000)

    def test_entropy_recorded(self):
        assert SeedManager(42).entropy == 42
        assert isinstance(SeedManager().entropy, int)


class TestDiagnostics:
    def test_running_means_converge_to_mean(self):
        out = Outcomes.from_tidy(tidy_table())
        trace = running_means(out)
        assert trace["A"].iloc[-1] == pytest.approx(out.summary().loc["A", "cost"])
        assert trace["A"].iloc[0] == out.costs_wide()["A"].iloc[0]
