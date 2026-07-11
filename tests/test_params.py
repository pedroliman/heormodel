"""Tests for distribution specs and correlated PSA sampling."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from heormodel.models import Outcomes
from heormodel.params import (
    Beta,
    Dirichlet,
    Fixed,
    Gamma,
    LogNormal,
    Normal,
    ParameterSet,
    Uniform,
    mix_draws,
    read_draws,
    resample_posterior,
    single_draw,
)
from heormodel.run import run_psa


def _two_strategy_model(draws: pd.DataFrame) -> Outcomes:
    costs = pd.DataFrame({"A": draws["c"], "B": draws["c"] + 10.0}, index=draws.index)
    effects = pd.DataFrame({"A": draws["c"] * 0.0, "B": draws["c"] * 0.0 + 0.1}, index=draws.index)
    return Outcomes.from_wide(costs, effects)


class TestMethodOfMoments:
    def test_beta_matches_mean_and_se(self):
        d = Beta.from_mean_se(0.3, 0.05)
        assert d.mean() == pytest.approx(0.3)
        assert d.sd() == pytest.approx(0.05)

    def test_gamma_matches_mean_and_se(self):
        d = Gamma.from_mean_se(1200.0, 250.0)
        assert d.mean() == pytest.approx(1200.0)
        assert d.sd() == pytest.approx(250.0)

    def test_lognormal_matches_mean_and_se(self):
        d = LogNormal.from_mean_se(2.5, 0.8)
        assert d.mean() == pytest.approx(2.5)
        assert d.sd() == pytest.approx(0.8)

    def test_beta_rejects_impossible_se(self):
        with pytest.raises(ValueError, match="se too large"):
            Beta.from_mean_se(0.5, 0.6)

    def test_beta_rejects_mean_outside_unit_interval(self):
        with pytest.raises(ValueError):
            Beta.from_mean_se(1.2, 0.1)


class TestDistributions:
    def test_sampling_is_reproducible(self):
        d = Gamma.from_mean_se(100.0, 20.0)
        assert np.array_equal(d.sample(10, rng=5), d.sample(10, rng=5))

    def test_fixed_is_constant(self):
        assert np.all(Fixed(2.5).sample(4) == 2.5)
        assert Fixed(2.5).sd() == 0.0

    def test_dirichlet_mean(self):
        d = Dirichlet((8.0, 1.0, 1.0))
        np.testing.assert_allclose(d.mean(), [0.8, 0.1, 0.1])

    def test_dirichlet_component_labels(self):
        d = Dirichlet((1.0, 2.0), names=("a", "b"))
        assert d.component_labels("p") == ["p[a]", "p[b]"]

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            Gamma(-1.0, 2.0)
        with pytest.raises(ValueError):
            Normal(0.0, 0.0)
        with pytest.raises(ValueError):
            Uniform(1.0, 1.0)
        with pytest.raises(ValueError):
            Dirichlet((1.0,))


class TestParameterSet:
    def test_draw_matrix_shape_and_index(self):
        ps = ParameterSet({"a": Normal(0, 1), "b": Uniform(0, 1)})
        draws = ps.sample(100, seed=1)
        assert draws.shape == (100, 2)
        assert draws.index.name == "iteration"
        assert list(draws.columns) == ["a", "b"]

    def test_reproducible_with_same_seed(self):
        ps = ParameterSet({"a": Normal(0, 1)})
        pd.testing.assert_frame_equal(ps.sample(50, seed=3), ps.sample(50, seed=3))

    def test_marginals_are_exact_under_correlation(self):
        ps = ParameterSet(
            {"p": Beta.from_mean_se(0.2, 0.04), "c": Gamma.from_mean_se(500, 80)},
            correlation={("p", "c"): 0.6},
        )
        draws = ps.sample(50_000, seed=7)
        assert draws["p"].mean() == pytest.approx(0.2, abs=0.005)
        assert draws["p"].std() == pytest.approx(0.04, abs=0.005)
        assert draws["c"].mean() == pytest.approx(500, rel=0.02)
        # Kolmogorov-Smirnov against the target marginal
        target = stats.gamma((500 / 80) ** 2, scale=80**2 / 500)
        ks = stats.kstest(draws["c"], target.cdf)
        assert ks.pvalue > 0.01

    def test_spearman_correlation_matches_target(self):
        ps = ParameterSet(
            {"a": Normal(0, 1), "b": Gamma.from_mean_se(10, 3)},
            correlation={("a", "b"): 0.5},
        )
        draws = ps.sample(50_000, seed=11)
        rho = stats.spearmanr(draws["a"], draws["b"]).statistic
        assert rho == pytest.approx(0.5, abs=0.02)

    def test_dirichlet_expands_and_sums_to_one(self):
        ps = ParameterSet({"p": Dirichlet((70, 20, 10), names=("s", "p", "d"))})
        draws = ps.sample(20_000, seed=2)
        assert list(draws.columns) == ["p[s]", "p[p]", "p[d]"]
        np.testing.assert_allclose(draws.sum(axis=1), 1.0)
        np.testing.assert_allclose(draws.mean(), [0.7, 0.2, 0.1], atol=0.01)

    def test_unknown_parameter_in_correlation_raises(self):
        with pytest.raises(KeyError):
            ParameterSet({"a": Normal(0, 1)}, correlation={("a", "zzz"): 0.2})

    def test_correlation_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ParameterSet({"a": Normal(0, 1), "b": Normal(0, 1)}, correlation={("a", "b"): 1.5})

    def test_means_and_spec(self):
        ps = ParameterSet({"a": Fixed(3.0), "p": Dirichlet((1.0, 1.0))})
        assert ps.means()["a"] == 3.0
        assert ps.means()["p[0]"] == pytest.approx(0.5)
        assert "Fixed" in ps.spec()["a"]


class TestMixDraws:
    def test_columns_union_and_index_contract(self):
        a = ParameterSet({"x": Normal(0, 1), "y": Normal(0, 1)}).sample(300, seed=1)
        b = ParameterSet({"z": Uniform(0, 1)}).sample(500, seed=2)
        mixed = mix_draws(a, b, seed=0)
        assert list(mixed.columns) == ["x", "y", "z"]
        assert len(mixed) == 300  # min source length
        assert mixed.index.name == "iteration"
        assert list(mixed.index) == list(range(300))

    def test_explicit_n_resamples_short_source_with_replacement(self):
        a = ParameterSet({"x": Normal(0, 1)}).sample(50, seed=1)
        b = ParameterSet({"z": Uniform(0, 1)}).sample(2_000, seed=2)
        mixed = mix_draws(a, b, n=1_000, seed=0)
        assert len(mixed) == 1_000
        # the 50-row source must have duplicated rows to reach 1000
        assert mixed["x"].nunique() <= 50

    def test_reproducible_under_seed(self):
        a = ParameterSet({"x": Normal(0, 1)}).sample(400, seed=1)
        b = ParameterSet({"z": Uniform(0, 1)}).sample(400, seed=2)
        pd.testing.assert_frame_equal(mix_draws(a, b, seed=7), mix_draws(a, b, seed=7))

    def test_preserves_within_source_correlation(self):
        # Two columns from ONE correlated source: their Spearman correlation
        # must survive mixing because whole rows are resampled together.
        joint = ParameterSet(
            {"a": Normal(0, 1), "b": Gamma.from_mean_se(10, 3)},
            correlation={("a", "b"): 0.6},
        ).sample(20_000, seed=11)
        other = ParameterSet({"c": Uniform(0, 1)}).sample(20_000, seed=12)
        rho_before = stats.spearmanr(joint["a"], joint["b"]).statistic
        mixed = mix_draws(joint, other, seed=3)
        rho_after = stats.spearmanr(mixed["a"], mixed["b"]).statistic
        assert rho_after == pytest.approx(rho_before, abs=0.02)

    def test_sources_are_independent_after_mixing(self):
        # Two sources built from the same seed carry identical row order, yet
        # mixing must break any cross-source correlation.
        a = ParameterSet({"x": Normal(0, 1)}).sample(20_000, seed=1)
        b = ParameterSet({"z": Normal(0, 1)}).sample(20_000, seed=1)
        assert stats.spearmanr(a["x"].to_numpy(), b["z"].to_numpy()).statistic == pytest.approx(
            1.0
        )  # identical order before mixing
        mixed = mix_draws(a, b, seed=5)
        assert abs(stats.spearmanr(mixed["x"], mixed["z"]).statistic) < 0.03

    def test_disjoint_columns_required(self):
        a = ParameterSet({"x": Normal(0, 1)}).sample(10, seed=1)
        b = ParameterSet({"x": Uniform(0, 1)}).sample(10, seed=2)
        with pytest.raises(ValueError, match="disjoint"):
            mix_draws(a, b)

    def test_empty_source_rejected(self):
        a = ParameterSet({"x": Normal(0, 1)}).sample(10, seed=1)
        with pytest.raises(ValueError, match="empty"):
            mix_draws(a, a.iloc[:0])

    def test_no_sources_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            mix_draws()


class TestSingleDraw:
    def test_shape_and_index(self):
        m = single_draw({"p_die": 0.1, "cost": 1000.0})
        assert m.shape == (1, 2)
        assert m.index.name == "iteration"
        assert list(m.index) == [0]

    def test_run_psa_accepts_it(self):
        m = single_draw({"c": 100.0})
        out = run_psa(_two_strategy_model, m).outcomes
        assert out.n_iterations == 1

    def test_at_means_matches_single_draw(self):
        ps = ParameterSet({"c": Fixed(2.0), "d": Normal(5.0, 1.0)})
        base = ps.at_means()
        assert base.shape == (1, 2)
        assert float(base.loc[0, "c"]) == pytest.approx(2.0)
        assert float(base.loc[0, "d"]) == pytest.approx(5.0)

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            single_draw({})


class TestReadDraws:
    def test_dataframe_gets_iteration_index(self):
        df = pd.DataFrame({"c": [100.0, 110.0], "u": [0.5, 0.6]})
        m = read_draws(df)
        assert m.index.name == "iteration"
        assert list(m.index) == [0, 1]

    def test_run_psa_accepts_it(self):
        df = pd.DataFrame({"c": [100.0, 110.0, 120.0]})
        out = run_psa(_two_strategy_model, read_draws(df)).outcomes
        assert out.n_iterations == 3

    def test_reads_csv_path(self, tmp_path):
        path = tmp_path / "draws.csv"
        pd.DataFrame({"c": [1.0, 2.0]}).to_csv(path, index=False)
        m = read_draws(path)
        assert list(m["c"]) == [1.0, 2.0]

    def test_honours_explicit_iteration_column(self):
        df = pd.DataFrame({"iteration": [10, 20, 30], "c": [1.0, 2.0, 3.0]})
        m = read_draws(df, iteration="iteration")
        assert list(m.index) == [10, 20, 30]
        assert m.index.name == "iteration"
        assert list(m.columns) == ["c"]

    def test_rejects_non_numeric_column(self):
        df = pd.DataFrame({"c": [1.0, 2.0], "label": ["a", "b"]})
        with pytest.raises(ValueError, match="must be numeric"):
            read_draws(df)

    def test_missing_iteration_column_message(self):
        df = pd.DataFrame({"c": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not in the table"):
            read_draws(df, iteration="iter")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            read_draws(pd.DataFrame({"c": []}))


class TestResamplePosterior:
    def test_shape_and_index(self):
        post = pd.DataFrame({"beta": [0.1, 0.2, 0.3], "weight": [1.0, 2.0, 1.0]})
        m = resample_posterior(post, n=500, seed=0)
        assert m.shape == (500, 1)
        assert m.index.name == "iteration"
        assert "weight" not in m.columns

    def test_run_psa_accepts_it(self):
        post = pd.DataFrame({"c": [100.0, 110.0], "weight": [1.0, 1.0]})
        out = run_psa(_two_strategy_model, resample_posterior(post, n=50, seed=1)).outcomes
        assert out.n_iterations == 50

    def test_recovers_weighted_means(self):
        rng = np.random.default_rng(0)
        grid = pd.DataFrame(
            {
                "x": rng.normal(0.0, 1.0, 400),
                "y": rng.normal(5.0, 2.0, 400),
                "weight": rng.uniform(0.1, 1.0, 400),
            }
        )
        w = grid["weight"].to_numpy()
        expected_x = np.average(grid["x"], weights=w)
        expected_y = np.average(grid["y"], weights=w)
        resampled = resample_posterior(grid, n=200_000, seed=7)
        assert resampled["x"].mean() == pytest.approx(expected_x, abs=0.02)
        assert resampled["y"].mean() == pytest.approx(expected_y, abs=0.04)

    def test_preserves_column_correlation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 1.0, 3_000)
        y = 0.8 * x + rng.normal(0.0, 0.6, 3_000)  # strong rank dependence
        grid = pd.DataFrame({"x": x, "y": y, "weight": rng.uniform(0.2, 1.0, 3_000)})
        rho_before = stats.spearmanr(grid["x"], grid["y"]).statistic
        resampled = resample_posterior(grid, n=50_000, seed=3)
        rho_after = stats.spearmanr(resampled["x"], resampled["y"]).statistic
        assert rho_after == pytest.approx(rho_before, abs=0.03)

    def test_rejects_negative_weight(self):
        post = pd.DataFrame({"c": [1.0, 2.0], "weight": [-1.0, 1.0]})
        with pytest.raises(ValueError, match="non-negative"):
            resample_posterior(post, n=10)

    def test_rejects_zero_weight_sum(self):
        post = pd.DataFrame({"c": [1.0, 2.0], "weight": [0.0, 0.0]})
        with pytest.raises(ValueError, match="positive value"):
            resample_posterior(post, n=10)

    def test_missing_weight_column_message(self):
        post = pd.DataFrame({"c": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not in the table"):
            resample_posterior(post, n=10)

    def test_rejects_nonpositive_n(self):
        post = pd.DataFrame({"c": [1.0, 2.0], "weight": [1.0, 1.0]})
        with pytest.raises(ValueError, match="positive integer"):
            resample_posterior(post, n=0)
