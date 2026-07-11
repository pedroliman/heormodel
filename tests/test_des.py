"""Tests for the discrete-event simulation engine.

Two validation checks anchor the file. A single-resource M/M/1 clinic recovers
the analytic queue wait and throughput. A no-resource DES with exponential event
times reproduces the same exponential cohort integral used to validate the
continuous-time microsim, and the two engines agree with each other and the
closed form. The rest cover reproducibility across ``n_jobs``, common random
numbers, the trace side channel, disaggregated components, and the contract.
"""

import numpy as np
import pandas as pd
import pytest

simpy = pytest.importorskip("simpy")

from heormodel.models import (  # noqa: E402
    DESModel,
    MicrosimModel,
    ModelEngine,
    queue_waits,
)
from heormodel.run import run_psa  # noqa: E402


def _draws(n_iter=1):
    return pd.DataFrame(
        {"unused": np.zeros(n_iter)}, index=pd.RangeIndex(n_iter, name="iteration")
    )


class TestMM1Validation:
    def test_recovers_analytic_wait_and_throughput(self):
        lam, mu, n = 0.5, 1.0, 20_000  # rho = 0.5

        def entities(rng, n):
            return pd.DataFrame({"arrival": np.cumsum(rng.exponential(1.0 / lam, n))})

        def resources(env, params, strategy):
            return {"clinician": simpy.Resource(env, capacity=1)}

        def process(env, entity, params, strategy, toolkit):
            yield env.timeout(float(entity["arrival"]))
            with toolkit.request("clinician") as req:
                yield req
                yield env.timeout(toolkit.rng.exponential(1.0 / mu))

        engine = DESModel(
            process=process,
            population=entities,
            n_individuals=n,
            resources=resources,
            strategies={"clinic": {}},
            horizon=n / lam * 1.5,  # comfortably past the last arrival
        )
        trace = run_psa(engine, _draws(), seed=3, collect="events").events

        waits = queue_waits(trace).iloc[1_000:]  # drop the warm-up transient
        wq = (lam / mu) / (mu - lam)  # M/M/1 mean time in queue
        assert waits["wait"].mean() == pytest.approx(wq, rel=0.1)

        served = int((trace["event"] == "grant").sum())
        span = trace.loc[trace["event"] == "release", "t"].max()
        assert served / span == pytest.approx(lam, rel=0.05)  # throughput equals arrivals


class TestExponentialCohort:
    """A no-resource DES must match the exponential cohort and the microsim."""

    LAM, COST_YEAR, HORIZON, RATE = 0.05, 1_000.0, 40.0, 0.03

    def _des(self):
        lam, cost_year, horizon = self.LAM, self.COST_YEAR, self.HORIZON

        def process(env, entity, params, strategy, toolkit):
            tdeath = toolkit.rng.exponential(1.0 / lam)
            toolkit.accrue_rate(cost_year, 1.0, tdeath)  # truncated at the horizon
            yield env.timeout(min(tdeath, horizon))

        return DESModel(
            process=process,
            population=60_000,
            strategies={"care": {}},
            horizon=horizon,
            discount_rate=self.RATE,
        )

    def _microsim(self):
        lam, cost_year = self.LAM, self.COST_YEAR

        def event_times(params, strategy, state, attrs, rng):
            times = np.full((len(state), 2), np.inf)
            alive = state == 0
            times[alive, 1] = rng.exponential(1.0 / lam, int(alive.sum()))
            return times

        def reward_rates(params, strategy, state, attrs):
            alive = (state == 0).astype(float)
            return alive * cost_year, alive

        return MicrosimModel.continuous(
            states=("alive", "dead"),
            event_times=event_times,
            state_reward_rates=reward_rates,
            population=60_000,
            strategies=["care"],
            horizon=self.HORIZON,
            discount_rate=self.RATE,
        )

    def test_des_matches_closed_form(self):
        disc_ly = (1 - np.exp(-(self.RATE + self.LAM) * self.HORIZON)) / (self.RATE + self.LAM)
        got = run_psa(self._des(), _draws(), seed=7).outcomes.summary().loc["care"]
        assert got["cost"] == pytest.approx(self.COST_YEAR * disc_ly, rel=0.01)
        assert got["qaly"] == pytest.approx(disc_ly, rel=0.01)

    def test_des_and_microsim_agree(self):
        des = run_psa(self._des(), _draws(), seed=7).outcomes.summary().loc["care"]
        micro = run_psa(self._microsim(), _draws(), seed=7).outcomes.summary().loc["care"]
        assert des["cost"] == pytest.approx(micro["cost"], rel=0.01)
        assert des["qaly"] == pytest.approx(micro["qaly"], rel=0.01)


def _small_engine(**overrides):
    """A tiny stochastic no-resource engine for the behavioural tests."""

    def process(env, entity, params, strategy, toolkit):
        cost = toolkit.rng.gamma(2.0, params.get("scale", 1_000.0))
        toolkit.accrue_cost(cost)
        toolkit.accrue_rate(0.0, 1.0, toolkit.rng.exponential(5.0))
        yield env.timeout(1.0)

    kwargs = dict(
        process=process,
        population=200,
        strategies={"care": {}},
        horizon=10.0,
    )
    kwargs.update(overrides)
    return DESModel(**kwargs)


class TestReproducibility:
    def test_same_seed_identical_across_n_jobs(self):
        draws = _draws(6)
        serial = run_psa(_small_engine(), draws, seed=99, sequential=True).outcomes
        parallel = run_psa(_small_engine(), draws, seed=99, n_jobs=2).outcomes
        pd.testing.assert_frame_equal(serial.data, parallel.data)

    def test_different_iterations_differ(self):
        out = _small_engine().evaluate(_draws(2))
        costs = out.costs_wide()["care"]
        assert costs.iloc[0] != costs.iloc[1]

    def test_reevaluation_is_deterministic(self):
        engine = _small_engine()
        first = engine.evaluate(_draws(3))
        second = engine.evaluate(_draws(3))
        pd.testing.assert_frame_equal(first.data, second.data)


class TestCommonRandomNumbers:
    def test_crn_makes_identical_strategies_match(self):
        engine = _small_engine(strategies={"A": {}, "B": {}})
        out = engine.evaluate(_draws(4))
        a = out.select(["A"]).data.reset_index(drop=True)
        b = out.select(["B"]).data.reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_independent_streams_break_the_tie(self):
        engine = _small_engine(strategies={"A": {}, "B": {}}, independent_streams=True)
        out = engine.evaluate(_draws(4))
        a = out.select(["A"]).data.reset_index(drop=True)
        b = out.select(["B"]).data.reset_index(drop=True)
        assert not np.allclose(a.to_numpy(), b.to_numpy())


class TestContract:
    def test_is_model_engine(self):
        assert isinstance(_small_engine(), ModelEngine)

    def test_iteration_index_preserved(self):
        draws = _draws(5)
        out = run_psa(_small_engine(), draws).outcomes
        assert out.iterations.equals(draws.index)

    def test_strategies_in_declared_order(self):
        engine = _small_engine(strategies={"Tx": {}, "SoC": {}})
        assert engine.evaluate(_draws(2)).strategies == ["Tx", "SoC"]

    def test_overrides_reach_the_model(self):
        engine = _small_engine(strategies={"base": {}, "double": {"scale": 2_000.0}})
        summary = engine.evaluate(_draws(4)).summary()
        assert summary.loc["double", "cost"] == pytest.approx(
            2.0 * summary.loc["base", "cost"], rel=0.05
        )


class TestAccrualDetails:
    def test_point_cost_is_discounted_at_event_time(self):
        def process(env, entity, params, strategy, toolkit):
            yield env.timeout(2.0)
            toolkit.accrue_cost(100.0)

        engine = DESModel(
            process=process,
            population=1,
            strategies={"care": {}},
            horizon=10.0,
            discount_rate=0.03,
        )
        got = engine.evaluate(_draws()).summary().loc["care", "cost"]
        assert got == pytest.approx(100.0 * np.exp(-0.03 * 2.0), rel=1e-9)

    def test_rate_accrual_truncates_at_horizon(self):
        def process(env, entity, params, strategy, toolkit):
            toolkit.accrue_rate(1.0, 1.0, 1_000.0)  # far past the horizon
            yield env.timeout(1.0)

        engine = DESModel(
            process=process,
            population=1,
            strategies={"care": {}},
            horizon=10.0,
            discount_rate=0.0,
        )
        got = engine.evaluate(_draws()).summary().loc["care"]
        assert got["cost"] == pytest.approx(10.0)  # undiscounted length of the horizon
        assert got["qaly"] == pytest.approx(10.0)

    def test_components_map_to_outcome_columns(self):
        def process(env, entity, params, strategy, toolkit):
            toolkit.accrue_cost(100.0, component="cost_drug")
            toolkit.accrue_cost(30.0, component="cost_clinic")
            yield env.timeout(1.0)

        engine = DESModel(
            process=process,
            population=10,
            strategies={"care": {}},
            horizon=5.0,
            discount_rate=0.0,
        )
        out = engine.evaluate(_draws())
        assert set(out.components) == {"cost_drug", "cost_clinic"}
        summary = out.summary().loc["care"]
        assert summary["cost"] == pytest.approx(130.0)
        assert summary["cost_drug"] == pytest.approx(100.0)
        assert summary["cost_clinic"] == pytest.approx(30.0)


class TestTraceAndGuards:
    def test_trace_returns_event_rows(self):
        def process(env, entity, params, strategy, toolkit):
            toolkit.state("treated")
            toolkit.accrue_cost(10.0)
            yield env.timeout(1.0)

        engine = DESModel(
            process=process,
            population=5,
            strategies={"A": {}, "B": {}},
            horizon=5.0,
        )
        result = run_psa(engine, _draws(2), seed=1, collect="events")
        out, trace = result.outcomes, result.events
        assert out.strategies == ["A", "B"]
        assert set(trace.columns) == {
            "strategy",
            "iteration",
            "entity",
            "t",
            "event",
            "state",
            "resource",
        }
        assert (trace["event"] == "state").sum() == 2 * 2 * 5  # strategies x iters x entities

    def test_unknown_resource_raises(self):
        def process(env, entity, params, strategy, toolkit):
            with toolkit.request("missing") as req:
                yield req

        engine = DESModel(
            process=process,
            population=1,
            strategies={"care": {}},
            horizon=5.0,
        )
        with pytest.raises(KeyError, match="missing"):
            engine.evaluate(_draws())

    def test_rejects_bad_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            _small_engine(horizon=0.0)

    def test_rejects_empty_draws(self):
        with pytest.raises(ValueError, match="empty"):
            _small_engine().evaluate(pd.DataFrame(index=pd.RangeIndex(0, name="iteration")))
