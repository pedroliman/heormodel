"""Tests for the microsimulation engine.

Two validation checks anchor the file: the discrete clock converges to the
closed-form cohort solution it mirrors, and the continuous clock on constant
hazards recovers the exponential cohort integral. The rest cover reproducibility
across ``n_jobs``, common random numbers, and the output contract.
"""

import numpy as np
import pandas as pd
import pytest

from heval.models import MicrosimModel, ModelEngine
from heval.run import SeedManager, run_psa

# A three-state cohort: Healthy, Sick, Dead. Constant transitions and payoffs.
P = np.array(
    [
        [0.80, 0.15, 0.05],
        [0.00, 0.90, 0.10],
        [0.00, 0.00, 1.00],
    ]
)
COST_VEC = np.array([1_000.0, 3_000.0, 0.0])
EFF_VEC = np.array([1.0, 0.6, 0.0])


def _transition(params, state, attrs, rng):
    return P[state]


def _payoffs(params, state, attrs):
    return COST_VEC[state], EFF_VEC[state]


def _cohort_reference(horizon, rate_cost, rate_eff, half_cycle, cycle_length=1.0):
    """Exact discounted cohort totals with the engine's accrual convention."""
    p = np.array([1.0, 0.0, 0.0])
    n_points = horizon + 1
    weights = np.ones(n_points)
    if half_cycle:
        weights[0] = weights[-1] = 0.5
    cost = eff = 0.0
    for c in range(n_points):
        t = c * cycle_length
        cost += weights[c] * (1 + rate_cost) ** (-t) * float(p @ COST_VEC)
        eff += weights[c] * (1 + rate_eff) ** (-t) * float(p @ EFF_VEC)
        p = p @ P
    return cost, eff


def _draws(n_iter=1):
    return pd.DataFrame(
        {"unused": np.zeros(n_iter)}, index=pd.RangeIndex(n_iter, name="iteration")
    )


class TestDiscreteValidation:
    def test_converges_to_cohort_closed_form(self):
        horizon = 40
        engine = MicrosimModel(
            states=("H", "S", "D"),
            transition=_transition,
            payoffs=_payoffs,
            population=60_000,
            strategies={"care": {}},
            horizon=horizon,
            discount_rate=0.03,
            half_cycle_correction=True,
            seed_manager=SeedManager(2026),
        )
        out = engine.evaluate(_draws())
        ref_cost, ref_eff = _cohort_reference(horizon, 0.03, 0.03, half_cycle=True)
        got = out.summary().loc["care"]
        assert got["cost"] == pytest.approx(ref_cost, rel=0.01)
        assert got["qaly"] == pytest.approx(ref_eff, rel=0.01)

    def test_half_cycle_flag_changes_result(self):
        ref_on = _cohort_reference(40, 0.03, 0.03, half_cycle=True)
        ref_off = _cohort_reference(40, 0.03, 0.03, half_cycle=False)
        assert ref_on != ref_off


class TestContinuousValidation:
    def test_constant_hazard_matches_exponential_cohort(self):
        lam, cost_year, horizon, rate = 0.05, 1_000.0, 40.0, 0.03

        def hazards(params, state, attrs, rng):
            times = np.full((len(state), 2), np.inf)
            alive = state == 0
            times[alive, 1] = rng.exponential(1.0 / lam, int(alive.sum()))
            return times

        def payoffs(params, state, attrs):
            alive = (state == 0).astype(float)
            return alive * cost_year, alive

        engine = MicrosimModel(
            states=("alive", "dead"),
            clock="continuous",
            hazards=hazards,
            payoffs=payoffs,
            population=60_000,
            strategies={"care": {}},
            horizon=horizon,
            discount_rate=rate,
            seed_manager=SeedManager(7),
        )
        out = engine.evaluate(_draws())
        disc_ly = (1 - np.exp(-(rate + lam) * horizon)) / (rate + lam)
        got = out.summary().loc["care"]
        assert got["cost"] == pytest.approx(cost_year * disc_ly, rel=0.01)
        assert got["qaly"] == pytest.approx(disc_ly, rel=0.01)

    def test_max_events_guard(self):
        def hazards(params, state, attrs, rng):
            # Ping-pong between the two states, never absorbing: events pile up.
            times = np.empty((len(state), 2))
            times[:, 0] = np.where(state == 0, np.inf, 0.001)
            times[:, 1] = np.where(state == 0, 0.001, np.inf)
            return times

        def payoffs(params, state, attrs):
            return np.zeros(len(state)), np.zeros(len(state))

        engine = MicrosimModel(
            states=("a", "b"),
            clock="continuous",
            hazards=hazards,
            payoffs=payoffs,
            population=10,
            strategies={"care": {}},
            horizon=100.0,
            seed_manager=SeedManager(1),
            max_events=50,
        )
        with pytest.raises(RuntimeError, match="max_events"):
            engine.evaluate(_draws())


def _small_engine(**overrides):
    kwargs = dict(
        states=("H", "S", "D"),
        transition=_transition,
        payoffs=_payoffs,
        population=200,
        strategies={"care": {}},
        horizon=10,
        seed_manager=SeedManager(99),
    )
    kwargs.update(overrides)
    return MicrosimModel(**kwargs)


class TestReproducibility:
    def test_same_seed_identical_across_n_jobs(self):
        draws = _draws(6)
        serial = run_psa(_small_engine(), draws, sequential=True)
        parallel = run_psa(_small_engine(), draws, n_jobs=2)
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
        out = run_psa(_small_engine(), draws)
        assert out.iterations.equals(draws.index)

    def test_strategies_in_declared_order(self):
        engine = _small_engine(strategies={"Tx": {}, "SoC": {}})
        assert engine.evaluate(_draws(2)).strategies == ["Tx", "SoC"]

    def test_overrides_reach_the_model(self):
        # An override flips a payoff multiplier read from params.
        def payoffs(params, state, attrs):
            return COST_VEC[state] * params.get("mult", 1.0), EFF_VEC[state]

        engine = _small_engine(
            payoffs=payoffs, strategies={"base": {}, "double": {"mult": 2.0}}
        )
        summary = engine.evaluate(_draws(3)).summary()
        assert summary.loc["double", "cost"] == pytest.approx(
            2.0 * summary.loc["base", "cost"]
        )


class TestPopulationAndTrace:
    def test_population_sampler_supplies_attributes(self):
        def population(rng, n):
            return pd.DataFrame({"frail": rng.random(n)})

        def payoffs(params, state, attrs):
            # Frailer individuals cost more while sick.
            sick = (state == 1).astype(float)
            return COST_VEC[state] + sick * attrs["frail"].to_numpy() * 1_000.0, EFF_VEC[state]

        engine = _small_engine(payoffs=payoffs, population=population, n_individuals=300)
        out = engine.evaluate(_draws(2))
        assert out.n_iterations == 2

    def test_trace_returns_individual_rows(self):
        engine = _small_engine(strategies={"A": {}, "B": {}})
        out, trace = engine.evaluate(_draws(2), trace=True)
        assert out.strategies == ["A", "B"]
        assert len(trace) == 2 * 2 * 200  # strategies x iterations x individuals
        assert set(trace.columns) == {"strategy", "iteration", "individual", "cost", "qaly"}


class TestDurationGroups:
    def test_counter_spans_a_set_of_states(self):
        """A duration group counts consecutive cycles across S1 and S2 together."""
        # deterministic path H, S1, S1, S2, S2, H, S1 via a cycle-indexed transition
        path = ["H", "S1", "S1", "S2", "S2", "H", "S1"]
        idx = {"H": 0, "S1": 1, "S2": 2, "D": 3}
        seen = []

        def transition(params, state, attrs, rng):
            c = int(attrs["cycle"].iloc[0])
            seen.append(int(attrs["sick_dur"].iloc[0]))
            probs = np.zeros((1, 4))
            probs[0, idx[path[c + 1]]] = 1.0
            return probs

        def payoffs(params, state, attrs):
            return np.zeros(1), np.zeros(1)

        engine = MicrosimModel(
            states=("H", "S1", "S2", "D"), transition=transition, payoffs=payoffs,
            population=1, strategies={"s": {}}, horizon=len(path) - 1,
            seed_manager=SeedManager(0), duration_groups={"sick_dur": ("S1", "S2")},
        )
        engine.evaluate(_draws())
        # duration is 0 on entry to the sick complex and keeps counting through
        # the Sick to Sicker progression, resetting only when Healthy
        assert seen == [0, 0, 1, 2, 3, 0]  # cycles 0..5 (last cycle takes no step)

    def test_absent_by_default(self):
        def transition(params, state, attrs, rng):
            assert "sick_dur" not in attrs.columns
            return _transition(params, state, attrs, rng)

        engine = _small_engine(transition=transition)
        engine.evaluate(_draws())


class TestClockValidation:
    def test_bad_clock_rejected(self):
        with pytest.raises(ValueError, match="clock"):
            MicrosimModel(
                states=("H", "S", "D"), transition=_transition, payoffs=_payoffs,
                population=10, strategies={"care": {}}, seed_manager=SeedManager(0),
                clock="quarterly",
            )

    def test_discrete_requires_transition(self):
        with pytest.raises(TypeError, match="transition"):
            MicrosimModel(
                states=("H", "S", "D"), payoffs=_payoffs, population=10,
                strategies={"care": {}}, seed_manager=SeedManager(0),
            )

    def test_continuous_requires_hazards(self):
        with pytest.raises(TypeError, match="hazards"):
            MicrosimModel(
                states=("H", "S", "D"), clock="continuous", payoffs=_payoffs,
                population=10, strategies={"care": {}}, seed_manager=SeedManager(0),
            )
