"""Tests for the event-history trace and the epidemiological outcome helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heormodel.epi import prevalence, state_occupancy, survival
from heormodel.models import MicrosimModel
from heormodel.run import SeedManager


def _draws(n_iter=1):
    return pd.DataFrame(
        {"unused": np.zeros(n_iter)}, index=pd.RangeIndex(n_iter, name="iteration")
    )


def _hand_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": "care",
            "iteration": 0,
            "individual": [0, 0, 1],
            "time": [1.0, 3.0, 2.0],
            "from_state": ["H", "S", "H"],
            "to_state": ["S", "D", "D"],
        }
    )


class TestStateOccupancy:
    def test_hand_built_log(self):
        occ = state_occupancy(
            _hand_log(), states=("H", "S", "D"), initial_state="H",
            n_individuals=4, times=[0.0, 1.0, 2.5, 5.0],
        )
        assert occ.loc[("care", 0, 0.0)].tolist() == [1.0, 0.0, 0.0]
        # An event at exactly the requested time counts as having happened.
        assert occ.loc[("care", 0, 1.0)].tolist() == [0.75, 0.25, 0.0]
        assert occ.loc[("care", 0, 2.5)].tolist() == [0.5, 0.25, 0.25]
        assert occ.loc[("care", 0, 5.0)].tolist() == [0.5, 0.0, 0.5]
        assert np.allclose(occ.sum(axis=1), 1.0)

    def test_survival_and_prevalence(self):
        occ = state_occupancy(
            _hand_log(), states=("H", "S", "D"), initial_state="H",
            n_individuals=4, times=[2.5],
        )
        assert survival(occ, dead_state="D").tolist() == [0.75]
        assert prevalence(occ, states=("S",), dead_state="D").iloc[0] == pytest.approx(1 / 3)

    def test_prevalence_nan_when_no_one_alive(self):
        events = _hand_log()
        occ = state_occupancy(
            events, states=("H", "S", "D"), initial_state="H", n_individuals=2, times=[10.0]
        )
        assert np.isnan(prevalence(occ, states=("S",), dead_state="D").iloc[0])

    def test_rejects_unknown_states_and_missing_columns(self):
        with pytest.raises(ValueError, match="not listed in states"):
            state_occupancy(
                _hand_log(), states=("H", "D"), initial_state="H",
                n_individuals=4, times=[0.0],
            )
        with pytest.raises(ValueError, match="missing columns"):
            state_occupancy(
                _hand_log().drop(columns=["individual"]), states=("H", "S", "D"),
                initial_state="H", n_individuals=4, times=[0.0],
            )


class TestEventTrace:
    def _continuous_engine(self, population=20_000, seed=5):
        lam = 0.1

        def hazards(params, state, attrs, rng):
            times = np.full((len(state), 2), np.inf)
            alive = state == 0
            times[alive, 1] = rng.exponential(1.0 / lam, int(alive.sum()))
            return times

        def payoffs(params, state, attrs):
            alive = (state == 0).astype(float)
            return alive * 0.0, alive

        return MicrosimModel(
            states=("alive", "dead"),
            clock="continuous",
            hazards=hazards,
            payoffs=payoffs,
            population=population,
            strategies={"care": {}},
            horizon=50.0,
            seed_manager=SeedManager(seed),
        )

    def test_continuous_survival_matches_exponential(self):
        engine = self._continuous_engine()
        _, events = engine.evaluate(_draws(), trace="events")
        assert list(events.columns) == [
            "strategy", "iteration", "individual", "time", "from_state", "to_state",
        ]
        occ = state_occupancy(
            events, states=("alive", "dead"), initial_state="alive",
            n_individuals=20_000, times=[5.0, 10.0, 20.0],
        )
        surv = survival(occ, dead_state="dead")
        for t in (5.0, 10.0, 20.0):
            assert surv.loc[("care", 0, t)] == pytest.approx(np.exp(-0.1 * t), abs=0.01)

    def test_events_do_not_change_outcomes(self):
        plain = self._continuous_engine().evaluate(_draws())
        with_events, _ = self._continuous_engine().evaluate(_draws(), trace="events")
        pd.testing.assert_frame_equal(plain.data, with_events.data)

    def test_discrete_clock_logs_state_changes(self):
        def transition(params, state, attrs, rng):
            probs = np.zeros((len(state), 2))
            probs[state == 0] = [0.7, 0.3]
            probs[state == 1] = [0.0, 1.0]
            return probs

        def payoffs(params, state, attrs):
            alive = (state == 0).astype(float)
            return alive, alive

        engine = MicrosimModel(
            states=("alive", "dead"),
            transition=transition,
            payoffs=payoffs,
            population=50_000,
            strategies={"care": {}},
            horizon=10,
            seed_manager=SeedManager(6),
            half_cycle_correction=False,
        )
        _, events = engine.evaluate(_draws(), trace="events")
        occ = state_occupancy(
            events, states=("alive", "dead"), initial_state="alive",
            n_individuals=50_000, times=[1.0, 3.0],
        )
        assert occ.loc[("care", 0, 1.0), "alive"] == pytest.approx(0.7, abs=0.01)
        assert occ.loc[("care", 0, 3.0), "alive"] == pytest.approx(0.7**3, abs=0.01)

    def test_bad_trace_value_rejected(self):
        with pytest.raises(ValueError, match="trace must be"):
            self._continuous_engine(population=10).evaluate(_draws(), trace="event")
