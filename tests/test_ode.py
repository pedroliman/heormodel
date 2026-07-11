"""Tests for the ordinary differential equation (compartmental) engine.

A closed-form solution anchors the file: an exponential-decay compartment whose
discounted occupancy and flow-event rewards both integrate in closed form, which
the engine must reproduce. The rest cover discounting, the flow-reward channel,
the output contract, input validation, and a sanity check on the shipped SEIR
vaccination example.
"""

import numpy as np
import pandas as pd
import pytest

from heormodel.cea import icer_table
from heormodel.models import Intervention, ModelEngine, ODEModel, ODESpec, Outcomes


def _draws(n_iter=1, **cols):
    if not cols:
        cols = {"z": np.zeros(n_iter)}
    return pd.DataFrame(cols, index=pd.RangeIndex(n_iter, name="iteration"))


def _decay_engine(*, horizon=20.0, rate=0.03, **kw):
    """Two-compartment exponential decay: alive decays to dead at rate k."""

    def model(params, intervention):
        k = params["k"]
        return ODESpec(
            derivatives=lambda t, y: np.array([-k * y[0], k * y[0]]),
            initial=np.array([1.0, 0.0]),
            state_cost=np.array([params.get("c", 0.0), 0.0]),
            state_effect=np.array([1.0, 0.0]),
        )

    return ODEModel(
        states=("alive", "dead"), interventions=("s",), dynamics_and_rewards=model,
        horizon=horizon, discount_rate=rate, **kw,
    )


# -- closed-form validation -------------------------------------------------


def test_exponential_decay_matches_closed_form():
    """Discounted occupancy of a decaying compartment matches the analytic integral."""
    k, c, rate, horizon = 0.1, 1000.0, 0.03, 20.0
    engine = _decay_engine(horizon=horizon, rate=rate)
    got = engine.evaluate(_draws(1, k=[k], c=[c])).summary().loc["s"]
    # occupancy alive(t) = e^{-kt}; discounted integral of e^{-(k+r)t} over [0, H]
    integral = (1.0 - np.exp(-(k + rate) * horizon)) / (k + rate)
    assert got["qaly"] == pytest.approx(integral, rel=1e-5)
    assert got["cost"] == pytest.approx(c * integral, rel=1e-5)


def test_discounting_reduces_totals():
    undiscounted = _decay_engine(rate=0.0).evaluate(_draws(1, k=[0.1])).summary().loc["s", "qaly"]
    discounted = _decay_engine(rate=0.05).evaluate(_draws(1, k=[0.1])).summary().loc["s", "qaly"]
    assert discounted < undiscounted


def test_flow_event_reward_charges_once_per_event():
    """A one-time cost per death, charged on the death flow, sums to the closed form."""
    k, death_cost, horizon = 0.2, 5000.0, 40.0

    def model(params, intervention):
        return ODESpec(
            derivatives=lambda t, y: np.array([-k * y[0], k * y[0]]),
            initial=np.array([1.0, 0.0]),
            state_cost=np.zeros(2),
            state_effect=np.zeros(2),
            event_rates=lambda t, y: np.array([k * y[0]]),  # deaths per year
            event_cost=np.array([death_cost]),
            event_effect=np.array([0.0]),
        )

    engine = ODEModel(
        states=("alive", "dead"), interventions=("s",), dynamics_and_rewards=model,
        horizon=horizon, discount_rate=0.0,
    )
    cost = engine.evaluate(_draws()).summary().loc["s", "cost"]
    # undiscounted, the integral of k e^{-kt} over [0, H] is 1 - e^{-kH}
    assert cost == pytest.approx(death_cost * (1.0 - np.exp(-k * horizon)), rel=1e-5)


# -- contract and validation ------------------------------------------------


def test_engine_satisfies_protocol_and_contract():
    def model(params, intervention):
        return ODESpec(
            derivatives=lambda t, y: np.array([-0.1 * y[0], 0.1 * y[0]]),
            initial=np.array([1.0, 0.0]),
            state_cost=np.array([params["c"], 0.0]),
            state_effect=np.array([1.0, 0.0]),
        )

    engine = ODEModel(
        states=("a", "d"), interventions=("x", "y"), dynamics_and_rewards=model, horizon=5.0,
    )
    assert isinstance(engine, ModelEngine)
    draws = _draws(4, c=np.arange(4, dtype=float))
    out = engine.evaluate(draws)
    assert isinstance(out, Outcomes)
    assert list(out.iterations) == list(draws.index)
    assert out.interventions == ["x", "y"]
    assert out.n_iterations == 4


def test_is_comparator_flag_reaches_the_outcomes():
    def model(params, intervention):
        return ODESpec(
            derivatives=lambda t, y: np.array([-0.1 * y[0], 0.1 * y[0]]),
            initial=np.array([1.0, 0.0]),
            state_cost=np.array([params["c"], 0.0]),
            state_effect=np.array([1.0, 0.0]),
        )

    engine = ODEModel(
        states=("a", "d"),
        interventions=["x", Intervention("y", is_comparator=True)],
        dynamics_and_rewards=model,
        horizon=5.0,
    )
    assert engine.evaluate(_draws(1, c=[1.0])).comparator == "y"


def _absorbing(params, intervention):
    return ODESpec(
        derivatives=lambda t, y: np.zeros(2),
        initial=np.array([1.0, 0.0]),
        state_cost=np.zeros(2),
        state_effect=np.zeros(2),
    )


def test_horizon_must_be_positive():
    with pytest.raises(ValueError, match="horizon must be positive"):
        ODEModel(states=("a", "d"), interventions=("s",), dynamics_and_rewards=_absorbing,
                 horizon=0.0)


def test_at_least_two_states():
    with pytest.raises(ValueError, match="at least two states"):
        ODEModel(states=("a",), interventions=("s",), dynamics_and_rewards=_absorbing, horizon=5.0)


def test_empty_draws_rejected():
    engine = ODEModel(states=("a", "d"), interventions=("s",), dynamics_and_rewards=_absorbing,
                      horizon=5.0)
    with pytest.raises(ValueError, match="empty"):
        engine.evaluate(pd.DataFrame(index=pd.RangeIndex(0, name="iteration")))


def test_initial_wrong_length_rejected():
    def model(params, intervention):
        return ODESpec(
            derivatives=lambda t, y: np.zeros(2),
            initial=np.array([1.0, 0.0, 0.0]),  # three, not two
            state_cost=np.zeros(2), state_effect=np.zeros(2),
        )

    engine = ODEModel(states=("a", "d"), interventions=("s",), dynamics_and_rewards=model,
                      horizon=5.0)
    with pytest.raises(ValueError, match="initial must have shape"):
        engine.evaluate(_draws())


def test_state_reward_wrong_length_rejected():
    def model(params, intervention):
        return ODESpec(
            derivatives=lambda t, y: np.zeros(2),
            initial=np.array([1.0, 0.0]),
            state_cost=np.zeros(3), state_effect=np.zeros(2),  # cost wrong length
        )

    engine = ODEModel(states=("a", "d"), interventions=("s",), dynamics_and_rewards=model,
                      horizon=5.0)
    with pytest.raises(ValueError, match="state_cost must have shape"):
        engine.evaluate(_draws())


def test_derivatives_wrong_shape_rejected():
    def model(params, intervention):
        return ODESpec(
            derivatives=lambda t, y: np.zeros(3),  # three, not two
            initial=np.array([1.0, 0.0]),
            state_cost=np.zeros(2), state_effect=np.zeros(2),
        )

    engine = ODEModel(states=("a", "d"), interventions=("s",), dynamics_and_rewards=model,
                      horizon=5.0)
    with pytest.raises(ValueError, match="derivatives must return shape"):
        engine.evaluate(_draws())


def test_event_rewards_require_both_amounts():
    def model(params, intervention):
        return ODESpec(
            derivatives=lambda t, y: np.zeros(2),
            initial=np.array([1.0, 0.0]),
            state_cost=np.zeros(2), state_effect=np.zeros(2),
            event_rates=lambda t, y: np.array([0.0]),
            event_cost=np.array([1.0]),  # event_effect omitted
        )

    engine = ODEModel(states=("a", "d"), interventions=("s",), dynamics_and_rewards=model,
                      horizon=5.0)
    with pytest.raises(ValueError, match="event_cost and event_effect are required"):
        engine.evaluate(_draws())


def test_trajectory_returns_compartments():
    engine = _decay_engine()
    traj = engine.trajectory(pd.Series({"k": 0.1, "c": 0.0}), "s", n_points=50)
    assert list(traj.columns) == ["time", "alive", "dead"]
    assert len(traj) == 50
    # alive decays from 1 toward 0; dead rises; the pair conserves mass
    assert traj["alive"].iloc[0] == pytest.approx(1.0)
    assert traj["alive"].iloc[-1] < traj["alive"].iloc[0]
    assert (traj["alive"] + traj["dead"]).to_numpy() == pytest.approx(1.0, rel=1e-5)


# -- SEIR vaccination example -----------------------------------------------


def test_seir_vaccination_averts_infections_and_is_cost_effective():
    """The shipped example: vaccination cuts infections and buys QALYs at a positive ICER."""
    from seir_vaccination import BASE, INTERVENTIONS, STATES, seir

    engine = ODEModel(
        states=STATES, interventions=INTERVENTIONS, dynamics_and_rewards=seir,
        horizon=10.0, discount_rate=0.03,
    )
    base = pd.Series(BASE)
    no_vacc = engine.trajectory(base, "No vaccination")
    vacc = engine.trajectory(base, "Vaccination program")
    # vaccination all but eliminates the epidemic: far fewer ever infected (in R)
    assert vacc["R"].iloc[-1] < 0.05 * no_vacc["R"].iloc[-1]

    base_draws = pd.DataFrame([BASE], index=pd.RangeIndex(1, name="iteration"))
    table = icer_table(engine.evaluate(base_draws))
    # vaccination costs more but gains effect: a positive, finite ICER, no dominance
    assert table.loc["Vaccination program", "inc_cost"] > 0
    assert table.loc["Vaccination program", "inc_effect"] > 0
    assert 0 < table.loc["Vaccination program", "icer"] < 50_000
