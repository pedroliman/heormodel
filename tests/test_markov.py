"""Tests for the cohort state-transition (Markov) engine.

Two validation checks anchor the file: a two-state closed form the engine must
reproduce exactly, and the published deterministic ICERs of the DARTH
introductory Sick-Sicker cost-effectiveness example (Alarid-Escudero and others,
Medical Decision Making 2023). The rest cover per-cycle (age-varying)
transitions, transition rewards, the within-cycle correction, and the output
contract.
"""

import numpy as np
import pandas as pd
import pytest

from heormodel.cea import icer_table
from heormodel.models import Intervention, ModelEngine, Outcomes
from heormodel.models.markov import CohortSpec, MarkovModel, gen_wcc


def _draws(n_iter=1, **cols):
    if not cols:
        cols = {"z": np.zeros(n_iter)}
    return pd.DataFrame(cols, index=pd.RangeIndex(n_iter, name="iteration"))


# -- within-cycle correction ------------------------------------------------


def test_gen_wcc_variants():
    assert gen_wcc(4, "none").tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert gen_wcc(4, "half_cycle").tolist() == [0.5, 1.0, 1.0, 1.0, 0.5]
    # matches the DARTH gen_wcc reference: endpoints 1/3, interior alternating
    simpson = gen_wcc(4, "simpson")
    assert simpson.tolist() == pytest.approx([1 / 3, 2 / 3, 4 / 3, 2 / 3, 1 / 3])


def test_gen_wcc_rejects_bad_input():
    with pytest.raises(ValueError):
        gen_wcc(0, "simpson")
    with pytest.raises(ValueError):
        gen_wcc(4, "quadratic")


# -- closed-form validation -------------------------------------------------


def test_two_state_closed_form():
    """Alive/Dead survival model matches the analytic discounted sum."""
    p, cost, rate, horizon = 0.1, 1000.0, 0.03, 20

    def model(params, intervention):
        P = np.array([[1 - p, p], [0.0, 1.0]])
        return CohortSpec(P, np.array([cost, 0.0]), np.array([1.0, 0.0]))

    engine = MarkovModel(
        states=("alive", "dead"), interventions=("s",), transitions_and_rewards=model,
        n_cycles=horizon, discount_rate=rate, cycle_correction="none",
    )
    got = engine.evaluate(_draws()).summary().loc["s"]
    # occupancy of alive at cycle t is (1-p)^t; accrue discounted, unit weights
    t = np.arange(horizon + 1)
    alive = (1 - p) ** t
    disc = (1 + rate) ** (-t)
    assert got["qaly"] == pytest.approx(float(alive @ disc))
    assert got["cost"] == pytest.approx(float(alive * cost @ disc))


def test_half_cycle_correction_changes_result():
    def model(params, intervention):
        P = np.array([[0.9, 0.1], [0.0, 1.0]])
        return CohortSpec(P, np.array([100.0, 0.0]), np.array([1.0, 0.0]))

    common = dict(
        states=("a", "d"), interventions=("s",), transitions_and_rewards=model, n_cycles=10
    )
    none = MarkovModel(**common, cycle_correction="none")
    half = MarkovModel(**common, cycle_correction="half_cycle")
    q_none = none.evaluate(_draws()).summary().loc["s", "qaly"]
    q_half = half.evaluate(_draws()).summary().loc["s", "qaly"]
    # half-cycle correction halves the first cycle's full-occupancy contribution
    assert q_half < q_none


# -- published Sick-Sicker replication --------------------------------------

_SICK_SICKER = dict(
    r_HD=0.002, r_HS1=0.15, r_S1H=0.5, r_S1S2=0.105, hr_S1=3.0, hr_S2=10.0,
    hr_S1S2_trtB=0.6, c_H=2000.0, c_S1=4000.0, c_S2=15000.0, c_trtA=12000.0,
    c_trtB=13000.0, u_H=1.0, u_S1=0.75, u_S2=0.5, u_trtA=0.95,
)
_INTERVENTIONS = ("Standard of care", "Intervention A", "Intervention B", "Intervention AB")


def _sick_sicker_model(params, intervention):
    def r2p(r):
        return 1.0 - np.exp(-r)

    p_HS1, p_S1H = r2p(params["r_HS1"]), r2p(params["r_S1H"])
    p_S1S2 = r2p(params["r_S1S2"])
    p_HD = r2p(params["r_HD"])
    p_S1D = r2p(params["r_HD"] * params["hr_S1"])
    p_S2D = r2p(params["r_HD"] * params["hr_S2"])
    p_prog = r2p(params["r_S1S2"] * params["hr_S1S2_trtB"]) if "B" in intervention else p_S1S2
    P = np.zeros((4, 4))
    P[0, 0], P[0, 1], P[0, 3] = (1 - p_HD) * (1 - p_HS1), (1 - p_HD) * p_HS1, p_HD
    P[1, 0], P[1, 3] = (1 - p_S1D) * p_S1H, p_S1D
    P[1, 1] = (1 - p_S1D) * (1 - p_S1H - p_prog)
    P[1, 2] = (1 - p_S1D) * p_prog
    P[2, 2], P[2, 3] = 1 - p_S2D, p_S2D
    P[3, 3] = 1.0
    add = {"Standard of care": 0.0, "Intervention A": params["c_trtA"],
           "Intervention B": params["c_trtB"],
           "Intervention AB": params["c_trtA"] + params["c_trtB"]}[intervention]
    cost = np.array([params["c_H"], params["c_S1"] + add, params["c_S2"] + add, 0.0])
    on_trtA = intervention in ("Intervention A", "Intervention AB")
    u_s1 = params["u_trtA"] if on_trtA else params["u_S1"]
    util = np.array([params["u_H"], u_s1, params["u_S2"], 0.0])
    return CohortSpec(P, cost, util)


def test_sick_sicker_matches_published_icers():
    """Reproduce the published deterministic CEA table of the intro tutorial."""
    engine = MarkovModel(
        states=("H", "S1", "S2", "D"), interventions=_INTERVENTIONS,
        transitions_and_rewards=_sick_sicker_model, n_cycles=75, initial_state="H",
        cycle_correction="simpson",
    )
    draws = pd.DataFrame([_SICK_SICKER], index=pd.RangeIndex(1, name="iteration"))
    table = icer_table(engine.evaluate(draws))
    soc = table.loc["Standard of care"]
    assert soc["cost"] == pytest.approx(151_580, abs=5)
    assert soc["effect"] == pytest.approx(20.711, abs=1e-2)
    assert table.loc["Intervention A", "status"] == "D"
    assert table.loc["Intervention B", "icer"] == pytest.approx(72_988, abs=5)
    assert table.loc["Intervention AB", "icer"] == pytest.approx(125_764, abs=5)


# -- age-varying transitions ------------------------------------------------


def test_per_cycle_transition_differs_from_constant():
    n_cycles = 30
    rising = np.linspace(0.01, 0.2, n_cycles)  # mortality climbs with cycle

    def model_varying(params, intervention):
        P = np.zeros((n_cycles, 2, 2))
        P[:, 0, 1] = rising
        P[:, 0, 0] = 1 - rising
        P[:, 1, 1] = 1.0
        return CohortSpec(P, np.array([100.0, 0.0]), np.array([1.0, 0.0]))

    def model_constant(params, intervention):
        P = np.array([[1 - 0.01, 0.01], [0.0, 1.0]])
        return CohortSpec(P, np.array([100.0, 0.0]), np.array([1.0, 0.0]))

    common = dict(states=("a", "d"), interventions=("s",), n_cycles=n_cycles,
                  cycle_correction="none")
    q_var = MarkovModel(transitions_and_rewards=model_varying, **common).evaluate(
        _draws()).summary().loc["s", "qaly"]
    q_con = MarkovModel(transitions_and_rewards=model_constant, **common).evaluate(
        _draws()).summary().loc["s", "qaly"]
    assert q_var < q_con  # rising mortality accrues fewer QALYs


# -- transition rewards -----------------------------------------------------


def test_transition_reward_adds_one_time_cost():
    """A one-time cost of dying is charged once per death, on the transition."""
    death_cost = 5000.0

    def model(params, intervention):
        P = np.array([[0.8, 0.2], [0.0, 1.0]])
        tc = np.zeros((2, 2))
        tc[0, 1] = death_cost
        return CohortSpec(P, np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                          transition_cost=tc)

    engine = MarkovModel(
        states=("a", "d"), interventions=("s",), transitions_and_rewards=model, n_cycles=40,
        discount_rate=0.0, cycle_correction="none",
    )
    cost = engine.evaluate(_draws()).summary().loc["s", "cost"]
    # everyone dies within 40 cycles from 'a'; total death cost approaches 5000
    assert cost == pytest.approx(death_cost, abs=5.0)


# -- occupancy trace --------------------------------------------------------


def test_trace_shape_and_occupancy():
    """trace returns a cycle-indexed occupancy distribution per state."""
    def model(params, intervention):
        p_die = 0.1
        transition = np.array([[1 - p_die, p_die], [0.0, 1.0]])
        return CohortSpec(transition, np.array([params["c"], 0.0]), np.array([1.0, 0.0]))

    engine = MarkovModel(
        states=("alive", "dead"), interventions=("care",),
        transitions_and_rewards=model, n_cycles=5,
    )
    trace = engine.trace(pd.Series({"c": 1000.0}), "care")
    assert list(trace.columns) == ["cycle", "alive", "dead"]
    assert trace["cycle"].tolist() == [0, 1, 2, 3, 4, 5]
    # occupancy is a distribution every cycle
    assert trace[["alive", "dead"]].sum(axis=1).to_numpy() == pytest.approx(1.0)
    # alive decays geometrically at the death probability
    assert trace["alive"].to_numpy() == pytest.approx(0.9 ** np.arange(6))


def test_trace_reflects_the_intervention():
    """Branching the model on the intervention name changes the trace."""
    def model(params, intervention):
        p_die = 0.3 if intervention == "aggressive" else 0.1
        transition = np.array([[1 - p_die, p_die], [0.0, 1.0]])
        return CohortSpec(transition, np.zeros(2), np.array([1.0, 0.0]))

    engine = MarkovModel(
        states=("alive", "dead"), interventions=("mild", "aggressive"),
        transitions_and_rewards=model, n_cycles=4,
    )
    mild = engine.trace(pd.Series(dtype=float), "mild")
    aggressive = engine.trace(pd.Series(dtype=float), "aggressive")
    # the higher-mortality arm has fewer alive at every cycle after the first
    assert (aggressive["alive"][1:].to_numpy() < mild["alive"][1:].to_numpy()).all()


def test_trace_unknown_intervention_rejected():
    def model(params, intervention):
        return CohortSpec(np.eye(2), np.zeros(2), np.zeros(2))

    engine = MarkovModel(
        states=("a", "d"), interventions=("s",), transitions_and_rewards=model, n_cycles=3,
    )
    with pytest.raises(KeyError, match="Unknown intervention"):
        engine.trace(pd.Series(dtype=float), "missing")


# -- contract and validation ------------------------------------------------


def test_engine_satisfies_protocol_and_contract():
    def model(params, intervention):
        P = np.array([[0.9, 0.1], [0.0, 1.0]])
        return CohortSpec(P, np.array([params["c"], 0.0]), np.array([1.0, 0.0]))

    engine = MarkovModel(
        states=("a", "d"), interventions=("x", "y"), transitions_and_rewards=model, n_cycles=5,
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
        P = np.array([[0.9, 0.1], [0.0, 1.0]])
        return CohortSpec(P, np.array([params["c"], 0.0]), np.array([1.0, 0.0]))

    engine = MarkovModel(
        states=("a", "d"),
        interventions=["x", Intervention("y", is_comparator=True)],
        transitions_and_rewards=model,
        n_cycles=5,
    )
    out = engine.evaluate(_draws(1, c=[1.0]))
    assert out.comparator == "y"


def test_start_distribution_forms():
    def model(params, intervention):
        P = np.eye(3)  # absorbing everywhere: trace stays at start
        return CohortSpec(P, np.zeros(3), np.array([1.0, 0.0, 0.0]))

    engine = MarkovModel(
        states=("a", "b", "c"), interventions=("s",), transitions_and_rewards=model, n_cycles=3,
        initial_state={"a": 0.6, "b": 0.4}, discount_rate=0.0, cycle_correction="none",
    )
    # only state 'a' has unit utility; with 60% starting there, each of 4 cycles
    # contributes 0.6
    assert engine.evaluate(_draws()).summary().loc["s", "qaly"] == pytest.approx(2.4)


def test_bad_start_rejected():
    def model(params, intervention):
        return CohortSpec(np.eye(2), np.zeros(2), np.zeros(2))

    with pytest.raises(ValueError, match="sum to 1"):
        MarkovModel(states=("a", "d"), interventions=("s",), transitions_and_rewards=model,
                           n_cycles=3, initial_state=[0.7, 0.7])


def _absorbing(params, intervention):
    return CohortSpec(np.eye(2), np.zeros(2), np.zeros(2))


def test_unknown_initial_state_label_rejected():
    with pytest.raises(ValueError, match="Unknown initial_state"):
        MarkovModel(states=("a", "d"), interventions=("s",), transitions_and_rewards=_absorbing,
                    n_cycles=3, initial_state="z")


def test_unknown_initial_state_in_mapping_rejected():
    with pytest.raises(ValueError, match="Unknown initial_state"):
        MarkovModel(states=("a", "d"), interventions=("s",), transitions_and_rewards=_absorbing,
                    n_cycles=3, initial_state={"a": 0.5, "z": 0.5})


def test_initial_state_array_wrong_length_rejected():
    with pytest.raises(ValueError, match="must have length"):
        MarkovModel(states=("a", "d"), interventions=("s",), transitions_and_rewards=_absorbing,
                    n_cycles=3, initial_state=[1.0, 0.0, 0.0])


def test_bad_transition_shape_rejected():
    def model(params, intervention):
        return CohortSpec(np.zeros((3, 3)), np.zeros(2), np.zeros(2))

    engine = MarkovModel(states=("a", "d"), interventions=("s",), transitions_and_rewards=model,
                                n_cycles=3)
    with pytest.raises(ValueError, match="shape"):
        engine.evaluate(_draws())


def test_rows_must_sum_to_one():
    def model(params, intervention):
        P = np.array([[0.5, 0.2], [0.0, 1.0]])  # first row sums to 0.7
        return CohortSpec(P, np.zeros(2), np.zeros(2))

    engine = MarkovModel(states=("a", "d"), interventions=("s",), transitions_and_rewards=model,
                                n_cycles=3)
    with pytest.raises(ValueError, match="sum to 1"):
        engine.evaluate(_draws())


def test_empty_draws_rejected():
    def model(params, intervention):
        return CohortSpec(np.eye(2), np.zeros(2), np.zeros(2))

    engine = MarkovModel(states=("a", "d"), interventions=("s",), transitions_and_rewards=model,
                                n_cycles=3)
    with pytest.raises(ValueError, match="empty"):
        engine.evaluate(pd.DataFrame(index=pd.RangeIndex(0, name="iteration")))

