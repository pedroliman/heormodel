"""Cross-validate the continuous-time Sick-Sicker replication against a closed form.

The test exercises the shipped building blocks in ``examples/mdm_des``: the
engine that `build_engine` assembles and the sojourn accrual in
`with_transition_costs_and_utilities`. It uses two simplifications that make the
model a continuous-time Markov chain: the Weibull progression shape is 1 (a
constant hazard equal to the proportional-hazards scale) and background
mortality is a single constant rate. Expected discounted costs and QALYs,
including the transition amounts, then solve ``(d I - Q) v = r`` on the alive
states for generator ``Q``. Each transition amount accrues over the sojourn that
ends in it, the companion's convention, so it enters ``r`` as the amount times
its exit rate over the state's total exit rate: its expected annual flow while
in the state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mdm_des.model import build_engine
from mdm_des.transitions import with_transition_costs_and_utilities

from heormodel.models import LifeTable
from heormodel.params import single_draw
from heormodel.run import SeedManager

MORTALITY_RATE = 0.01
DISCOUNT = 0.03
HORIZON = 1_000.0  # long enough that discounting makes the truncation negligible
STATES = ("H", "S1", "S2", "D")
LIFE_TABLE = LifeTable(ages=[0.0], rates=[MORTALITY_RATE])

BASE = dict(
    r_HS1=0.15, r_S1H=0.5, r_S1S2_scale=0.09, r_S1S2_shape=1.0,
    hr_S1=3.0, hr_S2=10.0, hr_S1S2_trtB=0.6,
    c_H=2000.0, c_S1=4000.0, c_S2=15000.0, c_trtA=12000.0, c_trtB=13000.0,
    u_H=1.0, u_S1=0.75, u_S2=0.5,
    du_HS1=0.01, ic_HS1=1000.0, ic_D=2000.0,
)


def _engine(strategies: dict, population: int, seed: int):
    return build_engine(
        life_table=LIFE_TABLE, states=STATES, strategies=strategies,
        age_start=0.0, horizon=HORIZON, discount_rate=DISCOUNT,
        population=population, seed_manager=SeedManager(seed),
    )


def ctmc_value(p: dict, *, trt_a: bool, trt_b: bool) -> tuple[float, float]:
    """Expected discounted cost and QALYs starting Healthy, by linear algebra."""
    lam_prog = p["r_S1S2_scale"] * (p["hr_S1S2_trtB"] if trt_b else 1.0)
    m = MORTALITY_RATE
    rates = {
        (0, 1): p["r_HS1"], (0, 3): m,
        (1, 0): p["r_S1H"], (1, 2): lam_prog, (1, 3): p["hr_S1"] * m,
        (2, 3): p["hr_S2"] * m,
    }
    q = np.zeros((3, 3))
    exit_rate = np.zeros(3)
    for (i, j), rate in rates.items():
        exit_rate[i] += rate
        if j < 3:
            q[i, j] += rate
        q[i, i] -= rate
    tx_cost = trt_a * p["c_trtA"] + trt_b * p["c_trtB"]
    # A transition amount accrues over the sojourn, so it enters the state's flow
    # as the amount times its transition rate over the state's total exit rate:
    # its expected annual flow while occupying the state.
    reward_cost = np.array([
        rates[(0, 1)] * p["ic_HS1"] + rates[(0, 3)] * p["ic_D"],
        rates[(1, 3)] * p["ic_D"],
        rates[(2, 3)] * p["ic_D"],
    ]) / exit_rate
    reward_eff = np.array([rates[(0, 1)] * (-p["du_HS1"]), 0.0, 0.0]) / exit_rate
    cost_rate = np.array([p["c_H"], p["c_S1"] + tx_cost, p["c_S2"] + tx_cost]) + reward_cost
    eff_rate = np.array([
        p["u_H"],
        (p["u_S1"] + 0.20) if trt_a else p["u_S1"],
        p["u_S2"],
    ]) + reward_eff
    lhs = DISCOUNT * np.eye(3) - q
    cost = float(np.linalg.solve(lhs, cost_rate)[0])
    eff = float(np.linalg.solve(lhs, eff_rate)[0])
    return cost, eff


@pytest.mark.parametrize("strategy,overrides", [
    ("Standard of care", {"trtA": 0.0, "trtB": 0.0}),
    ("Strategy AB", {"trtA": 1.0, "trtB": 1.0}),
])
def test_engine_matches_ctmc_closed_form(strategy, overrides):
    engine = _engine({strategy: overrides}, population=120_000, seed=13)
    draws = single_draw(BASE)
    outcomes, events = engine.evaluate(draws, trace="events")
    outcomes = with_transition_costs_and_utilities(
        outcomes, events, draws, n_individuals=120_000, discount_rate=DISCOUNT
    )
    got = outcomes.summary().loc[strategy]
    want_cost, want_eff = ctmc_value(BASE, trt_a=bool(overrides["trtA"]),
                                     trt_b=bool(overrides["trtB"]))
    assert got["cost"] == pytest.approx(want_cost, rel=0.01)
    assert got["qaly"] == pytest.approx(want_eff, rel=0.01)


def test_common_random_numbers_tie_equivalent_dynamics():
    # A shares SoC's transition dynamics; with common random numbers their event
    # histories are identical, so survival curves coincide exactly.
    engine = _engine(
        {
            "Standard of care": {"trtA": 0.0, "trtB": 0.0},
            "Strategy A": {"trtA": 1.0, "trtB": 0.0},
        },
        population=2_000,
        seed=21,
    )
    _, events = engine.evaluate(single_draw(BASE), trace="events")
    soc = events[events["strategy"] == "Standard of care"].drop(columns="strategy")
    a = events[events["strategy"] == "Strategy A"].drop(columns="strategy")
    pd.testing.assert_frame_equal(soc.reset_index(drop=True), a.reset_index(drop=True))
