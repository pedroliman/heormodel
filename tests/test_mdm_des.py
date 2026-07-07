"""Cross-validate the continuous-time Sick-Sicker replication against a closed form.

The model mirrors ``examples/mdm_des.py`` with two simplifications that make it
a continuous-time Markov chain: the Weibull progression shape is 1 (a constant
hazard equal to the proportional-hazards scale) and background mortality is a
single constant rate. Expected discounted costs and QALYs, including the
one-time transition rewards, then solve ``(d I - Q) v = r`` on the alive states
for generator ``Q``, with each transition reward entering ``r`` as its rate
times the reward. The test asserts the continuous-clock engine, the `LifeTable`
sampler, `payoffs`, and `transition_payoffs` jointly reproduce that solution
within Monte Carlo error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heormodel.models import LifeTable, MicrosimModel
from heormodel.params import single_draw
from heormodel.run import SeedManager

MORTALITY_RATE = 0.01
MORTALITY = LifeTable(ages=[0.0], rates=[MORTALITY_RATE])
DISCOUNT = 0.03
HORIZON = 1_000.0  # long enough that discounting makes the truncation negligible

BASE = dict(
    r_HS1=0.15, r_S1H=0.5, r_S1S2_scale=0.09, r_S1S2_shape=1.0,
    hr_S1=3.0, hr_S2=10.0, hr_S1S2_trtB=0.6,
    c_H=2000.0, c_S1=4000.0, c_S2=15000.0, c_trtA=12000.0, c_trtB=13000.0,
    u_H=1.0, u_S1=0.75, u_S2=0.5, u_trtA=0.95,
    du_HS1=0.01, ic_HS1=1000.0, ic_D=2000.0,
)


def hazards(p, state, attrs, rng):
    n = len(state)
    times = np.full((n, 4), np.inf)
    age = attrs["time"].to_numpy()
    h = state == 0
    if h.any():
        times[h, 1] = rng.exponential(1.0 / p["r_HS1"], int(h.sum()))
        times[h, 3] = MORTALITY.sample_time_to_death(rng, age[h])
    s1 = state == 1
    if s1.any():
        times[s1, 0] = rng.exponential(1.0 / p["r_S1H"], int(s1.sum()))
        scale_ph = p["r_S1S2_scale"] * (p["hr_S1S2_trtB"] if p["trtB"] else 1.0)
        aft_scale = scale_ph ** (-1.0 / p["r_S1S2_shape"])
        times[s1, 2] = aft_scale * rng.weibull(p["r_S1S2_shape"], int(s1.sum()))
        times[s1, 3] = MORTALITY.sample_time_to_death(rng, age[s1], hazard_ratio=p["hr_S1"])
    s2 = state == 2
    if s2.any():
        times[s2, 3] = MORTALITY.sample_time_to_death(rng, age[s2], hazard_ratio=p["hr_S2"])
    return times


def payoffs(p, state, attrs):
    n = len(state)
    cost = np.zeros(n)
    util = np.zeros(n)
    on_a, on_b = bool(p["trtA"]), bool(p["trtB"])
    tx_cost = on_a * p["c_trtA"] + on_b * p["c_trtB"]
    h = state == 0
    cost[h], util[h] = p["c_H"], p["u_H"]
    s1 = state == 1
    cost[s1] = p["c_S1"] + tx_cost
    util[s1] = p["u_trtA"] if on_a else p["u_S1"]
    s2 = state == 2
    cost[s2] = p["c_S2"] + tx_cost
    util[s2] = p["u_S2"]
    return cost, util


def transition_payoffs(p, state_from, state_to, attrs):
    n = len(state_from)
    cost = np.zeros(n)
    eff = np.zeros(n)
    onset = (state_from == 0) & (state_to == 1)
    cost[onset] = p["ic_HS1"]
    eff[onset] = -p["du_HS1"]
    cost[state_to == 3] += p["ic_D"]
    return cost, eff


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
    for (i, j), rate in rates.items():
        if j < 3:
            q[i, j] += rate
        q[i, i] -= rate
    tx_cost = trt_a * p["c_trtA"] + trt_b * p["c_trtB"]
    cost_rate = np.array([
        p["c_H"] + rates[(0, 1)] * p["ic_HS1"] + rates[(0, 3)] * p["ic_D"],
        p["c_S1"] + tx_cost + rates[(1, 3)] * p["ic_D"],
        p["c_S2"] + tx_cost + rates[(2, 3)] * p["ic_D"],
    ])
    eff_rate = np.array([
        p["u_H"] - rates[(0, 1)] * p["du_HS1"],
        p["u_trtA"] if trt_a else p["u_S1"],
        p["u_S2"],
    ])
    lhs = DISCOUNT * np.eye(3) - q
    cost = float(np.linalg.solve(lhs, cost_rate)[0])
    eff = float(np.linalg.solve(lhs, eff_rate)[0])
    return cost, eff


@pytest.mark.parametrize("strategy,overrides", [
    ("Standard of care", {"trtA": 0.0, "trtB": 0.0}),
    ("Strategy AB", {"trtA": 1.0, "trtB": 1.0}),
])
def test_engine_matches_ctmc_closed_form(strategy, overrides):
    engine = MicrosimModel(
        states=("H", "S1", "S2", "D"),
        clock="continuous",
        hazards=hazards,
        payoffs=payoffs,
        transition_payoffs=transition_payoffs,
        population=120_000,
        strategies={strategy: overrides},
        horizon=HORIZON,
        discount_rate=DISCOUNT,
        seed_manager=SeedManager(13),
    )
    got = engine.evaluate(single_draw(BASE)).summary().loc[strategy]
    want_cost, want_eff = ctmc_value(BASE, trt_a=bool(overrides["trtA"]),
                                     trt_b=bool(overrides["trtB"]))
    assert got["cost"] == pytest.approx(want_cost, rel=0.01)
    assert got["qaly"] == pytest.approx(want_eff, rel=0.01)


def test_common_random_numbers_tie_equivalent_dynamics():
    # A shares SoC's transition dynamics; with common random numbers their
    # event histories are identical, so survival curves coincide exactly.
    engine = MicrosimModel(
        states=("H", "S1", "S2", "D"),
        clock="continuous",
        hazards=hazards,
        payoffs=payoffs,
        transition_payoffs=transition_payoffs,
        population=2_000,
        strategies={
            "Standard of care": {"trtA": 0.0, "trtB": 0.0},
            "Strategy A": {"trtA": 1.0, "trtB": 0.0},
        },
        horizon=75.0,
        discount_rate=DISCOUNT,
        seed_manager=SeedManager(21),
    )
    _, events = engine.evaluate(single_draw(BASE), trace="events")
    soc = events[events["strategy"] == "Standard of care"].drop(columns="strategy")
    a = events[events["strategy"] == "Strategy A"].drop(columns="strategy")
    pd.testing.assert_frame_equal(soc.reset_index(drop=True), a.reset_index(drop=True))
