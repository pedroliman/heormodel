"""Replicate the time-dependent (age-varying) cohort CEA of the Sick-Sicker model.

Reproduces the deterministic cost-effectiveness results of Alarid-Escudero and
others, "A Tutorial on Time-Dependent Cohort State-Transition Models in R Using a
Cost-Effectiveness Analysis Example," Medical Decision Making 2023;43(1):21-41.

This extends the introductory model in two ways. Background mortality now varies
by age: the Healthy-to-Dead rate follows a US life table, and the Sick and Sicker
states scale it by their hazard ratios, so `model` returns a per-cycle transition
array rather than one matrix. Transition rewards attach a one-time cost of dying,
a one-time cost of becoming Sick, and a disutility of onset to the flows between
states, not to the states themselves.

Run it with::

    uv run python examples/mdm_cohort_timedep.py

Outputs (written to ``examples/output/``):
    - ce_plane_timedep.png, ceac_timedep.png, frontier_timedep.png
    - run_report_timedep.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heormodel.cea import ceac, ceaf, icer_table
from heormodel.models import CohortSpec, MarkovModel
from heormodel.params import Beta, Gamma, LogNormal, ParameterSet
from heormodel.report import capture_run, plot_ce_plane, plot_ceac, plot_frontier
from heormodel.run import SeedManager, run_psa
from heormodel.voi import evpi, evppi_ranking

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("H", "S1", "S2", "D")
STRATEGIES = ("Standard of care", "Strategy A", "Strategy B", "Strategy AB")
N_CYCLES = 75  # ages 25 to 100, annual cycles
WTP = 100_000.0
N_SIM = 1_000

# US all-cause mortality rate by single year of age, 25 to 99 (period life
# table, total population). Used as the age-varying Healthy-to-Dead rate.
MORTALITY_BY_AGE = np.array([
    0.001014, 0.000999, 0.00107, 0.001087, 0.001162, 0.001167, 0.001213, 0.001289, 0.001331,
    0.001375, 0.00142, 0.00149, 0.00155, 0.001616, 0.001657, 0.001747, 0.001902, 0.002052,
    0.002173, 0.002395, 0.002559, 0.002807, 0.003023, 0.003349, 0.003712, 0.004085, 0.00449,
    0.004905, 0.005364, 0.005806, 0.006253, 0.006775, 0.007395, 0.007895, 0.008418, 0.008974,
    0.009666, 0.010456, 0.011384, 0.011838, 0.012667, 0.013593, 0.0147, 0.015732, 0.01734,
    0.018758, 0.020967, 0.022917, 0.024913, 0.026767, 0.029707, 0.032412, 0.035982, 0.039238,
    0.043595, 0.048727, 0.053735, 0.059911, 0.066618, 0.074051, 0.08219, 0.090754, 0.103968,
    0.115093, 0.124341, 0.137872, 0.154177, 0.172393, 0.1941, 0.212654, 0.243752, 0.259087,
    0.287781, 0.316429, 0.339149,
])

BASE = dict(
    r_HS1=0.15, r_S1H=0.5, r_S1S2=0.105, hr_S1=3.0, hr_S2=10.0, hr_S1S2_trtB=0.6,
    c_H=2000.0, c_S1=4000.0, c_S2=15000.0, c_trtA=12000.0, c_trtB=13000.0,
    u_H=1.0, u_S1=0.75, u_S2=0.5, u_trtA=0.95,
    du_HS1=0.01, ic_HS1=1000.0, ic_D=2000.0,
)


def rate_to_prob(rate: float | np.ndarray, t: float = 1.0) -> np.ndarray:
    """Convert an instantaneous rate to a per-cycle probability."""
    return 1.0 - np.exp(-np.asarray(rate) * t)


def model(params: pd.Series, strategy: str) -> CohortSpec:
    """Per-cycle transition array, state payoffs, and transition rewards."""
    p_HS1 = rate_to_prob(params["r_HS1"])
    p_S1H = rate_to_prob(params["r_S1H"])
    p_S1S2 = rate_to_prob(params["r_S1S2"])
    v_p_HD = rate_to_prob(MORTALITY_BY_AGE)  # age-varying, length N_CYCLES
    v_p_S1D = rate_to_prob(MORTALITY_BY_AGE * params["hr_S1"])
    v_p_S2D = rate_to_prob(MORTALITY_BY_AGE * params["hr_S2"])
    treats_b = strategy in ("Strategy B", "Strategy AB")
    p_prog = rate_to_prob(params["r_S1S2"] * params["hr_S1S2_trtB"]) if treats_b else p_S1S2

    P = np.zeros((N_CYCLES, 4, 4))
    P[:, 0, 0] = (1 - v_p_HD) * (1 - p_HS1)
    P[:, 0, 1] = (1 - v_p_HD) * p_HS1
    P[:, 0, 3] = v_p_HD
    P[:, 1, 0] = (1 - v_p_S1D) * p_S1H
    P[:, 1, 1] = (1 - v_p_S1D) * (1 - (p_S1H + p_prog))
    P[:, 1, 2] = (1 - v_p_S1D) * p_prog
    P[:, 1, 3] = v_p_S1D
    P[:, 2, 2] = 1 - v_p_S2D
    P[:, 2, 3] = v_p_S2D
    P[:, 3, 3] = 1.0

    add = {
        "Standard of care": 0.0,
        "Strategy A": params["c_trtA"],
        "Strategy B": params["c_trtB"],
        "Strategy AB": params["c_trtA"] + params["c_trtB"],
    }[strategy]
    cost = np.array([params["c_H"], params["c_S1"] + add, params["c_S2"] + add, 0.0])
    treats_a = strategy in ("Strategy A", "Strategy AB")
    u_s1 = params["u_trtA"] if treats_a else params["u_S1"]
    effect = np.array([params["u_H"], u_s1, params["u_S2"], 0.0])

    # transition rewards: cost of onset (H to S1), cost of dying, disutility of onset
    trans_cost = np.zeros((4, 4))
    trans_cost[0, 1] = params["ic_HS1"]
    trans_cost[0, 3] = trans_cost[1, 3] = trans_cost[2, 3] = params["ic_D"]
    trans_effect = np.zeros((4, 4))
    trans_effect[0, 1] = -params["du_HS1"]
    return CohortSpec(P, cost, effect, transition_cost=trans_cost,
                      transition_effect=trans_effect)


def parameters() -> ParameterSet:
    """PSA distributions from the tutorial's supplementary code."""
    return ParameterSet(
        {
            "r_HS1": Gamma(shape=30, scale=1 / 200),
            "r_S1H": Gamma(shape=60, scale=1 / 120),
            "r_S1S2": Gamma(shape=84, scale=1 / 800),
            "hr_S1": LogNormal(mu=np.log(3.0), sigma=0.01),
            "hr_S2": LogNormal(mu=np.log(10.0), sigma=0.02),
            "hr_S1S2_trtB": LogNormal(mu=np.log(0.6), sigma=0.02),
            "c_H": Gamma(shape=100, scale=20.0),
            "c_S1": Gamma(shape=177.8, scale=22.5),
            "c_S2": Gamma(shape=225, scale=66.7),
            "c_trtA": Gamma(shape=73.5, scale=163.3),
            "c_trtB": Gamma(shape=86.2, scale=150.8),
            "u_H": Beta(200, 3),
            "u_S1": Beta(130, 45),
            "u_S2": Beta(230, 230),
            "u_trtA": Beta(300, 15),
            "du_HS1": Beta(11, 1088),
            "ic_HS1": Gamma(shape=100, scale=10.0),
            "ic_D": Gamma(shape=100, scale=20.0),
        }
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seeds = SeedManager(20260705)
    engine = MarkovModel(
        states=STATES, strategies=STRATEGIES, transitions_and_rewards=model, n_cycles=N_CYCLES,
        initial_state="H", discount_rate=0.03,
        cycle_correction="simpson",
    )

    base_draws = pd.DataFrame([BASE], index=pd.RangeIndex(1, name="iteration"))
    print("Deterministic base-case CEA (reproduces the published table):")
    print(icer_table(engine.evaluate(base_draws)).round(2).to_string())

    draws = parameters().sample(N_SIM, seed=seeds.generator())
    outcomes = run_psa(engine, draws).outcomes
    print("\nProbabilistic CEA (mean over draws):")
    print(icer_table(outcomes).round(2).to_string())
    print(f"\nEVPI at WTP {WTP:,.0f}: {evpi(outcomes, WTP):,.1f}")
    print("\nEVPPI ranking (top parameters):")
    print(evppi_ranking(outcomes, draws, WTP).round(1).head().to_string())

    grid = np.linspace(0, 200_000, 41)
    plot_ce_plane(outcomes, comparator="Standard of care", wtp=WTP).figure.savefig(
        OUT / "ce_plane_timedep.png", dpi=150, bbox_inches="tight"
    )
    plot_ceac(ceac(outcomes, grid), ceaf_df=ceaf(outcomes, grid)).figure.savefig(
        OUT / "ceac_timedep.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(outcomes).figure.savefig(
        OUT / "frontier_timedep.png", dpi=150, bbox_inches="tight"
    )

    record = capture_run(
        seed=seeds,
        outcomes=outcomes,
        draw_sources=dict.fromkeys(draws.columns, "Sick-Sicker time-dependent tutorial (MDM 2023)"),
        note=(
            "Age-dependent cohort Sick-Sicker model with US life-table mortality "
            "and transition rewards, 75 annual cycles, Simpson's 1/3 correction."
        ),
    )
    (OUT / "run_report_timedep.md").write_text(
        record.to_markdown("heormodel time-dependent Sick-Sicker run report")
    )
    print(f"\nWrote plots and run report to {OUT}/")


if __name__ == "__main__":
    main()
