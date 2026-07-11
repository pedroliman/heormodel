"""Replicate the introductory cohort state-transition CEA of the Sick-Sicker model.

Reproduces the deterministic cost-effectiveness results of Alarid-Escudero and
others, "An Introductory Tutorial on Cohort State-Transition Models in R Using a
Cost-Effectiveness Analysis Example," Medical Decision Making 2023;43(1):3-20,
then runs the probabilistic sensitivity analysis on the same parameters through
the analysis layer.

The Sick-Sicker model has four states (Healthy, Sick, Sicker, Dead) over 75
annual cycles from age 25. Four strategies compare standard of care with a
treatment A that improves the quality of life of the Sick, a treatment B that
slows progression from Sick to Sicker, and their combination AB. `model`
constructs each strategy's transition matrix and per-state payoffs from a
parameter row; `MarkovModel` sweeps the cohort and emits `Outcomes`.

Run it with::

    uv run python examples/mdm_cohort.py

Outputs (written to ``examples/output/``):
    - ce_plane_cohort.png, ceac_cohort.png, frontier_cohort.png
    - run_report_cohort.md
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
WTP = 100_000.0  # willingness-to-pay threshold used in the tutorial
N_SIM = 1_000

# Deterministic base-case parameters (Table 1 of the tutorial).
BASE = dict(
    r_HD=0.002, r_HS1=0.15, r_S1H=0.5, r_S1S2=0.105, hr_S1=3.0, hr_S2=10.0,
    hr_S1S2_trtB=0.6, c_H=2000.0, c_S1=4000.0, c_S2=15000.0, c_trtA=12000.0,
    c_trtB=13000.0, u_H=1.0, u_S1=0.75, u_S2=0.5, u_trtA=0.95,
)


def rate_to_prob(rate: float | np.ndarray, t: float = 1.0) -> np.ndarray:
    """Convert an instantaneous rate to a per-cycle probability."""
    return 1.0 - np.exp(-np.asarray(rate) * t)


def model(params: pd.Series, strategy: str) -> CohortSpec:
    """Transition matrix and per-state payoffs for one strategy and draw."""
    p_HS1 = rate_to_prob(params["r_HS1"])
    p_S1H = rate_to_prob(params["r_S1H"])
    p_S1S2 = rate_to_prob(params["r_S1S2"])
    p_HD = rate_to_prob(params["r_HD"])
    p_S1D = rate_to_prob(params["r_HD"] * params["hr_S1"])
    p_S2D = rate_to_prob(params["r_HD"] * params["hr_S2"])
    # treatment B slows progression from Sick to Sicker
    treats_b = strategy in ("Strategy B", "Strategy AB")
    p_prog = rate_to_prob(params["r_S1S2"] * params["hr_S1S2_trtB"]) if treats_b else p_S1S2

    P = np.zeros((4, 4))
    P[0, 0] = (1 - p_HD) * (1 - p_HS1)
    P[0, 1] = (1 - p_HD) * p_HS1
    P[0, 3] = p_HD
    P[1, 0] = (1 - p_S1D) * p_S1H
    P[1, 1] = (1 - p_S1D) * (1 - (p_S1H + p_prog))
    P[1, 2] = (1 - p_S1D) * p_prog
    P[1, 3] = p_S1D
    P[2, 2] = 1 - p_S2D
    P[2, 3] = p_S2D
    P[3, 3] = 1.0

    # treatment cost is added in the Sick and Sicker states
    add = {
        "Standard of care": 0.0,
        "Strategy A": params["c_trtA"],
        "Strategy B": params["c_trtB"],
        "Strategy AB": params["c_trtA"] + params["c_trtB"],
    }[strategy]
    cost = np.array([params["c_H"], params["c_S1"] + add, params["c_S2"] + add, 0.0])
    # treatment A raises the Sick-state utility
    treats_a = strategy in ("Strategy A", "Strategy AB")
    u_s1 = params["u_trtA"] if treats_a else params["u_S1"]
    effect = np.array([params["u_H"], u_s1, params["u_S2"], 0.0])
    return CohortSpec(P, cost, effect)


def parameters() -> ParameterSet:
    """PSA distributions from the tutorial's supplementary code."""
    return ParameterSet(
        {
            "r_HD": Gamma(shape=20, scale=1 / 10000),
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

    # Deterministic base case reproduces the tutorial's CEA table.
    base_draws = pd.DataFrame([BASE], index=pd.RangeIndex(1, name="iteration"))
    print("Deterministic base-case CEA (reproduces the published table):")
    print(icer_table(engine.evaluate(base_draws)).round(2).to_string())

    # Probabilistic sensitivity analysis on the same model.
    draws = parameters().sample(N_SIM, seed=seeds.generator())
    outcomes = run_psa(engine, draws).outcomes
    print("\nProbabilistic CEA (mean over draws):")
    print(icer_table(outcomes).round(2).to_string())
    print(f"\nEVPI at WTP {WTP:,.0f}: {evpi(outcomes, WTP):,.1f}")
    print("\nEVPPI ranking (top parameters):")
    print(evppi_ranking(outcomes, draws, WTP).round(1).head().to_string())

    grid = np.linspace(0, 200_000, 41)
    plot_ce_plane(outcomes, comparator="Standard of care", wtp=WTP).figure.savefig(
        OUT / "ce_plane_cohort.png", dpi=150, bbox_inches="tight"
    )
    plot_ceac(ceac(outcomes, grid), ceaf_df=ceaf(outcomes, grid)).figure.savefig(
        OUT / "ceac_cohort.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(outcomes).figure.savefig(
        OUT / "frontier_cohort.png", dpi=150, bbox_inches="tight"
    )

    record = capture_run(
        seed=seeds,
        outcomes=outcomes,
        draw_sources=dict.fromkeys(draws.columns, "Sick-Sicker tutorial (MDM 2023)"),
        note=(
            f"Cohort state-transition Sick-Sicker model, {N_CYCLES} annual cycles, "
            "Simpson's 1/3 within-cycle correction, four strategies."
        ),
    )
    (OUT / "run_report_cohort.md").write_text(
        record.to_markdown("heormodel cohort Sick-Sicker run report")
    )
    print(f"\nWrote plots and run report to {OUT}/")


if __name__ == "__main__":
    main()
