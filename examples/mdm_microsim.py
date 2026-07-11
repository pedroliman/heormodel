"""Replicate the Sick-Sicker microsimulation of Krijkamp and others (2018).

Reproduces the extended (history-dependent) microsimulation of Krijkamp and
others, "Microsimulation Modeling for Health Decision Sciences Using R: A
Tutorial," Medical Decision Making 2018;38(3):400-422, Table 3.

Two features make this an individual-level model rather than a cohort: mortality
in the Sick and Sicker states rises with the number of consecutive years spent
sick, and a per-individual effect modifier scales the treatment utility. The
first needs a counter that spans the Sick and Sicker states together, which the
engine supplies through ``duration_groups``. The counter is 0 on the first sick
cycle, so mortality reads it as ``dur + 1`` and utility reads it directly, the
same offset the source tutorial uses.

Run it with::

    uv run python examples/mdm_microsim.py

Outputs (written to ``examples/output/``):
    - ce_plane_microsim_mdm.png, frontier_microsim_mdm.png
    - run_report_microsim_mdm.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heormodel.cea import icer_table
from heormodel.models import MicrosimModel
from heormodel.report import capture_run, plot_ce_plane, plot_frontier
from heormodel.run import SeedManager, run_psa

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("H", "S1", "S2", "D")
HORIZON = 30  # annual cycles
POP = 100_000  # individuals, matching the tutorial

BASE = dict(
    p_HD=0.005, p_HS1=0.15, p_S1H=0.5, p_S1S2=0.105, rr_S1=3.0, rr_S2=10.0,
    rp_S1S2=0.2, c_H=2000.0, c_S1=4000.0, c_S2=15000.0, c_Trt=12000.0,
    u_H=1.0, u_S1=0.75, u_S2=0.5, u_Trt=0.95, ru_S1S2=0.03,
)


def population(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Per-individual treatment effect modifier, uniform on [0.95, 1.05]."""
    return pd.DataFrame({"x": rng.uniform(0.95, 1.05, n)})


def transition_probabilities(
    params: pd.Series, strategy: str, state: np.ndarray, attrs: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-cycle transition probabilities with duration-dependent mortality."""
    n = len(state)
    probs = np.zeros((n, 4))
    dur = attrs["dur"].to_numpy() + 1.0  # cycles in the sick complex, at the out-transition
    mult = 1.0 + dur * params["rp_S1S2"]  # mortality rate rises with duration
    r_HD = -np.log(1 - params["p_HD"])
    p_S1D = 1 - np.exp(-params["rr_S1"] * r_HD * mult)
    p_S2D = 1 - np.exp(-params["rr_S2"] * r_HD * mult)
    p_HS1, p_S1H, p_S1S2 = params["p_HS1"], params["p_S1H"], params["p_S1S2"]

    h = state == 0
    probs[h] = np.array([1 - p_HS1 - params["p_HD"], p_HS1, 0.0, params["p_HD"]])
    s1 = state == 1
    probs[s1, 0] = p_S1H
    probs[s1, 2] = p_S1S2
    probs[s1, 3] = p_S1D[s1]
    probs[s1, 1] = 1 - p_S1H - p_S1S2 - p_S1D[s1]
    s2 = state == 2
    probs[s2, 2] = 1 - p_S2D[s2]
    probs[s2, 3] = p_S2D[s2]
    probs[state == 3, 3] = 1.0
    return probs


def state_rewards(
    params: pd.Series, strategy: str, state: np.ndarray, attrs: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cycle cost and utility, with a duration-decaying treatment utility."""
    n = len(state)
    cost = np.zeros(n)
    util = np.zeros(n)
    dur = attrs["dur"].to_numpy()  # cycles already spent sick
    x = attrs["x"].to_numpy()
    on_tx = bool(params["on_treatment"])
    tx_cost = params["c_Trt"] if on_tx else 0.0

    h = state == 0
    cost[h] = params["c_H"]
    util[h] = params["u_H"]
    s1 = state == 1
    cost[s1] = params["c_S1"] + tx_cost
    if on_tx:
        util[s1] = x[s1] * (params["u_Trt"] - dur[s1] * params["ru_S1S2"])
    else:
        util[s1] = params["u_S1"]
    s2 = state == 2
    cost[s2] = params["c_S2"] + tx_cost
    util[s2] = params["u_S2"]
    return cost, util


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seeds = SeedManager(1)
    engine = MicrosimModel.discrete(
        states=STATES, transition_probabilities=transition_probabilities,
        state_rewards=state_rewards, population=population,
        n_individuals=POP,
        strategies={"No Treatment": {"on_treatment": 0.0},
                    "Treatment": {"on_treatment": 1.0}},
        n_cycles=HORIZON, discount_rate=0.03,
        cycle_correction="none",
        duration_groups={"dur": ("S1", "S2")},
    )

    draws = pd.DataFrame([BASE], index=pd.RangeIndex(1, name="iteration"))
    outcomes = run_psa(engine, draws, seed=seeds.entropy).outcomes
    print(f"Extended Sick-Sicker microsimulation, {POP:,} individuals:")
    print(icer_table(outcomes).round(2).to_string())
    print(
        "\nPublished Table 3 (n=100,000, seed=1): "
        "No-treatment 62,667 / 15.28, Treatment 117,455 / 15.79, ICER 107,986."
    )

    plot_ce_plane(outcomes, comparator="No Treatment", wtp=100_000.0).figure.savefig(
        OUT / "ce_plane_microsim_mdm.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(outcomes).figure.savefig(
        OUT / "frontier_microsim_mdm.png", dpi=150, bbox_inches="tight"
    )

    record = capture_run(
        seed=seeds,
        outcomes=outcomes,
        draw_sources={c: "Sick-Sicker microsimulation tutorial (MDM 2018)" for c in draws.columns},
        note=(
            f"Extended Sick-Sicker microsimulation, {POP:,} individuals over "
            f"{HORIZON} cycles, duration-dependent mortality and treatment utility."
        ),
    )
    (OUT / "run_report_microsim_mdm.md").write_text(
        record.to_markdown("heormodel Sick-Sicker microsimulation run report")
    )
    print(f"\nWrote plots and run report to {OUT}/")


if __name__ == "__main__":
    main()
