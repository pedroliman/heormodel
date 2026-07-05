"""Deterministic sensitivity analysis on the Sick-Sicker cohort model.

Companion to ``examples/mdm_cohort.py``. Where that script runs a
probabilistic sensitivity analysis, this one runs the deterministic forms:
a one-way sweep, a one-at-a-time tornado, and a two-way grid. All three feed
`heval.dsa` designs through the same `run_psa` loop the PSA uses, then read
the results with the report layer.

The base case is the tutorial's Table 1 point estimates. The outcome of
interest is the incremental net monetary benefit of Strategy AB over standard
of care at a willingness-to-pay threshold of 100,000 per QALY.

Run it with::

    uv run python examples/dsa.py

Outputs (written to ``examples/output/``):
    - tornado_dsa.png, heatmap_dsa.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from heval.cea import nmb
from heval.dsa import grid, one_at_a_time, one_way
from heval.models import CohortSpec, MarkovModel
from heval.report import heatmap_data, plot_tornado, tornado_data
from heval.run import run_psa

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("H", "S1", "S2", "D")
STRATEGIES = ("Standard of care", "Strategy A", "Strategy B", "Strategy AB")
N_CYCLES = 75
WTP = 100_000.0

BASE = pd.Series(
    dict(
        r_HD=0.002,
        r_HS1=0.15,
        r_S1H=0.5,
        r_S1S2=0.105,
        hr_S1=3.0,
        hr_S2=10.0,
        hr_S1S2_trtB=0.6,
        c_H=2000.0,
        c_S1=4000.0,
        c_S2=15000.0,
        c_trtA=12000.0,
        c_trtB=13000.0,
        u_H=1.0,
        u_S1=0.75,
        u_S2=0.5,
        u_trtA=0.95,
    )
)


def rate_to_prob(rate: float | np.ndarray) -> np.ndarray:
    """Convert an instantaneous rate to a per-cycle probability."""
    return 1.0 - np.exp(-np.asarray(rate))


def model(params: pd.Series, strategy: str) -> CohortSpec:
    """Transition matrix and per-state payoffs for one strategy and draw."""
    p_HS1 = rate_to_prob(params["r_HS1"])
    p_S1H = rate_to_prob(params["r_S1H"])
    p_S1S2 = rate_to_prob(params["r_S1S2"])
    p_HD = rate_to_prob(params["r_HD"])
    p_S1D = rate_to_prob(params["r_HD"] * params["hr_S1"])
    p_S2D = rate_to_prob(params["r_HD"] * params["hr_S2"])
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
    return CohortSpec(P, cost, effect)


def incremental_nmb(design: pd.DataFrame) -> pd.Series:
    """Incremental NMB of Strategy AB over standard of care, per scenario."""
    outcomes = run_psa(engine, design)
    nb = nmb(outcomes, WTP)
    return nb["Strategy AB"] - nb["Standard of care"]


engine = MarkovModel(
    states=STATES,
    strategies=STRATEGIES,
    model_fn=model,
    n_cycles=N_CYCLES,
    start="H",
    discount_rate=0.03,
    half_cycle_correction="simpson",
)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # One-way sweep: incremental NMB of AB as the treatment-B cost moves.
    one_way_design, _ = one_way(BASE, "c_trtB", [8_000.0, 13_000.0, 18_000.0])
    print("One-way sweep of c_trtB (incremental NMB of AB over standard of care):")
    sweep = incremental_nmb(one_way_design)
    for value, nb in zip([8_000.0, 13_000.0, 18_000.0], sweep, strict=True):
        print(f"  c_trtB = {value:>8,.0f}: incremental NMB = {nb:>12,.0f}")

    # One-at-a-time tornado: each parameter to +/- 20% of its base value.
    ranges = {name: (0.8 * BASE[name], 1.2 * BASE[name]) for name in BASE.index}
    oat_design, oat_descriptor = one_at_a_time(BASE, ranges)
    oat_outcomes = run_psa(engine, oat_design)
    td = tornado_data(
        oat_outcomes,
        (oat_design, oat_descriptor),
        wtp=WTP,
        strategy="Strategy AB",
        comparator="Standard of care",
    )
    print("\nTornado table (+/- 20% one-at-a-time, incremental NMB of AB):")
    print(td.round(0).to_string())
    plot_tornado(td).figure.savefig(OUT / "tornado_dsa.png", dpi=150, bbox_inches="tight")

    # Two-way grid: incremental NMB of AB across the two treatment costs.
    a_values = [6_000.0, 12_000.0, 18_000.0]
    b_values = [8_000.0, 13_000.0, 18_000.0]
    grid_design, grid_descriptor = grid(BASE, {"c_trtA": a_values, "c_trtB": b_values})
    grid_nmb = incremental_nmb(grid_design)
    hm = heatmap_data(grid_nmb, grid_descriptor, x="c_trtA", y="c_trtB")
    print("\nTwo-way grid of incremental NMB (rows c_trtB, columns c_trtA):")
    print(hm.round(0).to_string())

    fig, ax = plt.subplots()
    im = ax.imshow(hm.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(a_values)), [f"{v:,.0f}" for v in a_values])
    ax.set_yticks(range(len(b_values)), [f"{v:,.0f}" for v in b_values])
    ax.set_xlabel("c_trtA")
    ax.set_ylabel("c_trtB")
    ax.set_title("Incremental NMB of Strategy AB")
    fig.colorbar(im, ax=ax, label="Net monetary benefit")
    fig.savefig(OUT / "heatmap_dsa.png", dpi=150, bbox_inches="tight")

    print(f"\nWrote tornado and heatmap plots to {OUT}/")


if __name__ == "__main__":
    main()
