"""End-to-end example: a discrete-time microsimulation through CEA and VoI.

An individual-level model earns its keep when outcomes depend on individual
history or heterogeneity, which a cohort average cannot carry. This model has
both: each person carries a ``frailty`` attribute that scales their sick-state
cost and mortality, and the Sick to Dead risk rises with time already spent
sick. Neither is expressible as a single cohort transition matrix.

The model is three states, Healthy, Sick, Dead, over a 30-year horizon:

    Healthy -> Sick     onset probability p_hs, scaled by rr_tx on treatment
    Sick -> Dead        p_sd, scaled by frailty and rising with time in Sick
    Healthy/Sick -> Dead background mortality

Two strategies share one population through common random numbers (the engine
default), so the incremental result reflects the treatment effect rather than
sampling noise. `MicrosimModel.evaluate` conforms to the model
contract, so `run_psa`, `heval.cea`, and `heval.voi` treat it like any engine.

Run it with::

    uv run python examples/microsim.py

Outputs (written to ``examples/output/``):
    - ce_plane_micro.png, ceac_micro.png, frontier_micro.png
    - run_report_micro.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heormodel.cea import ceac, ceaf, icer_table
from heormodel.models import MicrosimModel
from heormodel.params import Beta, Fixed, Gamma, ParameterSet
from heormodel.report import capture_run, plot_ce_plane, plot_ceac, plot_frontier
from heormodel.run import SeedManager, run_psa
from heormodel.voi import evpi, evppi_ranking

HERE = Path(__file__).parent
OUT = HERE / "output"

WTP = 11_000.0  # near the base-case ICER, so the decision is genuinely uncertain
N = 256  # PSA iterations
POP = 800  # individuals per iteration
HORIZON = 30  # cycles (years)
BACKGROUND_MORTALITY = 0.01
WORSENING = 0.05  # extra Sick -> Dead risk per year already spent sick

STATES = ("Healthy", "Sick", "Dead")


def population(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Sample per-individual frailty, lognormal around 1."""
    return pd.DataFrame({"frailty": rng.lognormal(mean=0.0, sigma=0.3, size=n)})


def transition_probabilities(
    params: pd.Series, state: np.ndarray, attrs: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Per-cycle transition probabilities, with history and heterogeneity."""
    n = len(state)
    probs = np.zeros((n, 3))
    on_tx = bool(params["on_treatment"])
    p_hs = params["p_hs"] * (params["rr_tx"] if on_tx else 1.0)

    healthy = state == 0
    probs[healthy, 1] = p_hs
    probs[healthy, 2] = BACKGROUND_MORTALITY
    probs[healthy, 0] = 1.0 - p_hs - BACKGROUND_MORTALITY

    sick = state == 1
    frailty = attrs["frailty"].to_numpy()[sick]
    time_sick = attrs["time_in_state"].to_numpy()[sick]
    p_sd = np.clip(params["p_sd"] * frailty * (1.0 + WORSENING * time_sick), 0.0, 1.0)
    probs[sick, 2] = p_sd
    probs[sick, 1] = 1.0 - p_sd

    probs[state == 2, 2] = 1.0  # Dead is absorbing
    return probs


def state_costs_and_utilities(
    params: pd.Series, state: np.ndarray, attrs: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cycle cost and QALY of each individual's current state."""
    n = len(state)
    cost = np.zeros(n)
    qaly = np.zeros(n)
    frailty = attrs["frailty"].to_numpy()
    tx_cost = params["c_treat"] if bool(params["on_treatment"]) else 0.0

    healthy = state == 0
    cost[healthy] = params["c_well"] + tx_cost
    qaly[healthy] = 1.0

    sick = state == 1
    cost[sick] = params["c_sick"] * frailty[sick] + tx_cost
    qaly[sick] = params["u_sick"]
    return cost, qaly


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seeds = SeedManager(20260704)

    parameters = ParameterSet(
        {
            "p_hs": Beta.from_mean_se(0.08, 0.02),
            "p_sd": Beta.from_mean_se(0.12, 0.03),
            "rr_tx": Beta.from_mean_se(0.60, 0.05),
            "u_sick": Beta.from_mean_se(0.65, 0.05),
            "c_well": Fixed(500.0),
            "c_sick": Gamma.from_mean_se(9_000.0, 1_500.0),
            "c_treat": Gamma.from_mean_se(2_000.0, 300.0),
        }
    )
    draws = parameters.sample(N, seed=seeds.generator())

    engine = MicrosimModel(
        states=STATES,
        transition_probabilities=transition_probabilities,
        state_costs_and_utilities=state_costs_and_utilities,
        population=population,
        n_individuals=POP,
        strategies={
            "Standard care": {"on_treatment": 0.0},
            "Treatment": {"on_treatment": 1.0},
        },
        horizon=HORIZON,
        seed_manager=seeds,
    )

    outcomes = run_psa(engine, draws)
    print(outcomes)
    print("\nIncremental analysis:")
    print(icer_table(outcomes).round(3).to_string())

    print(f"\nEVPI at WTP {WTP:,.0f}: {evpi(outcomes, WTP):,.1f}")
    print("\nEVPPI ranking:")
    print(evppi_ranking(outcomes, draws, WTP).round(1).to_string())

    grid = np.linspace(0, 80_000, 41)
    plot_ce_plane(outcomes, comparator="Standard care", wtp=WTP).figure.savefig(
        OUT / "ce_plane_micro.png", dpi=150, bbox_inches="tight"
    )
    plot_ceac(ceac(outcomes, grid), ceaf_df=ceaf(outcomes, grid)).figure.savefig(
        OUT / "ceac_micro.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(outcomes).figure.savefig(
        OUT / "frontier_micro.png", dpi=150, bbox_inches="tight"
    )

    record = capture_run(
        seed=seeds,
        outcomes=outcomes,
        draw_sources=dict.fromkeys(draws.columns, "literature (mean/SE)"),
        note=(
            f"Discrete-time microsimulation, {POP} individuals over {HORIZON} "
            "cycles per iteration, common random numbers across strategies."
        ),
    )
    (OUT / "run_report_micro.md").write_text(record.to_markdown("heval microsimulation run report"))
    print(f"\nWrote plots and run report to {OUT}/")


if __name__ == "__main__":
    main()
