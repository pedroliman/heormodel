"""End-to-end example: bring-your-own-outputs PSA -> CEA -> VoI -> report.

This script demonstrates the adoption wedge of ``heormodel``: a costs/effects
PSA table produced by *any* external model (here, synthesised and written to
CSV to stand in for a spreadsheet export or a legacy simulator) is loaded
with ``as_outcomes`` and driven through the full analysis layer without
touching a model engine.

Run it with::

    uv run python examples/byoo_example.py

Outputs (written next to this script, in ``examples/output/``):
    - ce_plane.png, ceac.png, frontier.png, tornado.png
    - run_report.md, run_record.json
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heormodel.cea import ceac, ceaf, icer_table
from heormodel.params import Beta, Gamma, Normal, ParameterSet
from heormodel.report import (
    capture_run,
    plot_ce_plane,
    plot_ceac,
    plot_frontier,
    plot_tornado,
    tornado_data,
)
from heormodel.run import SeedManager, as_outcomes
from heormodel.voi import evpi, evppi_ranking

HERE = Path(__file__).parent
OUT = HERE / "output"
WTP = 30_000.0  # sits between the two frontier ICERs, so the decision is uncertain
N = 5_000


def make_external_psa_table(seed_manager: SeedManager) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stand-in for an external model: produce a tidy PSA table and its draws.

    Three interventions for a chronic disease: standard of care, a new drug,
    and drug + monitoring. Costs and QALYs per iteration are simple
    functions of sampled parameters, so the VoI results are traceable.
    """
    params = ParameterSet(
        {
            "p_response": Beta.from_mean_se(0.35, 0.05),
            "rr_drug": Normal(0.75, 0.08),
            "c_drug": Gamma.from_mean_se(12_000, 1_500),
            "c_monitoring": Gamma.from_mean_se(2_000, 400),
            "u_gain": Beta.from_mean_se(0.12, 0.03),
        },
        correlation={("p_response", "u_gain"): 0.3},
    )
    draws = params.sample(N, seed=seed_manager.generator())

    base_qaly = 8.0
    rows = []
    for i, row in draws.iterrows():
        effect_drug = base_qaly + row["u_gain"] * row["p_response"] / row["rr_drug"] * 10
        effect_mon = effect_drug + 0.15 * row["p_response"]
        rows += [
            {"intervention": "Standard care", "iteration": i, "cost": 40_000.0, "qaly": base_qaly},
            {
                "intervention": "New drug",
                "iteration": i,
                "cost": 40_000.0 + row["c_drug"],
                "qaly": effect_drug,
            },
            {
                "intervention": "Drug + monitoring",
                "iteration": i,
                "cost": 40_000.0 + row["c_drug"] + row["c_monitoring"],
                "qaly": effect_mon,
            },
        ]
    return pd.DataFrame(rows), draws


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seed_manager = SeedManager(20260704)

    # --- an "external" PSA table arrives as a CSV -------------------------
    psa_table, draws = make_external_psa_table(seed_manager)
    csv_path = OUT / "external_psa.csv"
    psa_table.to_csv(csv_path, index=False)

    # --- bring your own outputs: one call into the standard schema --------
    outcomes = as_outcomes(csv_path)
    print(outcomes)

    # --- cost-effectiveness analysis --------------------------------------
    table = icer_table(outcomes)
    print("\nIncremental analysis:")
    print(table.round(3).to_string())

    grid = np.linspace(0, 80_000, 51)
    ceac_df = ceac(outcomes, grid)
    ceaf_df = ceaf(outcomes, grid)

    # --- value of information ---------------------------------------------
    print(f"\nEVPI at WTP {WTP:,.0f}: {evpi(outcomes, WTP):,.1f}")
    ranking = evppi_ranking(outcomes, draws, WTP)
    print("\nEVPPI ranking (research prioritisation):")
    print(ranking.round(1).to_string())

    # --- plots -------------------------------------------------------------
    plot_ce_plane(outcomes, comparator="Standard care", wtp=WTP).figure.savefig(
        OUT / "ce_plane.png", dpi=150, bbox_inches="tight"
    )
    plot_ceac(ceac_df, ceaf_df=ceaf_df).figure.savefig(
        OUT / "ceac.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(outcomes).figure.savefig(OUT / "frontier.png", dpi=150, bbox_inches="tight")
    td = tornado_data(outcomes, draws, WTP, intervention="New drug", comparator="Standard care")
    plot_tornado(td).figure.savefig(OUT / "tornado.png", dpi=150, bbox_inches="tight")

    # --- reproducibility record --------------------------------------------
    record = capture_run(
        seed=seed_manager,
        outcomes=outcomes,
        note="Bring-your-own-outputs example: external PSA CSV through CEA and VoI.",
    )
    record.to_json(OUT / "run_record.json")
    (OUT / "run_report.md").write_text(record.to_markdown("heormodel example run report"))
    print(f"\nWrote plots, run report, and run record to {OUT}/")


if __name__ == "__main__":
    main()
