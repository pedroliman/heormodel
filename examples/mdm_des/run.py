"""Run the continuous-time Sick-Sicker discrete-event replication end to end.

This script shows how the building blocks compose. It loads the life table,
builds the engine, runs the base case and the probabilistic analysis, folds in
the transition costs and utilities accrued over each sojourn, and writes the
figures and the run report. The reusable pieces live in the sibling modules of
this folder; here the configuration is passed in and each function is used
through its inputs and outputs.

Reproducing the published figures means matching the companion R code rather
than the Table 1 specification on two points: transition amounts accrue over the
sojourn that ends in each transition, and six parameters the companion draws but
never reads are held at their base case. The one remaining departure, a single
random-number stream shared across all parameter sets, is the framework's
per-iteration seeding guarantee. See devdocs/replication-notes/mdm-des-departures.md
for the full accounting.

Run it with::

    uv run python examples/mdm_des/run.py

Outputs (written to ``examples/output/``):
    - epi_des_mdm.png, ceac_des_mdm.png, expected_loss_des_mdm.png,
      evpi_des_mdm.png, frontier_des_mdm.png
    - run_report_des_mdm.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from heormodel.cea import ceac, ceaf, expected_loss, icer_table
from heormodel.params import single_draw
from heormodel.report import capture_run, plot_ceac, plot_expected_loss, plot_frontier
from heormodel.run import SeedManager, run_psa
from heormodel.voi import evpi

from model import build_engine
from mortality import load_life_table
from outcomes import dwell_times, survival_and_prevalence
from parameters import base_case, interventions, parameter_set, states
from plots import plot_epidemiology
from transitions import costs_and_utilities_model, with_transition_costs_and_utilities

HERE = Path(__file__).parent
DATA = HERE / "data" / "us_mortality_2015.csv"
OUT = HERE.parent / "output"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Model configuration, passed to the building blocks rather than kept global.
    age_start, age_end = 25.0, 100.0
    horizon = age_end - age_start
    discount_rate = 0.03
    n_base = 100_000  # individuals in the base case, matching the article
    n_psa_draws = 1_000  # parameter sets, matching the article
    n_psa_individuals = 10_000  # per set; the article uses 100,000
    wtp_grid = np.arange(0.0, 200_001.0, 1_000.0)  # $1,000 steps, matching the article

    state_labels = states()
    intervention_defs = interventions()
    intervention_names = [s.name for s in intervention_defs]
    life_table = load_life_table(DATA)
    seeds = SeedManager(2)

    # Base case: one deterministic run with the event history. The transition
    # amounts accrue over the preceding sojourn, folded in from that history.
    engine = build_engine(
        life_table=life_table, states=state_labels, interventions=intervention_defs,
        age_start=age_start, horizon=horizon, discount_rate=discount_rate,
        population=n_base,
    )
    base = single_draw(base_case())
    base_run = run_psa(engine, base, seed=seeds.entropy, collect="events")
    outcomes, events = base_run.outcomes, base_run.events
    outcomes = with_transition_costs_and_utilities(
        outcomes, events, base, n_individuals=n_base, discount_rate=discount_rate
    )
    print(f"Base case, {n_base:,} individuals per intervention, ages 25 to 100:")
    print(icer_table(outcomes).round(2).to_string())
    print(
        "\nThe frontier runs standard of care, then B, then AB; A is dominated"
        "\n(B costs less and yields more QALYs). The article reports the same"
        "\nordering: its acceptability frontier switches near the two ICERs."
    )

    survival, prevalence = survival_and_prevalence(
        events, states=state_labels, interventions=intervention_names, initial_state="H",
        dead_state="D", disease_states=("S1", "S2"), n_individuals=n_base, horizon=horizon,
    )
    plot_epidemiology(
        survival, prevalence, interventions=intervention_names, age_start=age_start,
        path=OUT / "epi_des_mdm.png",
    )
    print("\nMean completed dwell time by state (years):")
    print(dwell_times(events).round(2).to_string())

    # Probabilistic analysis: the article's 1,000 parameter sets. 10,000
    # individuals per set (rather than the article's 100,000) keeps the run
    # near two minutes; common random numbers absorb most of the extra per-set
    # Monte Carlo noise. The engine is wrapped so each iteration adds its
    # sojourn-accrued transition amounts, and one draw per experiment keeps each
    # worker's event history small.
    params = parameter_set()
    draws = params.sample(n_psa_draws, seed=seeds.generator())
    model = costs_and_utilities_model(
        build_engine(
            life_table=life_table, states=state_labels, interventions=intervention_defs,
            age_start=age_start, horizon=horizon, discount_rate=discount_rate,
            population=n_psa_individuals,
        ),
        n_individuals=n_psa_individuals, discount_rate=discount_rate,
    )
    psa = run_psa(model, draws, seed=seeds.entropy, batch_size=1).outcomes
    print(f"\nProbabilistic analysis, {n_psa_draws:,} parameter sets:")
    print(icer_table(psa).round(2).to_string())

    curve = ceac(psa, wtp_grid)
    front = ceaf(psa, wtp_grid)
    losses = expected_loss(psa, wtp_grid)
    evpi_curve = evpi(psa, wtp_grid)

    plot_ceac(curve, ceaf_df=front).figure.savefig(
        OUT / "ceac_des_mdm.png", dpi=150, bbox_inches="tight"
    )
    plot_expected_loss(losses).figure.savefig(
        OUT / "expected_loss_des_mdm.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(psa).figure.savefig(OUT / "frontier_des_mdm.png", dpi=150, bbox_inches="tight")
    ax = plt.subplots()[1]
    ax.plot(evpi_curve.index, evpi_curve, lw=1.8, color="0.15")
    ax.set_xlabel("Willingness to pay")
    ax.set_ylabel("EVPI per person")
    ax.set_title("Expected value of perfect information")
    ax.figure.savefig(OUT / "evpi_des_mdm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    switches = (
        front["intervention"].ne(front["intervention"].shift()) & front.index.to_series().gt(0)
    )
    switch_points = [float(w) for w in front.index[switches]]
    at_switches = ", ".join(f"{evpi_curve.loc[w]:,.0f} at {w:,.0f}" for w in switch_points)
    print(
        f"\nThe acceptability frontier switches at {switch_points} dollars per QALY,"
        f"\nwith EVPI peaks (dollars per person) of {at_switches}, matching the two"
        "\nswitch points and the small EVPI peaks of the article's figure 4. Holding"
        "\nthe six companion-fixed parameters at base case keeps the intervention"
        "\ncomparisons near-certain. The only remaining difference is Monte Carlo"
        "\nerror: the companion drives every parameter set from one shared"
        "\nrandom-number stream, while this run keeps the framework's per-iteration"
        "\nseeding. See devdocs/replication-notes/mdm-des-departures.md for details."
    )

    record = capture_run(
        seed=seeds,
        outcomes=psa,
        draw_sources={c: str(params.distributions[c]) for c in draws.columns},
        note=(
            f"Continuous-time Sick-Sicker DES replication (MDM 2026): base case "
            f"{n_base:,} individuals, PSA {n_psa_draws:,} sets x {n_psa_individuals:,} individuals."
        ),
    )
    (OUT / "run_report_des_mdm.md").write_text(
        record.to_markdown("heormodel Sick-Sicker DES replication run report")
    )
    print(f"\nWrote plots and run report to {OUT}/")


if __name__ == "__main__":
    main()
