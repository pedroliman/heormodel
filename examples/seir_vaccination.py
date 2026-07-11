"""Cost-effectiveness of a vaccination program in a susceptible-exposed-infectious
-recovered (SEIR) transmission model.

The model follows a closed population of 100,000 through an epidemic of a
directly transmitted, non-fatal infection. Five compartments track the
susceptible (S), the exposed but not yet infectious (E), the infectious (I), the
recovered and immune (R), and the vaccinated and immune (V). A force of
infection proportional to the infectious prevalence couples the compartments,
which is what an ordinary differential equation model expresses that a cohort
transition matrix cannot: the hazard each susceptible faces depends on how many
others are infectious right now.

Two interventions are compared. Under "No vaccination" the epidemic runs its
course. Under "Vaccination program" susceptibles are vaccinated at a constant
per-capita rate, moving them to the immune compartment before they can be
infected. Costs fall on two flows: each dose administered (the vaccination flow)
and each infection treated (the incidence flow). Quality-adjusted life-years
accrue on compartment occupancy: everyone healthy contributes one per year and
the infectious contribute less, so averted illness raises the effect.

Vaccination is cost-effective but not cost-saving here: it costs more than the
treatment it averts, buying quality-adjusted life-years at a defensible price.
Whether it clears a given willingness-to-pay threshold depends on how large the
outbreak would have been, which is why the value of information is non-trivial.

Run it with::

    uv run python examples/seir_vaccination.py

Outputs (written to ``examples/output/``):
    - seir_epidemic_curves.png, ce_plane_seir.png, ceac_seir.png
    - run_report_seir.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heormodel.cea import ceac, ceaf, icer_table
from heormodel.models import ODEModel, ODESpec
from heormodel.params import Beta, LogNormal, ParameterSet
from heormodel.report import capture_run, plot_ce_plane, plot_ceac
from heormodel.run import SeedManager, run_psa
from heormodel.voi import evpi, evppi_ranking

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("S", "E", "I", "R", "V")
INTERVENTIONS = ("No vaccination", "Vaccination program")
HORIZON = 10.0  # years
POPULATION = 100_000.0
INITIAL_INFECTIOUS = 10.0
WTP = 50_000.0  # willingness-to-pay per quality-adjusted life-year
N_SIM = 1_000

# Deterministic base-case parameters.
BASE = dict(
    R0=1.8,          # basic reproduction number
    sigma=4.0,       # progression rate E -> I (mean latent period 0.25 year)
    gamma=6.0,       # recovery rate I -> R (mean infectious period about two months)
    nu=1.0,          # vaccination rate applied to susceptibles (per year)
    c_vacc=200.0,    # cost per dose administered
    c_case=150.0,    # cost per infection treated
    u_I=0.7,         # utility while infectious
)


def seir(params: pd.Series, intervention: str) -> ODESpec:
    """SEIR-with-vaccination dynamics and rewards for one intervention and draw."""
    r0, sigma, gamma = params["R0"], params["sigma"], params["gamma"]
    beta = r0 * gamma  # transmission rate from the reproduction number
    # Only the vaccination program vaccinates; the comparator sets the rate to zero.
    nu = params["nu"] if intervention == "Vaccination program" else 0.0

    def derivatives(t: float, y: np.ndarray) -> np.ndarray:
        s, e, i, r, v = y
        living = s + e + i + r + v
        foi = beta * i / living  # force of infection (frequency-dependent)
        new_infections = foi * s
        doses = nu * s
        return np.array([
            -new_infections - doses,
            new_infections - sigma * e,
            sigma * e - gamma * i,
            gamma * i,
            doses,
        ])

    def event_rates(t: float, y: np.ndarray) -> np.ndarray:
        s, e, i, r, v = y
        living = s + e + i + r + v
        foi = beta * i / living
        return np.array([nu * s, foi * s])  # doses per year, new infections per year

    initial = np.array([POPULATION - INITIAL_INFECTIOUS, 0.0, INITIAL_INFECTIOUS, 0.0, 0.0])
    # Effects accrue on occupancy: the healthy gain a quality-adjusted life-year a
    # year, the infectious gain less while ill.
    state_effect = np.array([1.0, 1.0, params["u_I"], 1.0, 1.0])
    return ODESpec(
        derivatives=derivatives,
        initial=initial,
        state_cost=np.zeros(5),
        state_effect=state_effect,
        event_rates=event_rates,
        event_cost=np.array([params["c_vacc"], params["c_case"]]),
        event_effect=np.zeros(2),
    )


def parameters() -> ParameterSet:
    """Probabilistic sensitivity analysis distributions for the epidemic and costs."""
    return ParameterSet(
        {
            "R0": LogNormal(mu=np.log(1.8), sigma=0.30),
            "sigma": LogNormal(mu=np.log(4.0), sigma=0.1),
            "gamma": LogNormal(mu=np.log(6.0), sigma=0.1),
            "nu": LogNormal(mu=np.log(1.0), sigma=0.15),
            "c_vacc": LogNormal(mu=np.log(200.0), sigma=0.25),
            "c_case": LogNormal(mu=np.log(150.0), sigma=0.25),
            "u_I": Beta(8.4, 3.6),  # mean 0.7
        }
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seeds = SeedManager(20260711)
    engine = ODEModel(
        states=STATES,
        interventions=INTERVENTIONS,
        dynamics_and_rewards=seir,
        horizon=HORIZON,
        discount_rate=0.03,
    )

    # Deterministic base case: integrate each arm once at the point estimates.
    base = pd.Series(BASE)
    print("Base-case cost-effectiveness (per 100,000 population):")
    base_draws = pd.DataFrame([BASE], index=pd.RangeIndex(1, name="iteration"))
    print(icer_table(engine.evaluate(base_draws)).round(2).to_string())

    # Epidemic curves show what the vaccination program changes.
    no_vacc = engine.trajectory(base, "No vaccination")
    vacc = engine.trajectory(base, "Vaccination program")
    print(
        f"\nTotal infections over {HORIZON:.0f} years: "
        f"{no_vacc['R'].iloc[-1]:,.0f} without vaccination, "
        f"{vacc['R'].iloc[-1]:,.0f} with."
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, traj, title in ((axes[0], no_vacc, "No vaccination"),
                            (axes[1], vacc, "Vaccination program")):
        for comp in ("S", "I", "R", "V"):
            ax.plot(traj["time"], traj[comp], label=comp)
        ax.set_title(title)
        ax.set_xlabel("Year")
    axes[0].set_ylabel("People")
    axes[1].legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "seir_epidemic_curves.png", dpi=150, bbox_inches="tight")

    # Probabilistic sensitivity analysis on the same model.
    draws = parameters().sample(N_SIM, seed=seeds.generator())
    outcomes = run_psa(engine, draws).outcomes
    print("\nProbabilistic cost-effectiveness (mean over draws):")
    print(icer_table(outcomes).round(2).to_string())
    print(f"\nEVPI at WTP {WTP:,.0f}: {evpi(outcomes, WTP):,.1f}")
    print("\nEVPPI ranking (top parameters):")
    print(evppi_ranking(outcomes, draws, WTP).round(1).head().to_string())

    grid = np.linspace(0, 100_000, 41)
    plot_ce_plane(outcomes, comparator="No vaccination", wtp=WTP).figure.savefig(
        OUT / "ce_plane_seir.png", dpi=150, bbox_inches="tight"
    )
    plot_ceac(ceac(outcomes, grid), ceaf_df=ceaf(outcomes, grid)).figure.savefig(
        OUT / "ceac_seir.png", dpi=150, bbox_inches="tight"
    )

    record = capture_run(
        seed=seeds,
        outcomes=outcomes,
        draw_sources=dict.fromkeys(draws.columns, "SEIR vaccination illustrative example"),
        note=(
            f"SEIR-with-vaccination ordinary differential equation model, {HORIZON:.0f}-year "
            "horizon, continuous 3% discounting, two interventions."
        ),
    )
    (OUT / "run_report_seir.md").write_text(
        record.to_markdown("heormodel SEIR vaccination run report")
    )
    print(f"\nWrote epidemic curves, plots, and run report to {OUT}/")


if __name__ == "__main__":
    main()
