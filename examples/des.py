"""End-to-end example: a resource-constrained clinic through CEA and VoI.

A discrete-event model earns its keep when a shared, scarce resource makes
entities queue, so an individual's outcome depends on the others in the system.
A cohort or independent-microsimulation model cannot carry that coupling.

The model is a specialist clinic over a four-year horizon. Patients arrive over
time and wait for a specialist (a SimPy resource). While waiting, untreated
disease costs ``c_wait_year`` per year and holds utility at ``u_wait``. Once
seen, a one-off treatment cost is incurred, and the patient spends the rest of
the horizon treated at the higher ``u_treated`` with a follow-up cost rate.

Two strategies differ only in capacity. Standard staffing runs one specialist;
expanded staffing runs two at a per-patient overhead ``c_capacity``. More
capacity cuts the queue, so patients spend less time at the low waiting utility.
Both strategies see the same patients and the same service draws through common
random numbers (the engine default), so the incremental result reflects the
capacity change, not sampling noise. `DESEngine.evaluate` conforms to the model
contract, so `run_psa`, `heval.cea`, and `heval.voi` treat it like any engine.

Run it with::

    uv run python examples/des.py

Outputs (written to ``examples/output/``):
    - ce_plane_des.png, ceac_des.png, frontier_des.png
    - run_report_des.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import simpy

from heval.cea import ceac, ceaf, icer_table
from heval.models import DESEngine, queue_waits
from heval.params import Beta, Gamma, ParameterSet
from heval.report import capture_run, plot_ce_plane, plot_ceac, plot_frontier
from heval.run import SeedManager, run_psa
from heval.voi import evpi, evppi_ranking

HERE = Path(__file__).parent
OUT = HERE / "output"

WTP = 30_000.0  # the two staffing levels are about equally likely to be optimal here
N_PSA = 256  # PSA iterations
N_PATIENTS = 30  # patients per iteration
HORIZON = 4.0  # years
ARRIVAL_RATE = 15.0  # patients per year


def patients(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Sample patient arrival times as a Poisson process over the horizon."""
    return pd.DataFrame({"arrival": np.cumsum(rng.exponential(1.0 / ARRIVAL_RATE, n))})


def resources(env: simpy.Environment, params: pd.Series, strategy: str) -> dict[str, Any]:
    """One specialist resource whose capacity is the strategy's staffing level."""
    return {"specialist": simpy.Resource(env, capacity=int(params["n_servers"]))}


def clinic(
    env: simpy.Environment,
    patient: pd.Series,
    params: pd.Series,
    strategy: str,
    toolkit: Any,
) -> Any:
    """One patient: arrive, queue for a specialist, be treated, follow up."""
    arrival = float(patient["arrival"])
    if arrival >= HORIZON:
        return
    yield env.timeout(arrival)
    toolkit.state("waiting")
    with toolkit.request("specialist") as slot:
        # Race the queue against the horizon: a patient still waiting at the
        # horizon is never seen, and only the waiting segment is billed.
        result = yield slot | env.timeout(HORIZON - env.now)
        served = slot in result
        toolkit.accrue_over(
            arrival, env.now, params["c_wait_year"], params["u_wait"], component="cost_waiting"
        )
        if served:
            toolkit.state("treatment")
            toolkit.accrue_cost(
                params["c_treat"] + params["c_capacity"], component="cost_treatment"
            )
            # From treatment to the horizon the patient is treated: higher utility.
            toolkit.accrue_over(
                env.now,
                HORIZON,
                params["c_followup_year"],
                params["u_treated"],
                component="cost_followup",
            )
            service = toolkit.rng.gamma(2.0, params["service_time"] / 2.0)  # mean service_time
            yield env.timeout(service)
            toolkit.state("treated")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seeds = SeedManager(20260704)

    parameters = ParameterSet(
        {
            "service_time": Gamma.from_mean_se(0.08, 0.015),  # years, about 29 days
            "u_wait": Beta.from_mean_se(0.60, 0.05),
            "u_treated": Beta.from_mean_se(0.85, 0.03),
            "c_wait_year": Gamma.from_mean_se(9_000.0, 1_500.0),
            "c_treat": Gamma.from_mean_se(5_000.0, 800.0),
            "c_followup_year": Gamma.from_mean_se(1_500.0, 300.0),
            "c_capacity": Gamma.from_mean_se(3_500.0, 600.0),  # expanded arm only
        }
    )
    draws = parameters.sample(N_PSA, seed=seeds.generator())

    engine = DESEngine(
        process=clinic,
        entities=patients,
        n_entities=N_PATIENTS,
        resources=resources,
        strategies={
            "Standard capacity": {"n_servers": 1, "c_capacity": 0.0},
            "Expanded capacity": {"n_servers": 2},
        },
        horizon=HORIZON,
        seed_manager=seeds,
    )

    outcomes = run_psa(engine, draws)
    print(outcomes)
    print("\nMean outcome per patient:")
    print(outcomes.summary().round(3).to_string())
    print("\nIncremental analysis:")
    print(icer_table(outcomes).round(3).to_string())

    print(f"\nEVPI at WTP {WTP:,.0f}: {evpi(outcomes, WTP):,.1f}")
    print("\nEVPPI ranking:")
    print(evppi_ranking(outcomes, draws, WTP).round(1).to_string())

    # Queueing report from the trace, one iteration at the posterior mean draw.
    _, trace = engine.evaluate(draws.iloc[[0]], trace=True)
    waits = queue_waits(trace)
    mean_wait = waits.groupby("strategy", sort=False)["wait"].mean() * 365.0
    print("\nMean queue wait (days), first iteration:")
    print(mean_wait.round(1).to_string())

    grid = np.linspace(0, 80_000, 41)
    plot_ce_plane(outcomes, comparator="Standard capacity", wtp=WTP).figure.savefig(
        OUT / "ce_plane_des.png", dpi=150, bbox_inches="tight"
    )
    plot_ceac(ceac(outcomes, grid), ceaf_df=ceaf(outcomes, grid)).figure.savefig(
        OUT / "ceac_des.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(outcomes).figure.savefig(
        OUT / "frontier_des.png", dpi=150, bbox_inches="tight"
    )

    record = capture_run(
        seed=seeds,
        outcomes=outcomes,
        draw_sources=dict.fromkeys(draws.columns, "literature (mean/SE)"),
        note=(
            f"Discrete-event clinic, {N_PATIENTS} patients over {HORIZON:.0f} years "
            "per iteration, common random numbers across staffing levels."
        ),
    )
    (OUT / "run_report_des.md").write_text(record.to_markdown("heval discrete-event run report"))
    print(f"\nWrote plots and run report to {OUT}/")


if __name__ == "__main__":
    main()
