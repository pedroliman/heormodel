"""End-to-end example: calibrate some parameters, take the rest from the
literature, then mix both into one PSA that flows through CEA and VoI.

This is the workflow most applied models need. A natural-history model has
transition rates that no single study reports, so they are calibrated to
observed prevalence. Utilities and costs come from the literature as
mean/SE distributions. `heormodel.params.mix_draws` joins the calibrated
posterior and the literature draws into one matrix, and from that point the
analysis layer does not care where a parameter came from.

The disease model is a three-state continuous-time Markov chain over Healthy,
Sick, and Dead, starting at age 40. Two rates are calibrated:

    onset:       Healthy -> Sick hazard
    progression: excess Sick -> Dead hazard

Treatment multiplies the progression hazard by ``rr_progression`` (a
literature parameter), buying time in the Sick state at cost ``c_treat`` per
year. Discounted state occupancy is the closed-form integral of the
transition matrix, ``e0 @ (r I - Q)^-1``, so no time loop is needed.

Run it with::

    uv pip install -e '.[calibration]'
    uv run python examples/calibration_workflow.py

Outputs (written to ``examples/output/``):
    - ce_plane_calib.png, ceac_calib.png, frontier_calib.png, tornado_calib.png
    - run_report_calib.md, run_record_calib.json
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from heormodel.calibrate import abc_calibrate
from heormodel.cea import ceac, ceaf, icer_table
from heormodel.models import Outcomes
from heormodel.params import Beta, Gamma, ParameterSet, Uniform, mix_draws
from heormodel.report import (
    capture_run,
    plot_ce_plane,
    plot_ceac,
    plot_frontier,
    plot_tornado,
    tornado_data,
)
from heormodel.run import SeedManager, run_psa
from heormodel.voi import evpi, evppi_ranking

# Quiet pyabc's per-population logging. pyabc reads this on first import, which
# happens lazily inside abc_calibrate, so setting it here takes effect.
os.environ.setdefault("ABC_LOG_LEVEL", "WARNING")

HERE = Path(__file__).parent
OUT = HERE / "output"

WTP = 50_000.0  # near the base-case ICER, so the decision is genuinely uncertain
N = 2_000  # PSA iterations and posterior draws
DISCOUNT = 0.03
BACKGROUND_MORTALITY = 0.012  # fixed all-cause hazard, held constant in the simulator
START_AGE = 40

# Registry targets: disease prevalence among the living at two ages. Generated
# from onset=0.02, progression=0.05 to give the calibration a known answer.
OBSERVED = {"prev_age50": 0.147, "prev_age70": 0.283}


def _generator(onset: float, progression: float) -> np.ndarray:
    """Three-state Markov generator over [Healthy, Sick, Dead]."""
    mu = BACKGROUND_MORTALITY
    return np.array(
        [
            [-(onset + mu), onset, mu],
            [0.0, -(mu + progression), mu + progression],
            [0.0, 0.0, 0.0],
        ]
    )


def natural_history(params: dict[str, float]) -> dict[str, float]:
    """Simulator for calibration: rates -> prevalence at ages 50 and 70.

    Prevalence is the Sick fraction among the living, read off the
    transition matrix ``expm(Q t)`` started from a fully Healthy cohort.
    """
    q = _generator(params["onset"], params["progression"])
    out = {}
    for label, age in (("prev_age50", 50), ("prev_age70", 70)):
        p = np.array([1.0, 0.0, 0.0]) @ expm(q * (age - START_AGE))
        out[label] = float(p[1] / (p[0] + p[1]))
    return out


def _occupancy(onset: np.ndarray, progression: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discounted person-years in Healthy and Sick, vectorised over iterations.

    ``integral_0^inf e^{-rt} e^{Qt} dt = (r I - Q)^{-1}`` gives discounted
    occupancy in closed form; row 0 holds time in each state started from
    Healthy.
    """
    mu = BACKGROUND_MORTALITY
    n = len(onset)
    q = np.zeros((n, 3, 3))
    q[:, 0, 0] = -(onset + mu)
    q[:, 0, 1] = onset
    q[:, 0, 2] = mu
    q[:, 1, 1] = -(mu + progression)
    q[:, 1, 2] = mu + progression
    m = np.linalg.inv(DISCOUNT * np.eye(3) - q)
    return m[:, 0, 0], m[:, 0, 1]  # time in Healthy, time in Sick


def disease_model(draws: pd.DataFrame) -> Outcomes:
    """Map the mixed draw matrix to per-strategy costs and QALYs.

    Standard care lets the disease progress at the calibrated rate.
    Treatment scales the progression hazard by ``rr_progression``, extending
    time in Sick at ``u_sick`` utility while adding ``c_treat`` per year.
    """
    onset = draws["onset"].to_numpy()
    progression = draws["progression"].to_numpy()
    u_sick = draws["u_sick"].to_numpy()
    c_sick = draws["c_sick"].to_numpy()
    c_treat = draws["c_treat"].to_numpy()
    rr = draws["rr_progression"].to_numpy()

    _, sick_std = _occupancy(onset, progression)
    healthy, sick_treat = _occupancy(onset, progression * rr)

    qaly_std = healthy + sick_std * u_sick
    qaly_treat = healthy + sick_treat * u_sick
    cost_std = sick_std * c_sick
    cost_treat = sick_treat * (c_sick + c_treat)

    costs = pd.DataFrame({"Standard care": cost_std, "Treatment": cost_treat}, index=draws.index)
    effects = pd.DataFrame({"Standard care": qaly_std, "Treatment": qaly_treat}, index=draws.index)
    return Outcomes.from_wide(costs, effects)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seeds = SeedManager(20260704)
    rng_abc, rng_lit, rng_mix = seeds.spawn(3)

    # --- 1. calibrate the natural-history rates to the registry targets ----
    priors = {"onset": Uniform(0.005, 0.04), "progression": Uniform(0.01, 0.1)}
    calibration = abc_calibrate(
        natural_history,
        priors=priors,
        observed=OBSERVED,
        population_size=200,
        max_populations=6,
        n_posterior=N,
        seed=int(rng_abc.integers(2**32)),
    )
    posterior = calibration.posterior
    print(
        f"Calibration: {calibration.n_populations} populations, "
        f"final epsilon {calibration.final_epsilon:.4f}"
    )
    print("Posterior means:", posterior.mean().round(4).to_dict())
    print(
        "Posterior corr(onset, progression):",
        round(posterior["onset"].corr(posterior["progression"]), 3),
    )

    # --- 2. literature parameters as mean/SE distributions -----------------
    literature = ParameterSet(
        {
            "u_sick": Beta.from_mean_se(0.60, 0.05),
            "c_sick": Gamma.from_mean_se(8_000, 1_500),
            "c_treat": Gamma.from_mean_se(4_000, 500),
            "rr_progression": Beta.from_mean_se(0.70, 0.06),
        },
        correlation={("u_sick", "c_sick"): -0.3},
    )
    lit_draws = literature.sample(N, seed=rng_lit)

    # --- 3. mix calibrated and literature draws into one PSA matrix --------
    draws = mix_draws(posterior, lit_draws, n=N, seed=rng_mix)
    print(f"\nMixed draw matrix: {draws.shape[0]} iterations, columns {list(draws.columns)}")

    # --- 4. run the decision model over the mixed draws --------------------
    outcomes = run_psa(disease_model, draws).outcomes
    print(outcomes)

    # --- 5. cost-effectiveness analysis ------------------------------------
    table = icer_table(outcomes)
    print("\nIncremental analysis:")
    print(table.round(3).to_string())

    grid = np.linspace(0, 120_000, 61)
    ceac_df = ceac(outcomes, grid)
    ceaf_df = ceaf(outcomes, grid)

    # --- 6. value of information across both parameter sources -------------
    print(f"\nEVPI at WTP {WTP:,.0f}: {evpi(outcomes, WTP):,.1f}")
    ranking = evppi_ranking(outcomes, draws, WTP)
    print("\nEVPPI ranking (calibrated and literature parameters together):")
    print(ranking.round(1).to_string())

    # --- 7. plots and provenance -------------------------------------------
    plot_ce_plane(outcomes, comparator="Standard care", wtp=WTP).figure.savefig(
        OUT / "ce_plane_calib.png", dpi=150, bbox_inches="tight"
    )
    plot_ceac(ceac_df, ceaf_df=ceaf_df).figure.savefig(
        OUT / "ceac_calib.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(outcomes).figure.savefig(OUT / "frontier_calib.png", dpi=150, bbox_inches="tight")
    td = tornado_data(outcomes, draws, WTP, strategy="Treatment", comparator="Standard care")
    plot_tornado(td).figure.savefig(OUT / "tornado_calib.png", dpi=150, bbox_inches="tight")

    calibrated = {name: "ABC posterior" for name in posterior.columns}
    from_lit = {name: "literature (mean/SE)" for name in lit_draws.columns}
    record = capture_run(
        seed=seeds,
        outcomes=outcomes,
        draw_sources={**calibrated, **from_lit},
        note=(
            "Calibration workflow example: ABC posterior mixed with literature "
            f"draws through CEA and VoI. ABC ran {calibration.n_populations} "
            f"populations to epsilon {calibration.final_epsilon:.4f}."
        ),
    )
    record.to_json(OUT / "run_record_calib.json")
    (OUT / "run_report_calib.md").write_text(
        record.to_markdown("heormodel calibration workflow run report")
    )
    print(f"\nWrote plots, run report, and run record to {OUT}/")


if __name__ == "__main__":
    main()
