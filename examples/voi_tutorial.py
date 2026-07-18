"""Value of information end to end.

This reproduces the Gaussian linear decision model that anchors the
regression value-of-information literature (Strong, Oakley & Brennan, 2014,
Medical Decision Making 34:311-326; Strong, Oakley, Brennan & Breeze, 2015,
Medical Decision Making 35:570-583). Two interventions, standard care and a new
drug, differ by an uncertain incremental effect ``dq`` (QALYs) and an
uncertain incremental cost ``dc``, both Normal. ``tests/test_voi.py`` checks
this same model's EVPI, EVPPI, and EVSI against their closed forms; this
script demonstrates the workflow.

The workflow is the standard one: a ``ParameterSet``, ``run_psa``, then the
``heormodel.voi`` estimators on the outcomes.

Run it with::

    uv run python examples/voi_tutorial.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heormodel.cea import expected_nmb, icer_table
from heormodel.models import Outcomes
from heormodel.params import Normal, ParameterSet
from heormodel.run import run_psa
from heormodel.voi import evpi, evppi_ranking, evsi_regression, simulate_summaries

WTP = 30_000.0
MU_Q, SD_Q = 0.20, 0.30  # incremental effect, QALYs
MU_C, SD_C = 4_000.0, 8_000.0  # incremental cost
SIGMA = 1.0  # per-patient sd of the observed QALY difference in the study
N = 100_000


def new_drug(draws: pd.DataFrame) -> Outcomes:
    """Two interventions differing by the drawn incremental cost and effect."""
    zero = np.zeros(len(draws))
    costs = pd.DataFrame({"Standard care": zero, "New drug": draws["dc"]}, index=draws.index)
    effects = pd.DataFrame({"Standard care": zero, "New drug": draws["dq"]}, index=draws.index)
    return Outcomes.from_wide(costs, effects)


def main() -> None:
    params = ParameterSet({"dq": Normal(MU_Q, SD_Q), "dc": Normal(MU_C, SD_C)})
    draws = params.sample(N, seed=2026)
    outcomes = run_psa(new_drug, draws, sequential=True).outcomes

    # --- decision -----------------------------------------------------------
    print(f"Willingness to pay: {WTP:,.0f} per QALY\n")
    print("ICER table:")
    print(icer_table(outcomes).round(1).to_string())
    exp_nmb = expected_nmb(outcomes, WTP)
    adopt = exp_nmb.idxmax()
    print(f"\nExpected NMB favours: {adopt} ({exp_nmb[adopt]:,.0f})")
    print("The new drug is adopted on the means; VoI prices the residual uncertainty.\n")

    # --- EVPI: the ceiling on research value --------------------------------
    print(f"EVPI (value of resolving all uncertainty), per person: {evpi(outcomes, WTP):,.0f}")

    # --- EVPPI: which parameters that value attaches to ---------------------
    ranking = evppi_ranking(outcomes, draws, WTP)
    print("\nEVPPI by parameter (spline metamodel, two parameters):")
    for p in ranking.index:
        print(f"  {p:>3}  {ranking[p]:>10,.0f}")
    print(f"  ranking: {' > '.join(ranking.index)}")

    # --- EVSI: the value of a proposed effect study, by sample size ---------
    sizes = (25, 50, 100, 200, 400, 800)
    est_by_size = {}
    print("\nEVSI of a two-arm effect study, per person:")
    print(f"  {'n/arm':>6}  {'estimate':>10}")
    for n_trial in sizes:
        tau = SIGMA / np.sqrt(n_trial)
        rng = np.random.default_rng(n_trial)
        summaries = simulate_summaries(
            draws, lambda row, r, t=tau: {"xbar": row["dq"] + r.normal(0.0, t)}, seed=rng
        )
        est = evsi_regression(outcomes, summaries, WTP)
        est_by_size[n_trial] = est
        print(f"  {n_trial:>6}  {est:>10,.0f}")

    # --- expected net benefit of sampling, with the sample size to fund -----
    evsi = pd.Series(est_by_size)
    years = np.arange(10)
    beneficiaries = 2_000 * (1.035**-years).sum()  # discounted future patients
    cost = 300_000 + 6_000 * 2 * evsi.index  # 2n participants over two arms
    enbs = beneficiaries * evsi - cost  # population EVSI minus study cost
    best = enbs.idxmax()
    print(
        "\nENBS: 2,000 patients a year over 10 years, 300,000 + 6,000 per participant."
        f"\nBest size {best} per arm, ENBS {enbs[best]:,.0f}."
    )


if __name__ == "__main__":
    main()
