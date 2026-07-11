"""Cross-validate the Markov cohort and microsimulation engines, then diverge them.

Builds one progressive Sick-Sicker-style model twice, from the same transition
rates, rewards, and horizon: as a `MarkovModel` cohort trace and as a
`MicrosimModel` individual simulation. The script makes three points in order.

1. Identical assumptions converge. With a memoryless, homogeneous population the
   microsimulation mean cost and QALYs approach the cohort trace as the
   population grows. The convergence sweep and its plot are the cross-validation:
   two independent implementations agree, so both are trusted.
2. Heterogeneity breaks the cohort assumption. A per-individual frailty
   multiplier on the progression and mortality hazards, with mean 1 so the mean
   hazard is unchanged, moves the microsimulation mean away from the cohort. The
   average of a non-linear survival curve is not the survival curve of the
   average hazard.
3. What else the microsimulation buys. History dependence (mortality that rises
   with time spent sick, through ``duration_groups``) that a cohort carries only
   by expanding its state space.

Run it with::

    uv run python examples/markov_vs_microsim.py

Outputs (written to ``examples/output/``):
    - convergence_markov_vs_microsim.png
    - run_report_markov_vs_microsim.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heormodel.models import CohortSpec, MarkovModel, MicrosimModel
from heormodel.report import capture_run
from heormodel.run import SeedManager, run_psa

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("H", "S1", "S2", "D")
STRATEGY = "Standard of care"
N_CYCLES = 40  # annual cycles, ages 40 to 80
DISCOUNT = 0.03

# One base-case parameter set drives both engines. Rates are annual hazards.
BASE = dict(
    r_HS1=0.12,  # onset, Healthy to Sick
    r_S1S2=0.10,  # progression, Sick to Sicker
    r_HD=0.010,  # background mortality hazard
    hr_S1=3.0,  # mortality rate ratio in Sick
    hr_S2=10.0,  # mortality rate ratio in Sicker
    c_H=1_000.0, c_S1=4_000.0, c_S2=15_000.0,
    u_H=1.0, u_S1=0.75, u_S2=0.5,
)

# Frailty variance for the heterogeneous run. Gamma frailty with mean 1 and this
# variance leaves the mean hazard unchanged but spreads individual hazards.
FRAILTY_VAR = 0.75


def _competing_risks(hazards: np.ndarray) -> np.ndarray:
    """Per-cycle transition probabilities from competing annual hazards.

    ``hazards`` is ``(n, n_states)`` of out-hazards from each individual's
    current state (the diagonal to itself is 0). Rows return a full transition
    row: the stay probability plus the destination split.
    """
    total = hazards.sum(axis=1)
    p_leave = 1.0 - np.exp(-total)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total[:, None] > 0, hazards / total[:, None], 0.0)
    probs = p_leave[:, None] * share
    return probs


def _hazards_from(params: pd.Series, state: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Out-hazard matrix for each individual, scaled by frailty ``z``.

    Frailty multiplies the progression and mortality hazards; onset is fixed.
    """
    n = len(state)
    haz = np.zeros((n, 4))
    r_HD = params["r_HD"]
    # Healthy: onset (fixed) and background death (frail).
    h = state == 0
    haz[h, 1] = params["r_HS1"]
    haz[h, 3] = r_HD * z[h]
    # Sick: progression and death, both frail.
    s1 = state == 1
    haz[s1, 2] = params["r_S1S2"] * z[s1]
    haz[s1, 3] = r_HD * params["hr_S1"] * z[s1]
    # Sicker: death only, frail.
    s2 = state == 2
    haz[s2, 3] = r_HD * params["hr_S2"] * z[s2]
    return haz


def cohort_model(params: pd.Series, strategy: str) -> CohortSpec:
    """Transition matrix and per-state rewards for the cohort trace (z = 1)."""
    state = np.arange(4)
    P = _competing_risks(_hazards_from(params, state, np.ones(4)))
    P[np.arange(4), np.arange(4)] += 1.0 - P.sum(axis=1)  # stay mass on the diagonal
    P[3] = np.array([0.0, 0.0, 0.0, 1.0])  # Dead is absorbing
    cost = np.array([params["c_H"], params["c_S1"], params["c_S2"], 0.0])
    effect = np.array([params["u_H"], params["u_S1"], params["u_S2"], 0.0])
    return CohortSpec(P, cost, effect)


def make_population(frailty_var: float):
    """Population sampler drawing a mean-1 Gamma frailty, or z = 1 when var = 0."""

    def population(rng: np.random.Generator, n: int) -> pd.DataFrame:
        if frailty_var == 0.0:
            z = np.ones(n)
        else:
            k = 1.0 / frailty_var
            z = rng.gamma(k, 1.0 / k, n)
        return pd.DataFrame({"z": z})

    return population


def micro_transition(
    params: pd.Series, state: np.ndarray, attrs: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Per-individual transition rows, frailty read from the ``z`` attribute."""
    z = attrs["z"].to_numpy()
    probs = _competing_risks(_hazards_from(params, state, z))
    stay = 1.0 - probs.sum(axis=1)
    probs[np.arange(len(state)), state] += stay  # remaining mass stays in place
    probs[state == 3] = np.array([0.0, 0.0, 0.0, 1.0])
    return probs


def micro_payoffs(
    params: pd.Series, state: np.ndarray, attrs: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cycle cost and utility of each individual's current state."""
    cost_vec = np.array([params["c_H"], params["c_S1"], params["c_S2"], 0.0])
    eff_vec = np.array([params["u_H"], params["u_S1"], params["u_S2"], 0.0])
    return cost_vec[state], eff_vec[state]


def build_microsim(
    n_individuals: int, frailty_var: float, seeds: SeedManager,
    *, transition_probabilities=micro_transition, **kwargs,
) -> MicrosimModel:
    """A microsimulation twin of the cohort model at one population size."""
    return MicrosimModel(
        states=STATES,
        transition_probabilities=transition_probabilities,
        state_costs_and_utilities=micro_payoffs,
        population=make_population(frailty_var),
        n_individuals=n_individuals,
        strategies={STRATEGY: {}},
        horizon=N_CYCLES,
        discount_rate=DISCOUNT,
        half_cycle_correction=True,
        seed_manager=seeds,
        **kwargs,
    )


def _draws() -> pd.DataFrame:
    return pd.DataFrame([BASE], index=pd.RangeIndex(1, name="iteration"))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    draws = _draws()

    cohort = MarkovModel(
        states=STATES, strategies=(STRATEGY,), model_fn=cohort_model,
        n_cycles=N_CYCLES, start="H", discount_rate=DISCOUNT,
        half_cycle_correction="half-cycle",
    )
    cohort_out = cohort.evaluate(draws).summary().loc[STRATEGY]
    c_cost, c_qaly = float(cohort_out["cost"]), float(cohort_out["qaly"])
    print("Cohort trace (Markov):")
    print(f"  cost {c_cost:,.1f}  QALYs {c_qaly:.4f}")

    # 1. Convergence of the homogeneous microsimulation to the cohort trace.
    sizes = [1_000, 5_000, 20_000, 80_000]
    print("\nHomogeneous microsimulation vs cohort trace:")
    print(f"  {'n':>8}  {'cost':>12}  {'QALYs':>9}  {'QALY gap %':>11}")
    micro_cost, micro_qaly = [], []
    for n in sizes:
        seeds = SeedManager(20260705)
        out = run_psa(build_microsim(n, 0.0, seeds), draws, sequential=True)
        row = out.summary().loc[STRATEGY]
        micro_cost.append(float(row["cost"]))
        micro_qaly.append(float(row["qaly"]))
        gap = 100.0 * (micro_qaly[-1] - c_qaly) / c_qaly
        print(f"  {n:>8,}  {micro_cost[-1]:>12,.1f}  {micro_qaly[-1]:>9.4f}  {gap:>10.2f}%")

    # 2. Heterogeneity: a mean-1 frailty moves the microsimulation off the cohort.
    seeds = SeedManager(20260705)
    het = run_psa(build_microsim(80_000, FRAILTY_VAR, seeds), draws, sequential=True)
    h_row = het.summary().loc[STRATEGY]
    h_cost, h_qaly = float(h_row["cost"]), float(h_row["qaly"])
    print(f"\nHeterogeneous microsimulation (frailty variance {FRAILTY_VAR}, n=80,000):")
    print(f"  cost {h_cost:,.1f}  QALYs {h_qaly:.4f}")
    print(f"  QALY shift from cohort: {100.0 * (h_qaly - c_qaly) / c_qaly:+.2f}%")
    print(f"  cost shift from cohort: {100.0 * (h_cost - c_cost) / c_cost:+.2f}%")

    # 3. History dependence: mortality rising with time spent sick, cohort cannot
    #    hold it without tunnel states.
    seeds = SeedManager(20260705)

    def hist_transition(params, state, attrs, rng):
        z = attrs["z"].to_numpy()
        tis = attrs["tis"].to_numpy()
        haz = _hazards_from(params, state, z)
        rising = 1.0 + 0.08 * tis  # 8% higher death hazard per sick year
        haz[:, 3] *= np.where(np.isin(state, (1, 2)), rising, 1.0)
        probs = _competing_risks(haz)
        probs[np.arange(len(state)), state] += 1.0 - probs.sum(axis=1)
        probs[state == 3] = np.array([0.0, 0.0, 0.0, 1.0])
        return probs

    hist = build_microsim(
        80_000, FRAILTY_VAR, seeds,
        transition_probabilities=hist_transition, duration_groups={"tis": ("S1", "S2")},
    )
    hist_out = run_psa(hist, draws, sequential=True)
    hist_row = hist_out.summary().loc[STRATEGY]
    print(f"\nWith duration-dependent mortality (n=80,000, frailty {FRAILTY_VAR}):")
    print(f"  cost {float(hist_row['cost']):,.1f}  QALYs {float(hist_row['qaly']):.4f}")

    # Convergence plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(c_qaly, color="black", lw=1.2, ls="--", label="Cohort trace")
    ax.plot(sizes, micro_qaly, "o-", color="#2a6f97", label="Microsimulation mean")
    ax.set_xscale("log")
    ax.set_xlabel("Population size (log scale)")
    ax.set_ylabel("Discounted QALYs")
    ax.set_title("Homogeneous microsimulation converges to the cohort trace")
    ax.legend()
    fig.savefig(OUT / "convergence_markov_vs_microsim.png", dpi=150, bbox_inches="tight")

    record = capture_run(
        seed=SeedManager(20260705),
        outcomes=het,
        draw_sources=dict.fromkeys(draws.columns, "Illustrative Sick-Sicker parameters"),
        note=(
            f"Markov cohort vs microsimulation cross-validation, {N_CYCLES} annual "
            f"cycles. Homogeneous microsimulation converges to the cohort trace; a "
            f"mean-1 frailty (variance {FRAILTY_VAR}) shifts QALYs off it."
        ),
    )
    (OUT / "run_report_markov_vs_microsim.md").write_text(
        record.to_markdown("Markov vs microsimulation run report")
    )
    print(f"\nWrote plot and run report to {OUT}/")


if __name__ == "__main__":
    main()
