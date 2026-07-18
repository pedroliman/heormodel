"""Calibrate a Markov model against a survey target with approximate Bayesian computation.

This is the first of four calibration examples that share one disease model, so their
posteriors can be read side by side. It calibrates two transition probabilities of a
three-state cohort state-transition (Markov) model to a prevalence survey, using
approximate Bayesian computation (ABC).

The model runs over Healthy, Sick, and Dead for 40 annual cycles. Two per-cycle
transition probabilities are unknown:

    p_HS: Healthy -> Sick probability
    p_SD: Sick -> Dead probability

The calibration target is the Sick-state prevalence at cycles 8, 16, and 28, as it would
be measured by a cross-sectional survey of a finite number of people. The survey is the
only source of uncertainty here: the model is deterministic, so a given parameter set
gives one prevalence curve, and what is uncertain is the count a survey of that curve
returns. The observed target is drawn once from a survey of 1000 people at each cycle,
from known truth p_HS = 0.08 and p_SD = 0.15, so the calibration has a right answer to
recover.

The three other examples reuse the same model, the same truth, and the same observed
survey (`calibrate_sbi.py`, `surrogate_calibration.py`, `calibrate_microsim.py`).

Run it with::

    uv pip install -e '.[calibration]'
    uv run python examples/calibrate_abc.py

Outputs (written to ``examples/output/``):
    - calibrate_abc_posterior.png
    - run_report_calibrate_abc.md
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["ABC_LOG_LEVEL"] = "WARNING"  # quiet pyabc's per-population logging

from heormodel.calibrate import abc_calibrate  # noqa: E402
from heormodel.models import CohortSpec, MarkovModel  # noqa: E402
from heormodel.params import Uniform  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("healthy", "sick", "dead")
INTERVENTION = "natural_history"
N_CYCLES = 40
TARGET_CYCLES = (8, 16, 28)
TARGET_LABELS = [f"sick_c{cycle}" for cycle in TARGET_CYCLES]
BACKGROUND_MORTALITY = 0.01  # fixed Healthy -> Dead per-cycle probability

CALIBRATED = ("p_HS", "p_SD")
BOUNDS = {"p_HS": (0.02, 0.20), "p_SD": (0.05, 0.35)}
TRUTH = {"p_HS": 0.08, "p_SD": 0.15}

SURVEY_SIZE = 1_000  # people surveyed for each prevalence reading
SURVEY_SEED = 20260718  # fixes the one observed survey, shared across the four examples


def transitions_and_rewards(params: pd.Series, intervention: str) -> CohortSpec:
    """Transition matrix for one parameter set; rewards are unused by calibration."""
    p_HS, p_SD = params["p_HS"], params["p_SD"]
    transition = np.array(
        [
            [1.0 - p_HS - BACKGROUND_MORTALITY, p_HS, BACKGROUND_MORTALITY],
            [0.0, 1.0 - p_SD, p_SD],
            [0.0, 0.0, 1.0],
        ]
    )
    return CohortSpec(transition, np.zeros(3), np.array([1.0, 0.8, 0.0]))


engine = MarkovModel(
    states=STATES,
    interventions=(INTERVENTION,),
    transitions_and_rewards=transitions_and_rewards,
    n_cycles=N_CYCLES,
    cycle_correction="none",
)


model_runs = {"count": 0}


def prevalence(params: dict[str, float]) -> np.ndarray:
    """Sick-state prevalence the deterministic model predicts at the target cycles."""
    model_runs["count"] += 1
    occupancy = engine.trace(pd.Series(params), INTERVENTION)["sick"].to_numpy()
    return np.array([occupancy[cycle] for cycle in TARGET_CYCLES])


def draw_survey(prevalences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One survey of SURVEY_SIZE people at each cycle: a binomial count over the size."""
    return rng.binomial(SURVEY_SIZE, prevalences) / SURVEY_SIZE


def observed_survey() -> np.ndarray:
    """The single observed survey, drawn once from the truth with a fixed seed."""
    return draw_survey(prevalence(TRUTH), np.random.default_rng(SURVEY_SEED))


def survey_epsilon(prevalences: np.ndarray) -> float:
    """Euclidean acceptance floor at the survey's own sampling scale."""
    variance = prevalences * (1.0 - prevalences) / SURVEY_SIZE
    return float(np.sqrt(variance.sum()))


def main() -> None:
    OUT.mkdir(exist_ok=True)

    true_prevalence = prevalence(TRUTH)
    observed = observed_survey()
    print("True prevalence:    ", true_prevalence.round(4))
    print("Observed survey:    ", observed.round(4))
    survey_se = np.sqrt(true_prevalence * (1 - true_prevalence) / SURVEY_SIZE)
    print("Survey SE per target:", survey_se.round(4))

    # The simulator mirrors the survey: it predicts the prevalence for a candidate
    # parameter set and returns a fresh survey of that prevalence. ABC then recovers the
    # parameters given a survey reading, propagating the survey's sampling error into the
    # posterior rather than treating the observed numbers as exact.
    sim_rng = np.random.default_rng(2024)

    def simulator(params: dict[str, float]) -> dict[str, float]:
        survey = draw_survey(prevalence(params), sim_rng)
        return dict(zip(TARGET_LABELS, survey, strict=True))

    # Annealing the acceptance threshold to half the single-survey scale sharpens the
    # posterior below the width one survey would give, because the simulator draws its
    # own survey with the same sampling error as the data.
    model_runs["count"] = 0
    result = abc_calibrate(
        simulator,
        priors={name: Uniform(*BOUNDS[name]) for name in CALIBRATED},
        observed=dict(zip(TARGET_LABELS, observed, strict=True)),
        population_size=400,
        max_populations=15,
        min_epsilon=0.5 * survey_epsilon(true_prevalence),
        n_posterior=3_000,
        seed=1,
    )
    abc_runs = model_runs["count"]
    posterior = result.posterior
    print(f"\nABC over {result.n_populations} populations, final epsilon "
          f"{result.final_epsilon:.4f}, {abc_runs} model runs.")
    summary = pd.DataFrame(
        {
            "truth": [TRUTH[name] for name in CALIBRATED],
            "posterior_mean": posterior.mean().reindex(CALIBRATED).to_numpy(),
            "posterior_sd": posterior.std().reindex(CALIBRATED).to_numpy(),
        },
        index=list(CALIBRATED),
    )
    print("\nPosterior:\n", summary.round(4))

    fig, axes = plt.subplots(1, len(CALIBRATED), figsize=(9, 3.5))
    for axis, name in zip(axes, CALIBRATED, strict=True):
        axis.hist(posterior[name], bins=40, density=True, alpha=0.6, color="#2a6f97")
        axis.axvline(TRUTH[name], color="black", linestyle="--", linewidth=1, label="truth")
        axis.set_xlabel(name)
        axis.set_yticks([])
    axes[0].set_ylabel("posterior density")
    axes[-1].legend(fontsize=8)
    fig.suptitle("ABC posterior against a survey target")
    fig.tight_layout()
    fig.savefig(OUT / "calibrate_abc_posterior.png", dpi=150)
    plt.close(fig)

    (OUT / "run_report_calibrate_abc.md").write_text(
        "# ABC calibration against a survey target\n\n"
        f"Survey of {SURVEY_SIZE} people per reading; observed "
        f"{dict(zip(TARGET_LABELS, observed.round(4), strict=True))}. "
        f"{abc_runs} model runs.\n\n"
        + summary.round(4).to_markdown()
        + "\n"
    )
    print(f"\nWrote posterior figure and run report to {OUT}/")


if __name__ == "__main__":
    main()
