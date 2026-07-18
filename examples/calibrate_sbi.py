"""Calibrate the same Markov model with simulation-based inference on the model itself.

This is the second of four calibration examples that share one disease model. It repeats
the calibration of `calibrate_abc.py`, on the same three-state Markov model, the same
truth, and the same observed survey, but with a different inference method: neural
posterior estimation, one form of simulation-based inference (SBI).

Neural posterior estimation trains a conditional density estimator on parameter-and-
output pairs drawn from the prior, then evaluates it at the observed target to return the
posterior, without an explicit likelihood. It reaches the same posterior ABC does, which
is the point of running it here: the two methods agree when both use the model directly.

The cost is in the training set. The density estimator needs many simulated pairs, and
every pair is a model run, so this method spends about ten thousand model runs. The third
example replaces those runs with a surrogate.

Run it with::

    uv pip install -e '.[surrogate]'
    uv run python examples/calibrate_sbi.py

Outputs (written to ``examples/output/``):
    - calibrate_sbi_posterior.png
    - run_report_calibrate_sbi.md
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("sbi").setLevel(logging.WARNING)

import torch  # noqa: E402
from sbi.inference import NPE  # noqa: E402
from sbi.utils import BoxUniform  # noqa: E402

from heormodel.models import CohortSpec, MarkovModel  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("healthy", "sick", "dead")
INTERVENTION = "natural_history"
N_CYCLES = 40
TARGET_CYCLES = (8, 16, 28)
TARGET_LABELS = [f"sick_c{cycle}" for cycle in TARGET_CYCLES]
BACKGROUND_MORTALITY = 0.01

CALIBRATED = ("p_HS", "p_SD")
BOUNDS = {"p_HS": (0.02, 0.20), "p_SD": (0.05, 0.35)}
TRUTH = {"p_HS": 0.08, "p_SD": 0.15}

SURVEY_SIZE = 1_000
SURVEY_SEED = 20260718
TRAINING_SIMS = 10_000  # model runs used to train the density estimator
POSTERIOR_DRAWS = 3_000


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


def observed_survey() -> np.ndarray:
    """The single observed survey, drawn once from the truth with a fixed seed."""
    rng = np.random.default_rng(SURVEY_SEED)
    return rng.binomial(SURVEY_SIZE, prevalence(TRUTH)) / SURVEY_SIZE


def main() -> None:
    OUT.mkdir(exist_ok=True)
    low = np.array([BOUNDS[name][0] for name in CALIBRATED])
    high = np.array([BOUNDS[name][1] for name in CALIBRATED])

    observed = observed_survey()
    print("Observed survey:", observed.round(4))

    torch.manual_seed(0)
    survey_rng = np.random.default_rng(11)
    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32),
        high=torch.tensor(high, dtype=torch.float32),
    )

    # Each training pair runs the model once and takes a survey of its prevalence, so the
    # density estimator learns the same survey likelihood ABC targets.
    def simulator(theta: torch.Tensor) -> torch.Tensor:
        rows = theta.numpy()
        means = np.array([prevalence(dict(zip(CALIBRATED, row, strict=True))) for row in rows])
        survey = survey_rng.binomial(SURVEY_SIZE, means) / SURVEY_SIZE
        return torch.tensor(survey, dtype=torch.float32)

    model_runs["count"] = 0
    theta = prior.sample((TRAINING_SIMS,))
    x = simulator(theta)
    training_runs = model_runs["count"]
    inference = NPE(prior=prior, show_progress_bars=False)
    with contextlib.redirect_stdout(io.StringIO()):  # quiet sbi's training progress print
        inference.append_simulations(theta, x).train()
    neural_posterior = inference.build_posterior()
    samples = neural_posterior.sample(
        (POSTERIOR_DRAWS,),
        x=torch.tensor(observed, dtype=torch.float32),
        show_progress_bars=False,
    )
    posterior = pd.DataFrame(samples.numpy(), columns=list(CALIBRATED))
    print(f"\nNeural posterior estimation: {training_runs} model runs to train.")

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
        axis.hist(posterior[name], bins=40, density=True, alpha=0.6, color="#8c2d04")
        axis.axvline(TRUTH[name], color="black", linestyle="--", linewidth=1, label="truth")
        axis.set_xlabel(name)
        axis.set_yticks([])
    axes[0].set_ylabel("posterior density")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Neural posterior estimation on the model")
    fig.tight_layout()
    fig.savefig(OUT / "calibrate_sbi_posterior.png", dpi=150)
    plt.close(fig)

    (OUT / "run_report_calibrate_sbi.md").write_text(
        "# Neural posterior estimation on the model\n\n"
        f"Training runs: {training_runs}. Observed "
        f"{dict(zip(TARGET_LABELS, observed.round(4), strict=True))}.\n\n"
        + summary.round(4).to_markdown()
        + "\n"
    )
    print(f"\nWrote posterior figure and run report to {OUT}/")


if __name__ == "__main__":
    main()
