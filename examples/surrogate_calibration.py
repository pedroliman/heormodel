"""Calibrate through a surrogate, with both inference methods, and match the model.

This is the third of four calibration examples that share one disease model. The first
two calibrated the model directly, one with approximate Bayesian computation (ABC) and
one with neural posterior estimation, and reached the same posterior at a cost of
thousands to tens of thousands of model runs. This example replaces those runs with a
surrogate: it runs the model on a small fixed design, fits a Gaussian process to the
model's prevalence at each target cycle, and does the inference against the surrogate.

The example makes two points.

1. The surrogate is faithful. Both ABC and neural posterior estimation, run against the
   Gaussian process, reproduce the posterior a direct ABC run recovers against the model.
2. The two inference methods agree. Run on the same surrogate, ABC and neural posterior
   estimation give the same posterior, so the choice of method is separate from the
   choice to use a surrogate. This is the like-for-like comparison the first two examples
   set up: neither method is paired with a different model.

The surrogate reaches that posterior with about sixty model runs, against the thousands a
direct calibration takes. That gap is the reason to build a surrogate when a model is
slow to run, which the fourth example makes concrete with a microsimulation.

Run it with::

    uv pip install -e '.[calibration,surrogate]'
    uv run python examples/surrogate_calibration.py

Outputs (written to ``examples/output/``):
    - surrogate_calibration_posteriors.png
    - run_report_surrogate_calibration.md
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform
from scipy.stats.qmc import LatinHypercube, scale
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

os.environ["ABC_LOG_LEVEL"] = "WARNING"
logging.getLogger("sbi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
BACKGROUND_MORTALITY = 0.01

CALIBRATED = ("p_HS", "p_SD")
BOUNDS = {"p_HS": (0.02, 0.20), "p_SD": (0.05, 0.35)}
TRUTH = {"p_HS": 0.08, "p_SD": 0.15}

SURVEY_SIZE = 1_000
SURVEY_SEED = 20260718
LHS_POINTS = 60
SURROGATE_SIMS = 10_000
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


def draw_survey(prevalences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One followed registry of SURVEY_SIZE people: a binomial count over the size."""
    return rng.binomial(SURVEY_SIZE, prevalences) / SURVEY_SIZE


def survey_epsilon(prevalences: np.ndarray) -> float:
    """Euclidean acceptance floor at the survey's own sampling scale."""
    variance = prevalences * (1.0 - prevalences) / SURVEY_SIZE
    return float(np.sqrt(variance.sum()))


def posterior_summary(posteriors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean and standard deviation of each parameter under each posterior."""
    columns: dict[str, list[float]] = {"truth": [TRUTH[name] for name in CALIBRATED]}
    for label, draws in posteriors.items():
        columns[f"{label}_mean"] = draws.mean().reindex(CALIBRATED).to_numpy()
        columns[f"{label}_sd"] = draws.std().reindex(CALIBRATED).to_numpy()
    return pd.DataFrame(columns, index=list(CALIBRATED))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    low = np.array([BOUNDS[name][0] for name in CALIBRATED])
    high = np.array([BOUNDS[name][1] for name in CALIBRATED])

    true_prevalence = prevalence(TRUTH)
    observed = draw_survey(true_prevalence, np.random.default_rng(SURVEY_SEED))
    print("Observed survey:", observed.round(4))
    epsilon = 0.5 * survey_epsilon(true_prevalence)

    # --- Fit the surrogate on a small Latin hypercube design -----------------------
    model_runs["count"] = 0
    unit_design = LatinHypercube(d=len(CALIBRATED), seed=7).random(LHS_POINTS)
    design = pd.DataFrame(scale(unit_design, low, high), columns=list(CALIBRATED))
    design_targets = np.array([prevalence(row) for row in design.to_dict("records")])
    design_runs = model_runs["count"]
    print(f"Surrogate design: {design_runs} model runs.")

    kernel = ConstantKernel(1.0) * RBF([0.1, 0.1]) + WhiteKernel(1e-6, (1e-10, 1e-2))
    surrogates = [
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5).fit(
            design.to_numpy(), design_targets[:, target]
        )
        for target in range(len(TARGET_LABELS))
    ]

    # Hold-out accuracy on fresh model runs the surrogate never saw (not a calibration cost).
    holdout = scale(LatinHypercube(d=len(CALIBRATED), seed=99).random(300), low, high)
    holdout_targets = np.array(
        [prevalence(dict(zip(CALIBRATED, row, strict=True))) for row in holdout]
    )
    predicted = np.column_stack([surrogate.predict(holdout) for surrogate in surrogates])
    rmse = np.sqrt(((predicted - holdout_targets) ** 2).mean(axis=0))
    print("Surrogate hold-out RMSE per target:", rmse.round(5))

    def surrogate_prevalence(params: dict[str, float]) -> np.ndarray:
        point = np.array([[params[name] for name in CALIBRATED]])
        predicted = np.array([surrogate.predict(point)[0] for surrogate in surrogates])
        return np.clip(predicted, 0.0, 1.0)

    # --- ABC against the surrogate -------------------------------------------------
    abc_surrogate_rng = np.random.default_rng(2024)

    def abc_surrogate_simulator(params: dict[str, float]) -> dict[str, float]:
        survey = draw_survey(surrogate_prevalence(params), abc_surrogate_rng)
        return dict(zip(TARGET_LABELS, survey, strict=True))

    abc_surrogate = abc_calibrate(
        abc_surrogate_simulator,
        priors={name: Uniform(*BOUNDS[name]) for name in CALIBRATED},
        observed=dict(zip(TARGET_LABELS, observed, strict=True)),
        population_size=400,
        max_populations=15,
        min_epsilon=epsilon,
        n_posterior=POSTERIOR_DRAWS,
        seed=1,
    ).posterior

    # --- Neural posterior estimation against the surrogate -------------------------
    torch.manual_seed(0)
    sbi_survey_rng = np.random.default_rng(11)
    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32),
        high=torch.tensor(high, dtype=torch.float32),
    )

    def sbi_surrogate_simulator(theta: torch.Tensor) -> torch.Tensor:
        mean = np.column_stack([surrogate.predict(theta.numpy()) for surrogate in surrogates])
        survey = sbi_survey_rng.binomial(SURVEY_SIZE, np.clip(mean, 0, 1)) / SURVEY_SIZE
        return torch.tensor(survey, dtype=torch.float32)

    theta = prior.sample((SURROGATE_SIMS,))
    inference = NPE(prior=prior, show_progress_bars=False)
    with contextlib.redirect_stdout(io.StringIO()):
        inference.append_simulations(theta, sbi_surrogate_simulator(theta)).train()
    samples = inference.build_posterior().sample(
        (POSTERIOR_DRAWS,),
        x=torch.tensor(observed, dtype=torch.float32),
        show_progress_bars=False,
    )
    sbi_surrogate = pd.DataFrame(samples.numpy(), columns=list(CALIBRATED))

    # --- Direct ABC reference against the model ------------------------------------
    reference_rng = np.random.default_rng(2024)

    def reference_simulator(params: dict[str, float]) -> dict[str, float]:
        survey = draw_survey(prevalence(params), reference_rng)
        return dict(zip(TARGET_LABELS, survey, strict=True))

    model_runs["count"] = 0
    reference = abc_calibrate(
        reference_simulator,
        priors={name: Uniform(*BOUNDS[name]) for name in CALIBRATED},
        observed=dict(zip(TARGET_LABELS, observed, strict=True)),
        population_size=400,
        max_populations=15,
        min_epsilon=epsilon,
        n_posterior=POSTERIOR_DRAWS,
        seed=1,
    ).posterior
    reference_runs = model_runs["count"]

    # --- Compare -------------------------------------------------------------------
    summary = posterior_summary(
        {
            "reference": reference,
            "abc_surrogate": abc_surrogate,
            "sbi_surrogate": sbi_surrogate,
        }
    )
    print("\nPosterior comparison:\n", summary.round(4))
    print(f"\nModel-run budget: direct ABC {reference_runs}, surrogate design "
          f"{design_runs} ({reference_runs / design_runs:.0f} times fewer).")

    fig, axes = plt.subplots(1, len(CALIBRATED), figsize=(9, 3.5))
    for axis, name in zip(axes, CALIBRATED, strict=True):
        axis.hist(reference[name], bins=40, density=True, alpha=0.4, label="direct ABC")
        axis.hist(abc_surrogate[name], bins=40, density=True, alpha=0.4, label="ABC on surrogate")
        axis.hist(sbi_surrogate[name], bins=40, density=True, histtype="step",
                  linewidth=1.5, label="SBI on surrogate")
        axis.axvline(TRUTH[name], color="black", linestyle="--", linewidth=1, label="truth")
        axis.set_xlabel(name)
        axis.set_yticks([])
    axes[0].set_ylabel("posterior density")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Both methods on the surrogate match the direct posterior")
    fig.tight_layout()
    fig.savefig(OUT / "surrogate_calibration_posteriors.png", dpi=150)
    plt.close(fig)

    (OUT / "run_report_surrogate_calibration.md").write_text(
        "# Surrogate-accelerated calibration\n\n"
        f"Model-run budget: direct ABC {reference_runs}, surrogate design {design_runs}.\n\n"
        + summary.round(4).to_markdown()
        + "\n"
    )
    print(f"\nWrote posterior figure and run report to {OUT}/")


if __name__ == "__main__":
    main()
