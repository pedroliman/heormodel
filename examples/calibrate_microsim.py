"""Calibrate a stochastic microsimulation through a surrogate that carries its noise.

This is the fourth of four calibration examples that share one disease model. The first
three calibrated a deterministic cohort model, where the only uncertainty was the survey
that measured the target. This example replaces that model with its individual-level
twin: a microsimulation of the same three states and the same transition probabilities,
which returns a different prevalence every time it runs because it samples a finite
population. It calibrates that stochastic model and shows the posterior is wider than the
deterministic one, by an amount that tracks the model's own sampling noise.

The microsimulation is expensive, so its whole run budget is a small design: 60 parameter
sets, each run a few times. A Gaussian process is fit to those noisy runs. Its mean
estimates the prevalence surface, and its predictive spread carries the microsimulation's
replicate noise, so drawing from the surrogate reproduces the model's own variability
without running it again.

The calibration then compares two posteriors against the same observed survey:

1. Survey only. The surrogate is treated as an exact prevalence surface, so the only
   noise is the survey, as in the deterministic examples.
2. Survey and model. Each surrogate evaluation is a draw from the Gaussian process
   predictive distribution, so the model's replicate noise enters alongside the survey.

The second posterior is centered on the same truth but wider, because the finite
population the microsimulation samples adds uncertainty the deterministic model did not
have. Running a larger population narrows the gap.

Run it with::

    uv pip install -e '.[surrogate]'
    uv run python examples/calibrate_microsim.py

Outputs (written to ``examples/output/``):
    - calibrate_microsim_posteriors.png
    - run_report_calibrate_microsim.md
"""

from __future__ import annotations

import contextlib
import io
import logging
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

logging.getLogger("sbi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from heormodel.models import (  # noqa: E402
    CohortSpec,
    MarkovModel,
    MicrosimModel,
    state_occupancy,
)
from heormodel.run import run_psa  # noqa: E402

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
POPULATION = 2_000  # individuals per microsimulation run; smaller means noisier runs
LHS_POINTS = 60
REPLICATES = 10  # microsimulation runs per design point
SURROGATE_SIMS = 10_000
POSTERIOR_DRAWS = 3_000
RUN_SEED = 123


def transitions_and_rewards(params: pd.Series, intervention: str) -> CohortSpec:
    """Cohort transition matrix; the deterministic twin of the microsimulation."""
    p_HS, p_SD = params["p_HS"], params["p_SD"]
    transition = np.array(
        [
            [1.0 - p_HS - BACKGROUND_MORTALITY, p_HS, BACKGROUND_MORTALITY],
            [0.0, 1.0 - p_SD, p_SD],
            [0.0, 0.0, 1.0],
        ]
    )
    return CohortSpec(transition, np.zeros(3), np.array([1.0, 0.8, 0.0]))


def micro_transition(params, intervention, state, attrs, rng):
    """Per-individual transition rows, the same probabilities for everyone in a state."""
    p_HS, p_SD = params["p_HS"], params["p_SD"]
    probs = np.zeros((len(state), 3))
    probs[state == 0] = [1.0 - p_HS - BACKGROUND_MORTALITY, p_HS, BACKGROUND_MORTALITY]
    probs[state == 1] = [0.0, 1.0 - p_SD, p_SD]
    probs[state == 2] = [0.0, 0.0, 1.0]
    return probs


def micro_rewards(params, intervention, state, attrs):
    """Rewards are unused by calibration; return zeros to satisfy the engine."""
    zero = np.zeros(len(state))
    return zero, zero


cohort = MarkovModel(
    states=STATES,
    interventions=(INTERVENTION,),
    transitions_and_rewards=transitions_and_rewards,
    n_cycles=N_CYCLES,
    cycle_correction="none",
)


def deterministic_prevalence(params: dict[str, float]) -> np.ndarray:
    """Sick-state prevalence of the deterministic cohort twin at the target cycles."""
    occupancy = cohort.trace(pd.Series(params), INTERVENTION)["sick"].to_numpy()
    return np.array([occupancy[cycle] for cycle in TARGET_CYCLES])


def microsim_prevalence(param_rows: list[dict[str, float]], population: int) -> np.ndarray:
    """Sick prevalence from one microsimulation run per row, shape (len(rows), 3)."""
    engine = MicrosimModel.discrete(
        states=STATES,
        transition_probabilities=micro_transition,
        state_rewards=micro_rewards,
        population=population,
        interventions=[INTERVENTION],
        n_cycles=N_CYCLES,
        cycle_correction="none",
        initial_state="healthy",
    )
    draws = pd.DataFrame(param_rows)
    draws.index = pd.RangeIndex(len(draws), name="iteration")
    events = run_psa(engine, draws, seed=RUN_SEED, collect="events").events
    occupancy = state_occupancy(
        events, states=STATES, initial_state="healthy",
        n_individuals=population, times=[float(cycle) for cycle in TARGET_CYCLES],
    )
    result = np.zeros((len(draws), len(TARGET_CYCLES)))
    for iteration in range(len(draws)):
        for column, cycle in enumerate(TARGET_CYCLES):
            result[iteration, column] = occupancy.loc[
                (INTERVENTION, iteration, float(cycle)), "sick"
            ]
    return result


def to_unit(points: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map parameter points into the unit box so one length-scale bound fits both axes."""
    return (points - low) / (high - low)


def fit_surrogates(unit_points, design_targets):
    """One Gaussian process per target on unit-box inputs, with a replicate-noise term.

    Bounding the length scale keeps the marginal-likelihood optimizer from collapsing it
    onto the replicate noise, which a noisy design otherwise invites.
    """
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF([0.3, 0.3], length_scale_bounds=(0.05, 2.0))
        + WhiteKernel(1e-3, (1e-6, 1e-1))
    )
    return [
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=8).fit(
            unit_points, design_targets[:, target]
        )
        for target in range(len(TARGET_LABELS))
    ]


def npe_posterior(simulator, observed, prior, sims=SURROGATE_SIMS):
    """Neural posterior estimation with the given (stochastic) surrogate simulator."""
    torch.manual_seed(0)
    theta = prior.sample((sims,))
    inference = NPE(prior=prior, show_progress_bars=False)
    with contextlib.redirect_stdout(io.StringIO()):
        inference.append_simulations(theta, simulator(theta)).train()
    samples = inference.build_posterior().sample(
        (POSTERIOR_DRAWS,),
        x=torch.tensor(observed, dtype=torch.float32),
        show_progress_bars=False,
    )
    return pd.DataFrame(samples.numpy(), columns=list(CALIBRATED))


def calibrate_at_population(population, low, high, prior, observed, sims=SURROGATE_SIMS):
    """Fit a surrogate to microsimulation runs at one population, then infer two posteriors.

    Returns the surrogate hold-out RMSE against the deterministic surface, the
    survey-only posterior, and the posterior that also carries the surrogate predictive
    spread (the microsimulation's replicate noise).
    """
    unit_design = LatinHypercube(d=len(CALIBRATED), seed=7).random(LHS_POINTS)
    design = pd.DataFrame(scale(unit_design, low, high), columns=list(CALIBRATED))
    design_rows = [row for row in design.to_dict("records") for _ in range(REPLICATES)]
    unit_points = np.repeat(unit_design, REPLICATES, axis=0)
    design_targets = microsim_prevalence(design_rows, population=population)
    surrogates = fit_surrogates(unit_points, design_targets)

    holdout = scale(LatinHypercube(d=len(CALIBRATED), seed=99).random(200), low, high)
    holdout_targets = np.array(
        [deterministic_prevalence(dict(zip(CALIBRATED, row, strict=True))) for row in holdout]
    )
    predicted = np.column_stack([gp.predict(to_unit(holdout, low, high)) for gp in surrogates])
    rmse = np.sqrt(((predicted - holdout_targets) ** 2).mean(axis=0))

    survey_only_rng = np.random.default_rng(11)

    def survey_only_simulator(theta: torch.Tensor) -> torch.Tensor:
        unit = to_unit(theta.numpy(), low, high)
        mean = np.column_stack([gp.predict(unit) for gp in surrogates])
        survey = survey_only_rng.binomial(SURVEY_SIZE, np.clip(mean, 0, 1)) / SURVEY_SIZE
        return torch.tensor(survey, dtype=torch.float32)

    model_survey_rng = np.random.default_rng(12)

    def model_and_survey_simulator(theta: torch.Tensor) -> torch.Tensor:
        unit = to_unit(theta.numpy(), low, high)
        columns = [gp.predict(unit, return_std=True) for gp in surrogates]
        mean = np.column_stack([col[0] for col in columns])
        spread = np.column_stack([col[1] for col in columns])
        model_draw = mean + model_survey_rng.normal(0.0, 1.0, mean.shape) * spread
        survey = model_survey_rng.binomial(SURVEY_SIZE, np.clip(model_draw, 0, 1)) / SURVEY_SIZE
        return torch.tensor(survey, dtype=torch.float32)

    survey_only = npe_posterior(survey_only_simulator, observed, prior, sims=sims)
    model_survey = npe_posterior(model_and_survey_simulator, observed, prior, sims=sims)
    return rmse, survey_only, model_survey


def main() -> None:
    OUT.mkdir(exist_ok=True)
    low = np.array([BOUNDS[name][0] for name in CALIBRATED])
    high = np.array([BOUNDS[name][1] for name in CALIBRATED])

    true_prevalence = deterministic_prevalence(TRUTH)
    survey_rng = np.random.default_rng(SURVEY_SEED)
    observed = survey_rng.binomial(SURVEY_SIZE, true_prevalence) / SURVEY_SIZE
    print("Observed survey:", observed.round(4))

    # Confirm the microsimulation is the same model: its mean approaches the cohort trace.
    check = microsim_prevalence([TRUTH] * 40, population=POPULATION)
    print(f"Microsim mean at truth (population {POPULATION}, 40 runs):", check.mean(0).round(4))
    print("Deterministic prevalence at truth:            ", true_prevalence.round(4))
    print("Microsim replicate SD per target:             ", check.std(0).round(4))
    print(f"\nDesign: {LHS_POINTS} points x {REPLICATES} replicates ="
          f" {LHS_POINTS * REPLICATES} microsimulation runs per population.")

    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32),
        high=torch.tensor(high, dtype=torch.float32),
    )

    # Calibrate at the base population, and again at a larger one to show the extra width
    # narrow. The base run also supplies the posteriors the figure below draws.
    results = {
        population: calibrate_at_population(population, low, high, prior, observed)
        for population in (POPULATION, 8_000)
    }
    rmse, survey_only, model_survey = results[POPULATION]
    print("Surrogate-mean vs deterministic hold-out RMSE:", rmse.round(5))

    summary = pd.DataFrame(
        {
            "truth": [TRUTH[name] for name in CALIBRATED],
            "survey_only_mean": survey_only.mean().reindex(CALIBRATED).to_numpy(),
            "survey_only_sd": survey_only.std().reindex(CALIBRATED).to_numpy(),
            "model_survey_mean": model_survey.mean().reindex(CALIBRATED).to_numpy(),
            "model_survey_sd": model_survey.std().reindex(CALIBRATED).to_numpy(),
        },
        index=list(CALIBRATED),
    )
    widening = (summary["model_survey_sd"] / summary["survey_only_sd"]).to_numpy()
    print(f"\nPosterior comparison at population {POPULATION}:\n", summary.round(4))
    print("Posterior SD ratio (model+survey / survey only):", widening.round(2))

    # A larger population narrows the extra width, because the replicate noise the
    # surrogate carries shrinks as the population grows, toward the survey-only floor.
    sweep = pd.DataFrame(
        [
            {
                "population": population,
                "p_HS_sd": float(model_post["p_HS"].std()),
                "p_SD_sd": float(model_post["p_SD"].std()),
            }
            for population, (_, _, model_post) in results.items()
        ]
    ).set_index("population")
    print("\nModel+survey posterior SD as the microsimulation population grows:")
    print(sweep.round(4))
    print("Survey-only floor:", summary["survey_only_sd"].round(4).to_dict())

    fig, axes = plt.subplots(1, len(CALIBRATED), figsize=(9, 3.5))
    for axis, name in zip(axes, CALIBRATED, strict=True):
        axis.hist(survey_only[name], bins=40, density=True, alpha=0.5, label="survey only")
        axis.hist(model_survey[name], bins=40, density=True, alpha=0.5, label="survey + model")
        axis.axvline(TRUTH[name], color="black", linestyle="--", linewidth=1, label="truth")
        axis.set_xlabel(name)
        axis.set_yticks([])
    axes[0].set_ylabel("posterior density")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Model replicate noise widens the posterior")
    fig.tight_layout()
    fig.savefig(OUT / "calibrate_microsim_posteriors.png", dpi=150)
    plt.close(fig)

    (OUT / "run_report_calibrate_microsim.md").write_text(
        "# Calibrating a stochastic microsimulation through a surrogate\n\n"
        f"Population {POPULATION} per run, {LHS_POINTS * REPLICATES} microsimulation runs "
        f"for the design. Observed "
        f"{dict(zip(TARGET_LABELS, observed.round(4), strict=True))}.\n\n"
        + summary.round(4).to_markdown()
        + f"\n\nPosterior SD ratio (model+survey / survey only): "
        f"{dict(zip(CALIBRATED, widening.round(2), strict=True))}\n\n"
        "Model+survey posterior SD as population grows:\n\n"
        + sweep.round(4).to_markdown()
        + "\n"
    )
    print(f"\nWrote posterior figure and run report to {OUT}/")


if __name__ == "__main__":
    main()
