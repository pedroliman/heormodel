"""A stochastic microsimulation calibrates, and its noise widens the posterior.

A reduced-size version of ``examples/calibrate_microsim.py``. It fits a Gaussian process
to noisy microsimulation runs, then checks two things: the posterior recovers the truth,
and the posterior that carries the model's replicate noise is wider than the one that
carries only the survey noise.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sbi")

logging.getLogger("sbi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from sbi.inference import NPE  # noqa: E402
from sbi.utils import BoxUniform  # noqa: E402
from scipy.stats.qmc import LatinHypercube, scale  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa: E402

from heormodel.models import (  # noqa: E402
    CohortSpec,
    MarkovModel,
    MicrosimModel,
    state_occupancy,
)
from heormodel.run import run_psa  # noqa: E402

STATES = ("healthy", "sick", "dead")
INTERVENTION = "natural_history"
N_CYCLES = 40
TARGET_CYCLES = (8, 16, 28)
BACKGROUND_MORTALITY = 0.01
CALIBRATED = ("p_HS", "p_SD")
BOUNDS = {"p_HS": (0.02, 0.20), "p_SD": (0.05, 0.35)}
TRUTH = {"p_HS": 0.08, "p_SD": 0.15}
SURVEY_SIZE = 1_000
POPULATION = 800


def _cohort_prevalence(engine, params):
    occupancy = engine.trace(pd.Series(params), INTERVENTION)["sick"].to_numpy()
    return np.array([occupancy[cycle] for cycle in TARGET_CYCLES])


def _micro_transition(params, intervention, state, attrs, rng):
    p_HS, p_SD = params["p_HS"], params["p_SD"]
    probs = np.zeros((len(state), 3))
    probs[state == 0] = [1.0 - p_HS - BACKGROUND_MORTALITY, p_HS, BACKGROUND_MORTALITY]
    probs[state == 1] = [0.0, 1.0 - p_SD, p_SD]
    probs[state == 2] = [0.0, 0.0, 1.0]
    return probs


def _micro_rewards(params, intervention, state, attrs):
    zero = np.zeros(len(state))
    return zero, zero


def _microsim_prevalence(param_rows, population):
    engine = MicrosimModel.discrete(
        states=STATES, transition_probabilities=_micro_transition, state_rewards=_micro_rewards,
        population=population, interventions=[INTERVENTION], n_cycles=N_CYCLES,
        cycle_correction="none", initial_state="healthy",
    )
    draws = pd.DataFrame(param_rows)
    draws.index = pd.RangeIndex(len(draws), name="iteration")
    events = run_psa(engine, draws, seed=123, collect="events").events
    occupancy = state_occupancy(
        events, states=STATES, initial_state="healthy", n_individuals=population,
        times=[float(cycle) for cycle in TARGET_CYCLES],
    )
    result = np.zeros((len(draws), len(TARGET_CYCLES)))
    for iteration in range(len(draws)):
        for column, cycle in enumerate(TARGET_CYCLES):
            key = (INTERVENTION, iteration, float(cycle))
            result[iteration, column] = occupancy.loc[key, "sick"]
    return result


def _npe_posterior(simulator, observed, prior):
    torch.manual_seed(0)
    theta = prior.sample((2_500,))
    inference = NPE(prior=prior, show_progress_bars=False)
    inference.append_simulations(theta, simulator(theta)).train()
    samples = inference.build_posterior().sample(
        (1_000,), x=torch.tensor(observed, dtype=torch.float32), show_progress_bars=False
    )
    return pd.DataFrame(samples.numpy(), columns=list(CALIBRATED))


def test_microsim_calibrates_and_noise_widens_posterior():
    cohort = MarkovModel(
        states=STATES, interventions=(INTERVENTION,),
        transitions_and_rewards=lambda params, intervention: CohortSpec(
            np.array([
                [1.0 - params["p_HS"] - BACKGROUND_MORTALITY, params["p_HS"], BACKGROUND_MORTALITY],
                [0.0, 1.0 - params["p_SD"], params["p_SD"]],
                [0.0, 0.0, 1.0],
            ]),
            np.zeros(3), np.array([1.0, 0.8, 0.0]),
        ),
        n_cycles=N_CYCLES, cycle_correction="none",
    )
    low = np.array([BOUNDS[name][0] for name in CALIBRATED])
    high = np.array([BOUNDS[name][1] for name in CALIBRATED])
    true_prevalence = _cohort_prevalence(cohort, TRUTH)
    observed = np.random.default_rng(20260718).binomial(SURVEY_SIZE, true_prevalence) / SURVEY_SIZE

    unit_design = LatinHypercube(d=2, seed=7).random(30)
    design = pd.DataFrame(scale(unit_design, low, high), columns=list(CALIBRATED))
    rows = [row for row in design.to_dict("records") for _ in range(6)]
    unit_points = np.repeat(unit_design, 6, axis=0)
    targets = _microsim_prevalence(rows, POPULATION)

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2)) * RBF([0.3, 0.3], length_scale_bounds=(0.05, 2.0))
        + WhiteKernel(1e-3, (1e-6, 1e-1))
    )
    surrogates = [
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4)
        .fit(unit_points, targets[:, target])
        for target in range(len(TARGET_CYCLES))
    ]

    def to_unit(points):
        return (points - low) / (high - low)

    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32), high=torch.tensor(high, dtype=torch.float32)
    )
    survey_rng = np.random.default_rng(11)
    model_rng = np.random.default_rng(12)

    def survey_only(theta):
        mean = np.column_stack([gp.predict(to_unit(theta.numpy())) for gp in surrogates])
        return torch.tensor(survey_rng.binomial(SURVEY_SIZE, np.clip(mean, 0, 1)) / SURVEY_SIZE,
                            dtype=torch.float32)

    def model_and_survey(theta):
        columns = [gp.predict(to_unit(theta.numpy()), return_std=True) for gp in surrogates]
        mean = np.column_stack([col[0] for col in columns])
        spread = np.column_stack([col[1] for col in columns])
        draw = mean + model_rng.normal(0.0, 1.0, mean.shape) * spread
        return torch.tensor(model_rng.binomial(SURVEY_SIZE, np.clip(draw, 0, 1)) / SURVEY_SIZE,
                            dtype=torch.float32)

    survey_post = _npe_posterior(survey_only, observed, prior)
    model_post = _npe_posterior(model_and_survey, observed, prior)

    for name in CALIBRATED:
        assert abs(model_post[name].mean() - TRUTH[name]) < 0.03  # recovers truth

    # The model's replicate noise widens the posterior overall.
    survey_spread = float(survey_post.std().sum())
    model_spread = float(model_post.std().sum())
    assert model_spread > survey_spread  # replicate noise adds width
