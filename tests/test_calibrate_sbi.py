"""Neural posterior estimation recovers the truth on the model (skipped without sbi).

A reduced-size version of ``examples/calibrate_sbi.py``: it trains a density estimator on
model-and-survey pairs and checks the posterior mean recovers the known truth.
"""

import logging

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sbi")

logging.getLogger("sbi").setLevel(logging.WARNING)

import torch  # noqa: E402
from sbi.inference import NPE  # noqa: E402
from sbi.utils import BoxUniform  # noqa: E402

from heormodel.models import CohortSpec, MarkovModel  # noqa: E402

STATES = ("healthy", "sick", "dead")
INTERVENTION = "natural_history"
N_CYCLES = 40
TARGET_CYCLES = (8, 16, 28)
BACKGROUND_MORTALITY = 0.01
CALIBRATED = ("p_HS", "p_SD")
BOUNDS = {"p_HS": (0.02, 0.20), "p_SD": (0.05, 0.35)}
TRUTH = {"p_HS": 0.08, "p_SD": 0.15}
SURVEY_SIZE = 1_000


def _transitions_and_rewards(params, intervention):
    p_HS, p_SD = params["p_HS"], params["p_SD"]
    transition = np.array([
        [1.0 - p_HS - BACKGROUND_MORTALITY, p_HS, BACKGROUND_MORTALITY],
        [0.0, 1.0 - p_SD, p_SD],
        [0.0, 0.0, 1.0],
    ])
    return CohortSpec(transition, np.zeros(3), np.array([1.0, 0.8, 0.0]))


def _prevalence(engine, params):
    occupancy = engine.trace(pd.Series(params), INTERVENTION)["sick"].to_numpy()
    return np.array([occupancy[cycle] for cycle in TARGET_CYCLES])


def test_sbi_recovers_truth_on_model():
    engine = MarkovModel(
        states=STATES, interventions=(INTERVENTION,),
        transitions_and_rewards=_transitions_and_rewards,
        n_cycles=N_CYCLES, cycle_correction="none",
    )
    low = np.array([BOUNDS[name][0] for name in CALIBRATED])
    high = np.array([BOUNDS[name][1] for name in CALIBRATED])
    true_prevalence = _prevalence(engine, TRUTH)
    observed = np.random.default_rng(20260718).binomial(SURVEY_SIZE, true_prevalence) / SURVEY_SIZE

    torch.manual_seed(0)
    survey_rng = np.random.default_rng(11)
    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32), high=torch.tensor(high, dtype=torch.float32)
    )

    def simulator(theta):
        means = np.array(
            [_prevalence(engine, dict(zip(CALIBRATED, row, strict=True))) for row in theta.numpy()]
        )
        survey = survey_rng.binomial(SURVEY_SIZE, means) / SURVEY_SIZE
        return torch.tensor(survey, dtype=torch.float32)

    theta = prior.sample((3_000,))
    inference = NPE(prior=prior, show_progress_bars=False)
    inference.append_simulations(theta, simulator(theta)).train()
    samples = inference.build_posterior().sample(
        (1_000,), x=torch.tensor(observed, dtype=torch.float32), show_progress_bars=False
    ).numpy()
    mean = pd.DataFrame(samples, columns=list(CALIBRATED)).mean()
    for name in CALIBRATED:
        assert abs(mean[name] - TRUTH[name]) < 0.02  # posterior recovers truth
