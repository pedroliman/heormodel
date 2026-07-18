"""ABC recovers the truth from a survey target (skipped without pyabc).

A reduced-size version of ``examples/calibrate_abc.py`` on the same three-state Markov
model: it draws one survey from a known truth and checks the ABC posterior mean recovers
that truth.
"""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyabc")

os.environ["ABC_LOG_LEVEL"] = "WARNING"

from heormodel.calibrate import abc_calibrate  # noqa: E402
from heormodel.models import CohortSpec, MarkovModel  # noqa: E402
from heormodel.params import Uniform  # noqa: E402

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


def test_abc_recovers_truth_from_survey():
    engine = MarkovModel(
        states=STATES, interventions=(INTERVENTION,),
        transitions_and_rewards=_transitions_and_rewards,
        n_cycles=N_CYCLES, cycle_correction="none",
    )
    true_prevalence = _prevalence(engine, TRUTH)
    observed = np.random.default_rng(20260718).binomial(SURVEY_SIZE, true_prevalence) / SURVEY_SIZE
    epsilon = 0.5 * float(np.sqrt((true_prevalence * (1 - true_prevalence) / SURVEY_SIZE).sum()))

    sim_rng = np.random.default_rng(2024)

    def simulator(params):
        survey = sim_rng.binomial(SURVEY_SIZE, _prevalence(engine, params)) / SURVEY_SIZE
        return dict(zip(TARGET_LABELS, survey, strict=True))

    result = abc_calibrate(
        simulator,
        priors={name: Uniform(*BOUNDS[name]) for name in CALIBRATED},
        observed=dict(zip(TARGET_LABELS, observed, strict=True)),
        population_size=150,
        max_populations=8,
        min_epsilon=epsilon,
        n_posterior=1_000,
        seed=1,
    )
    mean = result.posterior.mean()
    for name in CALIBRATED:
        assert abs(mean[name] - TRUTH[name]) < 0.02  # posterior recovers truth
