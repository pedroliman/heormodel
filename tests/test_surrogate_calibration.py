"""Both methods on the surrogate match a direct calibration (skipped without deps).

A reduced-size version of ``examples/surrogate_calibration.py``. It fits a Gaussian
process to a small design of the model's prevalence, then checks that ABC and neural
posterior estimation, both run against the surrogate, recover the same truth a direct ABC
run recovers against the model.
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyabc")
pytest.importorskip("sbi")

os.environ["ABC_LOG_LEVEL"] = "WARNING"
logging.getLogger("sbi").setLevel(logging.WARNING)

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


def test_both_methods_on_surrogate_match_direct():
    import warnings

    from scipy.stats.qmc import LatinHypercube, scale
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    engine = MarkovModel(
        states=STATES, interventions=(INTERVENTION,),
        transitions_and_rewards=_transitions_and_rewards,
        n_cycles=N_CYCLES, cycle_correction="none",
    )
    low = np.array([BOUNDS[name][0] for name in CALIBRATED])
    high = np.array([BOUNDS[name][1] for name in CALIBRATED])
    true_prevalence = _prevalence(engine, TRUTH)
    observed = np.random.default_rng(20260718).binomial(SURVEY_SIZE, true_prevalence) / SURVEY_SIZE
    epsilon = 0.5 * float(np.sqrt((true_prevalence * (1 - true_prevalence) / SURVEY_SIZE).sum()))

    def survey(prevalences, rng):
        return rng.binomial(SURVEY_SIZE, np.clip(prevalences, 0, 1)) / SURVEY_SIZE

    # Gaussian process on a small design reproduces held-out model runs.
    design = pd.DataFrame(
        scale(LatinHypercube(d=2, seed=7).random(60), low, high), columns=list(CALIBRATED)
    )
    design_targets = np.array([_prevalence(engine, row) for row in design.to_dict("records")])
    kernel = ConstantKernel(1.0) * RBF([0.1, 0.1]) + WhiteKernel(1e-6, (1e-10, 1e-2))
    surrogates = [
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
        .fit(design.to_numpy(), design_targets[:, target])
        for target in range(len(TARGET_LABELS))
    ]
    holdout = scale(LatinHypercube(d=2, seed=99).random(100), low, high)
    holdout_targets = np.array(
        [_prevalence(engine, dict(zip(CALIBRATED, row, strict=True))) for row in holdout]
    )
    predicted = np.column_stack([surrogate.predict(holdout) for surrogate in surrogates])
    rmse = np.sqrt(((predicted - holdout_targets) ** 2).mean(axis=0))
    assert rmse.max() < 0.002  # surrogate reproduces the model well within the survey scale

    priors = {name: Uniform(*BOUNDS[name]) for name in CALIBRATED}
    observed_map = dict(zip(TARGET_LABELS, observed, strict=True))

    # Direct ABC against the model.
    direct_rng = np.random.default_rng(2024)

    def direct_simulator(params):
        drawn = survey(_prevalence(engine, params), direct_rng)
        return dict(zip(TARGET_LABELS, drawn, strict=True))

    direct_mean = abc_calibrate(
        direct_simulator, priors=priors, observed=observed_map,
        population_size=150, max_populations=8, min_epsilon=epsilon, n_posterior=1_000, seed=1,
    ).posterior.mean()

    # ABC against the surrogate.
    abc_surrogate_rng = np.random.default_rng(2024)

    def surrogate_prevalence(params):
        point = np.array([[params[name] for name in CALIBRATED]])
        return np.array([gp.predict(point)[0] for gp in surrogates])

    def abc_surrogate_simulator(params):
        drawn = survey(surrogate_prevalence(params), abc_surrogate_rng)
        return dict(zip(TARGET_LABELS, drawn, strict=True))

    abc_surrogate_mean = abc_calibrate(
        abc_surrogate_simulator, priors=priors, observed=observed_map,
        population_size=150, max_populations=8, min_epsilon=epsilon, n_posterior=1_000, seed=1,
    ).posterior.mean()

    # Neural posterior estimation against the surrogate.
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    torch.manual_seed(0)
    sbi_rng = np.random.default_rng(11)
    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32), high=torch.tensor(high, dtype=torch.float32)
    )

    def sbi_surrogate_simulator(theta):
        mean = np.column_stack([gp.predict(theta.numpy()) for gp in surrogates])
        return torch.tensor(survey(mean, sbi_rng), dtype=torch.float32)

    theta = prior.sample((3_000,))
    inference = NPE(prior=prior, show_progress_bars=False)
    inference.append_simulations(theta, sbi_surrogate_simulator(theta)).train()
    sbi_samples = inference.build_posterior().sample(
        (1_000,), x=torch.tensor(observed, dtype=torch.float32), show_progress_bars=False
    ).numpy()
    sbi_surrogate_mean = pd.DataFrame(sbi_samples, columns=list(CALIBRATED)).mean()

    for name in CALIBRATED:
        assert abs(direct_mean[name] - TRUTH[name]) < 0.02  # direct recovers truth
        assert abs(abc_surrogate_mean[name] - direct_mean[name]) < 0.02  # surrogate ABC matches
        assert abs(sbi_surrogate_mean[name] - direct_mean[name]) < 0.02  # surrogate SBI matches
