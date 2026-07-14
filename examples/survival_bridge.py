"""Turn a fitted parametric survival curve into engine inputs.

A health economist who has fit a survival curve to patient data needs two things
from a modeling package: a way to sample event times for an individual-level
model, and a way to build the per-cycle death probabilities a cohort model
consumes. This example does both from one Weibull overall-survival curve, shows
the two engines agree, and then adds the estimation step a real analysis goes
through, fitting the curve to a censored sample and carrying the fitted
uncertainty into a probabilistic analysis.

The survival curve is Weibull in the accelerated-failure-time parameterization,
``S(t) = exp(-(t / scale) ** shape)`` with shape 1.2 and scale 6.0 years. A
two-state alive-and-dead model runs it at a 3% annual discount rate and a utility
of 0.85 while alive. The helpers here (survival, hazard, event-time sampling,
per-cycle transition probabilities, Weibull maximum-likelihood fitting, and
parameter sampling from the fit) are bespoke and example-local, the way the
``examples/mdm_*`` replications keep their logic local. Items 19 and 20 reuse
them before any of it becomes a public module.

Run it with::

    uv run python examples/survival_bridge.py

Outputs (written to ``examples/output/``):
    - survival_bridge_fit.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma

from heormodel.models import CohortSpec, MarkovModel, MicrosimModel
from heormodel.run import run_psa

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("alive", "dead")
INTERVENTION = "Standard care"
SHAPE, SCALE = 1.2, 6.0  # Weibull shape and scale (years), the reference curve
DISCOUNT = 0.03
UTILITY = 0.85  # utility while alive
CENSOR = 12.0  # administrative censoring horizon in the recovery exercise (years)

# --- Bespoke survival helpers (the phase-1 deliverable items 19 and 20 reuse) ---


def weibull_survival(t: NDArray[np.float64], shape: float, scale: float) -> NDArray[np.float64]:
    """Weibull survival function ``S(t) = exp(-(t / scale) ** shape)``."""
    return np.exp(-((np.asarray(t, dtype=float) / scale) ** shape))


def weibull_hazard(t: NDArray[np.float64], shape: float, scale: float) -> NDArray[np.float64]:
    """Weibull hazard ``h(t) = (shape / scale) (t / scale) ** (shape - 1)``."""
    t = np.asarray(t, dtype=float)
    return (shape / scale) * (t / scale) ** (shape - 1.0)


def sample_event_times(
    rng: np.random.Generator, size: int, shape: float, scale: float
) -> NDArray[np.float64]:
    """Sample event times by inverse-transform sampling of the survival curve.

    Solving ``S(t) = u`` for a uniform draw ``u`` gives
    ``t = scale * (-log(u)) ** (1 / shape)``.
    """
    u = rng.random(size)
    return scale * (-np.log(u)) ** (1.0 / shape)


def annual_death_probabilities(shape: float, scale: float, n_cycles: int) -> NDArray[np.float64]:
    """Per-cycle transition-to-death probabilities ``1 - S(k+1) / S(k)``."""
    surv = weibull_survival(np.arange(n_cycles + 1), shape, scale)
    return 1.0 - surv[1:] / surv[:-1]


def apply_hazard_ratio(shape: float, scale: float, hazard_ratio: float) -> tuple[float, float]:
    """Proportional-hazards transform: multiply the hazard by a constant.

    For a Weibull curve the shape is unchanged and the scale becomes
    ``scale * hazard_ratio ** (-1 / shape)``, so a treatment arm is the comparator
    curve under a sampled hazard ratio.
    """
    return shape, scale * hazard_ratio ** (-1.0 / shape)


def discounted_life_expectancy(
    shape: float, scale: float, rate: float = DISCOUNT, utility: float = 1.0
) -> float:
    """Discounted (quality-adjusted) life expectancy, the integral of the curve.

    Integrates ``utility * exp(-rate * t) * S(t)`` directly, the analytic value the
    continuous sampler and the cohort both target.
    """
    value, _ = quad(lambda t: np.exp(-rate * t) * weibull_survival(t, shape, scale), 0, np.inf)
    return utility * value


def cohort_life_years(
    shape: float, scale: float, n_cycles: int, rate: float = DISCOUNT, utility: float = 1.0
) -> float:
    """Discrete annual cohort discounted life-years under the trapezoidal correction.

    The cohort occupancy trace is the survival curve on the cycle grid. Continuous
    discounting isolates the trapezoidal (half-cycle) correction as the only
    difference from the continuous integral.
    """
    times = np.arange(n_cycles + 1, dtype=float)
    occupancy = weibull_survival(times, shape, scale)
    weights = np.ones(n_cycles + 1)
    weights[0] = weights[-1] = 0.5  # trapezoidal correction
    return utility * float((occupancy * np.exp(-rate * times)) @ weights)


# --- Weibull fitting and its uncertainty (the estimation-to-model step) ---


def fit_weibull(
    times: NDArray[np.float64], events: NDArray[np.float64]
) -> tuple[float, float, NDArray[np.float64]]:
    """Fit a Weibull model to a right-censored sample by maximum likelihood.

    Optimizes the log-shape and log-scale so the asymptotic covariance is on the
    log scale, the scale on which the parameters are unconstrained and roughly
    normal. Returns the estimated shape, scale, and the two-by-two log-scale
    covariance.
    """

    def negative_log_likelihood(theta: NDArray[np.float64]) -> float:
        shape, scale = np.exp(theta)
        cumulative_hazard = (times / scale) ** shape
        log_hazard = np.log(shape / scale) + (shape - 1.0) * np.log(times / scale)
        return -float((events * log_hazard - cumulative_hazard).sum())

    start = np.log([1.0, float(np.mean(times))])
    result = minimize(negative_log_likelihood, start, method="BFGS")
    shape, scale = np.exp(result.x)
    return float(shape), float(scale), np.asarray(result.hess_inv, dtype=float)


def sample_survival_params(
    shape: float,
    scale: float,
    cov_log: NDArray[np.float64],
    n: int,
    rng: np.random.Generator,
    utility: float = 1.0,
) -> pd.DataFrame:
    """Draw parameter sets from the fit's asymptotic distribution onto the iteration index.

    Sampling on the log scale keeps the shape and scale positive. The draws land on
    the canonical ``iteration`` index, so survival uncertainty shares one index with
    every other parameter.
    """
    mean_log = np.log([shape, scale])
    draws = rng.multivariate_normal(mean_log, cov_log, size=n)
    shapes, scales = np.exp(draws).T
    index = pd.RangeIndex(1, n + 1, name="iteration")
    return pd.DataFrame({"shape": shapes, "scale": scales, "utility": utility}, index=index)


# --- The two-state model on each engine ---


def continuous_engine(n_individuals: int, horizon: float = 80.0) -> MicrosimModel:
    """Two-state model on the continuous clock: race a single death time per person."""

    def event_times(params, intervention, state, attrs, rng):
        times = np.full((len(state), 2), np.inf)  # columns: to alive, to dead
        alive = state == 0
        times[alive, 1] = sample_event_times(
            rng, int(alive.sum()), params["shape"], params["scale"]
        )
        return times

    def reward_rates(params, intervention, state, attrs):
        alive = (state == 0).astype(float)
        return np.zeros(len(state)), alive * params["utility"]

    return MicrosimModel.continuous(
        states=STATES, event_times=event_times, state_reward_rates=reward_rates,
        interventions=[INTERVENTION], horizon=horizon, n_individuals=n_individuals,
        discount_rate=DISCOUNT, effect="lifeyears",
    )


def cohort_engine(n_cycles: int = 60) -> MarkovModel:
    """Two-state model on the cohort clock: per-cycle death probabilities."""

    def transitions_and_rewards(params, intervention):
        death = annual_death_probabilities(params["shape"], params["scale"], n_cycles)
        transition = np.zeros((n_cycles, 2, 2))
        transition[:, 0, 0], transition[:, 0, 1] = 1.0 - death, death
        transition[:, 1, 1] = 1.0  # dead is absorbing
        return CohortSpec(transition, np.zeros(2), np.array([params["utility"], 0.0]))

    return MarkovModel(
        states=STATES, interventions=[INTERVENTION],
        transitions_and_rewards=transitions_and_rewards, n_cycles=n_cycles,
        initial_state="alive", discount_rate=DISCOUNT, cycle_correction="half_cycle",
        effect="lifeyears",
    )


def _draws(shape: float, scale: float, utility: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"shape": [shape], "scale": [scale], "utility": [utility]},
        index=pd.RangeIndex(1, name="iteration"),
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Reference table: every row from the analytic curve.
    discounted_qaly = discounted_life_expectancy(SHAPE, SCALE, utility=UTILITY)
    death = annual_death_probabilities(SHAPE, SCALE, 5)
    print("Reference model (Weibull shape 1.2, scale 6.0):")
    print(f"  Undiscounted life expectancy: {SCALE * gamma(1 + 1 / SHAPE):.5f}")
    print(f"  Discounted life expectancy:   {discounted_life_expectancy(SHAPE, SCALE):.5f}")
    print(f"  Discounted QALYs (u=0.85):    {discounted_qaly:.5f}")
    print(f"  Death probabilities, cycles 0-4: {', '.join(f'{p:.5f}' for p in death)}")
    print(f"  Cohort discounted life-years: {cohort_life_years(SHAPE, SCALE, 60):.5f}")

    # The two engines recover the same discounted life expectancy.
    life = _draws(SHAPE, SCALE, 1.0)
    qaly = _draws(SHAPE, SCALE, UTILITY)
    continuous = run_psa(continuous_engine(200_000), life, seed=1, sequential=True).outcomes
    continuous_qaly = run_psa(continuous_engine(200_000), qaly, seed=1, sequential=True).outcomes
    cohort = cohort_engine().evaluate(life).summary()
    continuous_le = continuous.summary().loc[INTERVENTION, "lifeyears"]
    continuous_qaly_value = continuous_qaly.summary().loc[INTERVENTION, "lifeyears"]
    print("\nEngine recovery of the discounted life expectancy (analytic 4.92709):")
    print(f"  Continuous sampler, life-years: {continuous_le:.5f}")
    print(f"  Continuous sampler, QALYs:      {continuous_qaly_value:.5f}")
    print(f"  Cohort, life-years:             {cohort.loc[INTERVENTION, 'lifeyears']:.5f}")

    # Parameter recovery: generate a censored sample, fit, propagate the uncertainty.
    rng = np.random.default_rng(20260714)
    event_time = sample_event_times(rng, 300, SHAPE, SCALE)
    observed = np.minimum(event_time, CENSOR)
    observed_event = (event_time <= CENSOR).astype(float)
    shape_hat, scale_hat, cov_log = fit_weibull(observed, observed_event)
    se_log = np.sqrt(np.diag(cov_log))
    print(f"\nFit to 300 censored patients: shape {shape_hat:.3f}, scale {scale_hat:.3f} "
          f"(log-scale SE {se_log[0]:.3f}, {se_log[1]:.3f})")

    params = sample_survival_params(shape_hat, scale_hat, cov_log, 1_000, rng)
    probabilistic = run_psa(continuous_engine(20_000), params, seed=2).outcomes
    per_iteration = probabilistic.effects_wide()[INTERVENTION]
    lower, upper = np.percentile(per_iteration, [2.5, 97.5])
    print(f"Probabilistic discounted life expectancy: {per_iteration.mean():.3f} "
          f"(95% credible interval {lower:.3f} to {upper:.3f})")

    # Survival curve with the fit overlaid.
    import matplotlib.pyplot as plt

    grid = np.linspace(0, CENSOR, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, weibull_survival(grid, SHAPE, SCALE), label="True curve")
    ax.plot(grid, weibull_survival(grid, shape_hat, scale_hat), "--", label="Fitted (n=300)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "survival_bridge_fit.png", dpi=150, bbox_inches="tight")
    print(f"\nWrote the survival curve plot to {OUT}/")


if __name__ == "__main__":
    main()
