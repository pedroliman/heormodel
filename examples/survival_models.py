"""Turn a fitted survival curve into engine inputs through `heormodel.survival`.

Phase 1 (`examples/survival_bridge.py`) reproduced the reference model below
with bespoke, example-local functions. This script reproduces the same
numbers through the promoted public module: `heormodel.survival.weibull`
builds the curve, `heormodel.survival.to_transition_matrix` feeds
`MarkovModel`, `SurvivalCurve.sample_time` feeds `MicrosimModel.continuous`,
and `heormodel.survival.from_lifelines`/`sample_params` replace the bespoke
maximum-likelihood fit with a fit from `lifelines`, an optional dependency
(``heormodel[survival]``).

The survival curve is Weibull in the accelerated-failure-time
parameterization, ``S(t) = exp(-(t / scale) ** shape)`` with shape 1.2 and
scale 6.0 years. A two-state alive-and-dead model runs it at a 3% annual
discount rate and a utility of 0.85 while alive.

Run it with::

    uv run python examples/survival_models.py

Outputs (written to ``examples/output/``):
    - survival_models_fit.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import WeibullFitter
from numpy.typing import NDArray
from scipy.integrate import quad

from heormodel.models import CohortSpec, MarkovModel, MicrosimModel
from heormodel.run import run_psa
from heormodel.survival import (
    SurvivalCurve,
    from_lifelines,
    sample_params,
    to_transition_matrix,
    weibull,
)

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("alive", "dead")
INTERVENTION = "Standard care"
SHAPE, SCALE = 1.2, 6.0  # Weibull shape and scale (years), the reference curve
DISCOUNT = 0.03
UTILITY = 0.85  # utility while alive
CENSOR = 12.0  # administrative censoring horizon in the recovery exercise (years)


def discounted_life_expectancy(
    curve: SurvivalCurve, rate: float = DISCOUNT, utility: float = 1.0
) -> float:
    """Discounted (quality-adjusted) life expectancy, the integral of the curve."""
    value, _ = quad(lambda t: np.exp(-rate * t) * curve.survival(t), 0, np.inf)
    return utility * value


def cohort_life_years(
    curve: SurvivalCurve, n_cycles: int, rate: float = DISCOUNT, utility: float = 1.0
) -> float:
    """Discrete annual cohort discounted life-years under the trapezoidal correction.

    The cohort occupancy trace is the curve's survival function on the cycle
    grid. Continuous discounting isolates the trapezoidal (half-cycle)
    correction as the only difference from `discounted_life_expectancy`'s
    continuous integral.
    """
    times = np.arange(n_cycles + 1, dtype=float)
    occupancy = curve.survival(times)
    weights = np.ones(n_cycles + 1)
    weights[0] = weights[-1] = 0.5  # trapezoidal correction
    return utility * float((occupancy * np.exp(-rate * times)) @ weights)


# --- The two-state model on each engine, built from a `SurvivalCurve` per draw ---


def continuous_engine(n_individuals: int, horizon: float = 80.0) -> MicrosimModel:
    """Two-state model on the continuous clock: race a single sampled death time."""

    def event_times(params, intervention, state, attrs, rng):
        curve = weibull(params["shape"], params["scale"])
        times: NDArray[np.float64] = np.full((len(state), 2), np.inf)  # to alive, to dead
        alive = state == 0
        times[alive, 1] = curve.sample_time(rng, int(alive.sum()))
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
    """Two-state model on the cohort clock: per-cycle transitions from the curve."""

    def transitions_and_rewards(params, intervention):
        curve = weibull(params["shape"], params["scale"])
        transition = to_transition_matrix(curve, n_cycles=n_cycles)
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

    # Reference table: every row from the curve `heormodel.survival.weibull` builds.
    curve = weibull(SHAPE, SCALE)
    discounted_qaly = discounted_life_expectancy(curve, utility=UTILITY)
    death = to_transition_matrix(curve, n_cycles=5)[:, 0, 1]
    print("Reference model (Weibull shape 1.2, scale 6.0):")
    print(f"  Undiscounted life expectancy: {discounted_life_expectancy(curve, rate=0.0):.5f}")
    print(f"  Discounted life expectancy:   {discounted_life_expectancy(curve):.5f}")
    print(f"  Discounted QALYs (u=0.85):    {discounted_qaly:.5f}")
    print(f"  Death probabilities, cycles 0-4: {', '.join(f'{p:.5f}' for p in death)}")
    print(f"  Cohort discounted life-years: {cohort_life_years(curve, 60):.5f}")

    # The two engines recover the same discounted life expectancy from the same curve.
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

    # Parameter recovery: generate a censored sample, fit with `lifelines`, propagate.
    rng = np.random.default_rng(20260714)
    event_time = curve.sample_time(rng, 300)
    observed = np.minimum(event_time, CENSOR)
    observed_event = (event_time <= CENSOR).astype(float)
    fit = WeibullFitter().fit(observed, event_observed=observed_event)
    fitted_curve = from_lifelines(fit)
    # lifelines' WeibullFitter names its parameters lambda_ (scale) and rho_ (shape).
    shape_hat, scale_hat = fit.params_["rho_"], fit.params_["lambda_"]
    se = np.sqrt(np.diag(fit.variance_matrix_))
    print(f"\nFit to 300 censored patients: shape {shape_hat:.3f}, scale {scale_hat:.3f} "
          f"(SE {se[1]:.3f}, {se[0]:.3f})")
    fitted_dle = discounted_life_expectancy(fitted_curve)
    print(f"Fitted-model discounted life expectancy: {fitted_dle:.3f}")

    params = sample_params(fit, n=1_000, seed=2)
    params = params.rename(columns={"lambda_": "scale", "rho_": "shape"})
    params["utility"] = 1.0
    probabilistic = run_psa(continuous_engine(20_000), params, seed=2).outcomes
    per_iteration = probabilistic.effects_wide()[INTERVENTION]
    lower, upper = np.percentile(per_iteration, [2.5, 97.5])
    print(f"Probabilistic discounted life expectancy: {per_iteration.mean():.3f} "
          f"(95% credible interval {lower:.3f} to {upper:.3f})")

    # Survival curve with the fit overlaid.
    import matplotlib.pyplot as plt

    grid = np.linspace(0, CENSOR, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, curve.survival(grid), label="True curve")
    ax.plot(grid, fitted_curve.survival(grid), "--", label="Fitted (n=300)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "survival_models_fit.png", dpi=150, bbox_inches="tight")
    print(f"\nWrote the survival curve plot to {OUT}/")


if __name__ == "__main__":
    main()
