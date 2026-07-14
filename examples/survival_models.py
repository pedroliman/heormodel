"""Turn a fitted parametric survival curve into engine inputs with `heormodel.survival`.

A health economist who has fit a survival curve to patient data needs two things
from a modeling package: a way to sample event times for an individual-level
model, and a way to build the per-cycle transition probabilities a cohort model
consumes. `heormodel.survival` does both from one `SurvivalCurve`, and carries the
uncertainty in the fitted parameters onto the iteration index so the survival
estimates flow through a probabilistic analysis unchanged.

The reference curve is Weibull in the accelerated-failure-time parameterization,
``S(t) = exp(-(t / scale) ** shape)`` with shape 1.2 and scale 6.0 years, run as a
two-state alive-and-dead model at a 3% annual discount rate and a utility of 0.85
while alive. The recovery exercise fits the curve to a right-censored sample with
a ``lifelines`` Weibull model and propagates the fitted uncertainty.

Run it with::

    uv run python examples/survival_models.py

Requires the survival extra (``uv pip install 'heormodel[survival]'``). Outputs
(written to ``examples/output/``):
    - survival_models_fit.png
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gamma

from heormodel import survival as sv
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

CurveOf = Callable[[pd.Series], sv.SurvivalCurve]


def discounted_life_expectancy(curve: sv.SurvivalCurve, utility: float = 1.0) -> float:
    """Analytic discounted (quality-adjusted) life expectancy of a curve."""
    value, _ = quad(lambda t: np.exp(-DISCOUNT * t) * curve.survival(t), 0, np.inf)
    return utility * value


def continuous_engine(
    curve_of: CurveOf, n_individuals: int, utility: float = 1.0, horizon: float = 80.0
) -> MicrosimModel:
    """Two-state model on the continuous clock: sample one death time per person."""

    def event_times(params, intervention, state, attrs, rng):
        times = np.full((len(state), 2), np.inf)  # columns: to alive, to dead
        alive = state == 0
        times[alive, 1] = curve_of(params).sample_time(rng, int(alive.sum()))
        return times

    def reward_rates(params, intervention, state, attrs):
        living = (state == 0).astype(float)
        return np.zeros(len(state)), living * utility

    return MicrosimModel.continuous(
        states=STATES, event_times=event_times, state_reward_rates=reward_rates,
        interventions=[INTERVENTION], horizon=horizon, n_individuals=n_individuals,
        discount_rate=DISCOUNT, effect="lifeyears",
    )


def cohort_engine(curve_of: CurveOf, utility: float = 1.0, n_cycles: int = 60) -> MarkovModel:
    """Two-state model on the cohort clock: per-cycle death probabilities."""

    def transitions_and_rewards(params, intervention):
        causes = {("alive", "dead"): curve_of(params)}
        transition = sv.to_transition_matrix(causes, STATES, n_cycles)
        return CohortSpec(transition, np.zeros(2), np.array([utility, 0.0]))

    return MarkovModel(
        states=STATES, interventions=[INTERVENTION],
        transitions_and_rewards=transitions_and_rewards, n_cycles=n_cycles,
        initial_state="alive", discount_rate=DISCOUNT, cycle_correction="half_cycle",
        effect="lifeyears",
    )


def _draws(**columns: float) -> pd.DataFrame:
    data = {name: [value] for name, value in columns.items()}
    return pd.DataFrame(data, index=pd.RangeIndex(1, name="iteration"))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    from lifelines import WeibullFitter

    reference = sv.weibull(SHAPE, SCALE)

    def weibull_of(params: pd.Series) -> sv.SurvivalCurve:
        return sv.weibull(params["shape"], params["scale"])

    # Reference table from the analytic curve.
    print("Reference model (Weibull shape 1.2, scale 6.0):")
    print(f"  Undiscounted life expectancy: {SCALE * gamma(1 + 1 / SHAPE):.5f}")
    print(f"  Discounted life expectancy:   {discounted_life_expectancy(reference):.5f}")
    print(f"  Discounted QALYs (u=0.85):    {discounted_life_expectancy(reference, UTILITY):.5f}")
    death = reference.cycle_transition_probabilities(5)
    print(f"  Death probabilities, cycles 0-4: {', '.join(f'{p:.5f}' for p in death)}")

    # The two engines recover the same discounted life expectancy.
    draws = _draws(shape=SHAPE, scale=SCALE)
    continuous = run_psa(continuous_engine(weibull_of, 200_000), draws, seed=1, sequential=True)
    continuous_qaly = run_psa(
        continuous_engine(weibull_of, 200_000, UTILITY), draws, seed=1, sequential=True
    )
    cohort = cohort_engine(weibull_of).evaluate(draws).summary()
    continuous_le = continuous.outcomes.summary().loc[INTERVENTION, "lifeyears"]
    continuous_qaly_value = continuous_qaly.outcomes.summary().loc[INTERVENTION, "lifeyears"]
    print("\nEngine recovery of the discounted life expectancy (analytic 4.92709):")
    print(f"  Continuous sampler, life-years: {continuous_le:.5f}")
    print(f"  Continuous sampler, QALYs:      {continuous_qaly_value:.5f}")
    print(f"  Cohort, life-years:             {cohort.loc[INTERVENTION, 'lifeyears']:.5f}")

    # Parameter recovery: generate a censored sample, fit with lifelines, propagate.
    rng = np.random.default_rng(20260714)
    event_time = reference.sample_time(rng, 300)
    observed = np.minimum(event_time, CENSOR)
    fit = WeibullFitter().fit(observed, (event_time <= CENSOR).astype(float))
    print(f"\nFit to 300 censored patients (lifelines Weibull): "
          f"lambda {fit.lambda_:.3f}, rho {fit.rho_:.3f}")

    def fitted_of(params: pd.Series) -> sv.SurvivalCurve:
        return sv.from_lifelines(fit, params)

    params = sv.sample_params(fit, 1_000, seed=1)
    probabilistic = run_psa(continuous_engine(fitted_of, 20_000), params, seed=2).outcomes
    per_iteration = probabilistic.effects_wide()[INTERVENTION]
    lower, upper = np.percentile(per_iteration, [2.5, 97.5])
    print(f"Probabilistic discounted life expectancy: {per_iteration.mean():.3f} "
          f"(95% credible interval {lower:.3f} to {upper:.3f})")

    # Survival curve with the fit overlaid.
    import matplotlib.pyplot as plt

    grid = np.linspace(0, CENSOR, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, reference.survival(grid), label="True curve")
    ax.plot(grid, sv.from_lifelines(fit).survival(grid), "--", label="Fitted (n=300)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "survival_models_fit.png", dpi=150, bbox_inches="tight")
    print(f"\nWrote the survival curve plot to {OUT}/")


if __name__ == "__main__":
    main()
