"""Turning fitted survival curves into engine inputs (`heormodel.survival`).

A health economist who has fit a survival curve to patient data needs two
things from a modeling package: a way to build the per-cycle transition
probabilities a cohort model consumes, and a way to sample event times for an
individual-level model. `SurvivalCurve` is the one value object this module
builds and consumes throughout: a cumulative hazard function plus how to
sample from it.

- `exponential`, `weibull`, and `gompertz` build a `SurvivalCurve` from
  distribution parameters.
- `from_lifelines` builds one from a fitted parametric survival model, and
  `sample_params` draws parameter sets from that fit's asymptotic covariance
  onto the canonical `iteration` index, so the fitted curve's uncertainty
  flows through `heormodel.run.run_psa` like any other parameter. Neither
  function imports the fitting package; both duck-type against a fitted
  model's public interface, so `heormodel.survival` imports cleanly whether
  or not that optional dependency (`heormodel[survival]`) is installed.
- `apply_hazard_ratio`, `apply_acceleration_factor`, `mix`, and `splice` build
  a new `SurvivalCurve` from one or more existing curves: a treatment arm
  under a sampled hazard ratio, a population mixture, or an observed curve
  extended by a parametric tail.
- `to_transition_matrix` converts one curve, or several cause-specific curves
  for competing risks, into the per-cycle transition array
  `heormodel.models.MarkovModel` consumes. `SurvivalCurve.sample_time` is the
  matching entry point for `heormodel.models.MicrosimModel.continuous`.
"""

from heormodel.survival.algebra import apply_acceleration_factor, apply_hazard_ratio, mix, splice
from heormodel.survival.curve import SurvivalCurve, exponential, gompertz, weibull
from heormodel.survival.lifelines_adapter import (
    FittedSurvivalModel,
    from_lifelines,
    sample_params,
)
from heormodel.survival.transitions import to_transition_matrix

__all__ = [
    "FittedSurvivalModel",
    "SurvivalCurve",
    "apply_acceleration_factor",
    "apply_hazard_ratio",
    "exponential",
    "from_lifelines",
    "gompertz",
    "mix",
    "sample_params",
    "splice",
    "to_transition_matrix",
    "weibull",
]
