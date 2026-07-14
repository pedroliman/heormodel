"""Survival curves as engine inputs (`heormodel.survival`).

Turn a fitted parametric survival model into the inputs the engines already
consume: sampled event times for `heormodel.models.MicrosimModel.continuous` and
per-cycle transition arrays for `heormodel.models.MarkovModel`. Fitted-parameter
uncertainty rides the canonical ``iteration`` index, so survival estimates flow
through `heormodel.run.run_psa` into cost-effectiveness and value-of-information
analysis with no special case.

`SurvivalCurve` is one composed value object holding a cumulative-hazard function
of time; the families (`exponential`, `weibull`, `gompertz`), the curve algebra
(`apply_hazard_ratio`, `apply_acceleration_factor`, `mix`, `splice`), and the
adapter (`from_lifelines`) are functions that return curves, so nothing depends on
a class hierarchy. `from_lifelines` and `sample_params` read a fit from a survival
package installed with the ``survival`` extra.
"""

from heormodel.survival.algebra import (
    apply_acceleration_factor,
    apply_hazard_ratio,
    mix,
    splice,
)
from heormodel.survival.curve import SurvivalCurve, exponential, gompertz, weibull
from heormodel.survival.fitting import from_lifelines, sample_params
from heormodel.survival.transitions import to_transition_matrix

__all__ = [
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
