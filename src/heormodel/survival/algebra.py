"""Curve algebra: build a new survival curve from existing ones.

Each function returns a fresh `SurvivalCurve`, so operations compose. A treatment
arm is the comparator curve under a sampled hazard ratio; a long-term model is an
observed curve spliced onto a background-mortality tail; a heterogeneous
population is a mixture of subgroup curves. The proportional-hazards and
acceleration-factor transforms carry the analytic hazard and quantile through
when the base curve has them, so sampling stays closed-form; mixtures and splices
have no closed-form quantile and fall back to bisection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from heormodel.survival.curve import _TINY, Array, SurvivalCurve, TimeFunction


def _scale_output(function: TimeFunction, factor: float) -> TimeFunction:
    """Return ``x -> factor * function(x)``."""

    def scaled(x: Array) -> Array:
        return factor * function(x)

    return scaled


def _scale_input(function: TimeFunction, factor: float) -> TimeFunction:
    """Return ``x -> function(factor * x)``."""

    def scaled(x: Array) -> Array:
        return function(factor * np.asarray(x, dtype=float))

    return scaled


def apply_hazard_ratio(curve: SurvivalCurve, ratio: float) -> SurvivalCurve:
    """Multiply the hazard by a constant, the proportional-hazards transform.

    The cumulative hazard scales by ``ratio``, so ``S'(t) = S(t) ** ratio``. For a
    constant hazard ``h`` this turns discounted life-years ``1 / (d + h)`` into
    ``1 / (d + ratio h)``.

    Example:
        >>> from heormodel.survival import exponential, apply_hazard_ratio
        >>> treated = apply_hazard_ratio(exponential(0.2), 0.5)
        >>> round(float(treated.hazard_at(3.0)), 3)
        0.1
    """
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError(f"hazard ratio must be positive, got {ratio}.")
    hazard = None if curve.hazard is None else _scale_output(curve.hazard, ratio)
    quantile = None if curve.quantile is None else _scale_input(curve.quantile, 1.0 / ratio)
    return SurvivalCurve(
        cumulative_hazard=_scale_output(curve.cumulative_hazard, ratio),
        hazard=hazard,
        quantile=quantile,
        name=f"{curve.name} x HR {ratio:g}",
    )


def apply_acceleration_factor(curve: SurvivalCurve, factor: float) -> SurvivalCurve:
    """Stretch or compress the time axis, the accelerated-failure-time transform.

    ``S'(t) = S(t / factor)``, so ``factor > 1`` slows failure and every event
    time scales by ``factor``.

    Example:
        >>> from heormodel.survival import weibull, apply_acceleration_factor
        >>> slower = apply_acceleration_factor(weibull(1.2, 6.0), 2.0)
        >>> round(float(slower.survival(12.0)) - float(weibull(1.2, 6.0).survival(6.0)), 9)
        0.0
    """
    factor = float(factor)
    if factor <= 0:
        raise ValueError(f"acceleration factor must be positive, got {factor}.")
    # h'(t) = h(t / factor) / factor; the quantile scales by the factor.
    hazard = None
    if curve.hazard is not None:
        hazard = _scale_output(_scale_input(curve.hazard, 1.0 / factor), 1.0 / factor)
    quantile = None if curve.quantile is None else _scale_output(curve.quantile, factor)
    return SurvivalCurve(
        cumulative_hazard=_scale_input(curve.cumulative_hazard, 1.0 / factor),
        hazard=hazard,
        quantile=quantile,
        name=f"{curve.name} x AF {factor:g}",
    )


def mix(curves: Sequence[SurvivalCurve], weights: Sequence[float]) -> SurvivalCurve:
    """Mixture curve ``S'(t) = sum(weight_i S_i(t))``.

    The weights are normalized to sum to one. A mixture models a population of
    subgroups with different survival, whose overall survival is the
    weight-averaged survival, not the average hazard.

    Example:
        >>> from heormodel.survival import exponential, mix
        >>> blend = mix([exponential(0.1), exponential(0.3)], [0.5, 0.5])
        >>> round(float(blend.survival(0.0)), 6)
        1.0
    """
    curves = list(curves)
    weight = np.asarray(weights, dtype=float)
    if len(curves) != weight.size:
        raise ValueError("curves and weights must have the same length.")
    if weight.size == 0:
        raise ValueError("Provide at least one curve.")
    if np.any(weight < 0):
        raise ValueError("weights must be non-negative.")
    weight = weight / weight.sum()

    def survival(t: Array) -> Array:
        time = np.asarray(t, dtype=float)
        return sum(w * curve.survival(time) for w, curve in zip(weight, curves, strict=True))

    def cumulative_hazard(t: Array) -> Array:
        return -np.log(np.clip(survival(t), _TINY, 1.0))

    def hazard(t: Array) -> Array:
        time = np.asarray(t, dtype=float)
        pairs = zip(weight, curves, strict=True)
        numerator = sum(w * curve.survival(time) * curve.hazard_at(time) for w, curve in pairs)
        denominator = survival(time)
        return numerator / np.where(denominator > 0, denominator, 1.0)

    return SurvivalCurve(cumulative_hazard=cumulative_hazard, hazard=hazard, name="mixture")


def splice(early: SurvivalCurve, late: SurvivalCurve, cutpoint: float) -> SurvivalCurve:
    """Follow ``early`` to ``cutpoint``, then continue on ``late``'s conditional survival.

    This is the extrapolation operation: the observed curve up to the last
    reliable follow-up time, then a parametric tail (often background mortality).
    Survival is continuous at the cutpoint by construction.

    Example:
        >>> from heormodel.survival import weibull, exponential, splice
        >>> extended = splice(weibull(1.2, 6.0), exponential(0.05), cutpoint=5.0)
        >>> round(float(extended.survival(5.0)) - float(weibull(1.2, 6.0).survival(5.0)), 9)
        0.0
    """
    cutpoint = float(cutpoint)
    survival_early_cut = float(early.survival(np.array([cutpoint]))[0])
    survival_late_cut = float(late.survival(np.array([cutpoint]))[0])
    if survival_late_cut <= 0:
        raise ValueError("late curve has zero survival at the cutpoint; cannot splice.")

    def survival(t: Array) -> Array:
        time = np.asarray(t, dtype=float)
        tail = survival_early_cut * late.survival(time) / survival_late_cut
        return np.where(time <= cutpoint, early.survival(time), tail)

    def cumulative_hazard(t: Array) -> Array:
        return -np.log(np.clip(survival(t), _TINY, 1.0))

    def hazard(t: Array) -> Array:
        time = np.asarray(t, dtype=float)
        return np.where(time <= cutpoint, early.hazard_at(time), late.hazard_at(time))

    return SurvivalCurve(
        cumulative_hazard=cumulative_hazard,
        hazard=hazard,
        name=f"{early.name} spliced at {cutpoint:g} onto {late.name}",
    )
