"""Curve algebra: build new survival curves from existing ones.

Each function takes one or more `heormodel.survival.SurvivalCurve` values and
returns a new one; none of them touch an engine or a parameter draw. A
treatment arm is the comparator curve under a sampled hazard ratio
(`apply_hazard_ratio`) or acceleration factor (`apply_acceleration_factor`); a
subgroup-weighted population curve is a mixture (`mix`); a long-term model is
an observed curve spliced onto a parametric tail (`splice`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from heormodel.survival.curve import SurvivalCurve, numeric_inverse_cumulative_hazard

_FloatArray = NDArray[np.float64]


def apply_hazard_ratio(curve: SurvivalCurve, hazard_ratio: float) -> SurvivalCurve:
    """Proportional-hazards transform: multiply the hazard by a constant.

    Scales the cumulative hazard directly, `cumulative_hazard(t) * hazard_ratio`, so a
    hazard ratio below 1 lowers the hazard and extends survival. Applied to
    `exponential`, this reproduces the closed form exactly: the discounted
    life expectancy under a hazard ratio `r` on a constant hazard `h` is
    `1 / (discount_rate + r * h)`.

    Args:
        curve: The comparator curve.
        hazard_ratio: Multiplicative factor on the hazard. Must be positive.

    Returns:
        A new `SurvivalCurve` with the scaled hazard.

    Example:
        >>> from heormodel.survival import apply_hazard_ratio, exponential
        >>> treated = apply_hazard_ratio(exponential(rate=0.2), hazard_ratio=0.5)
        >>> round(float(treated.survival(5.0)), 5)
        0.60653
    """
    if hazard_ratio <= 0:
        raise ValueError("hazard_ratio must be positive.")

    def cumulative_hazard(time: _FloatArray) -> _FloatArray:
        return hazard_ratio * curve.cumulative_hazard(np.asarray(time, dtype=np.float64))

    def inverse_cumulative_hazard(value: _FloatArray) -> _FloatArray:
        return curve.inverse_cumulative_hazard(np.asarray(value, dtype=np.float64) / hazard_ratio)

    return SurvivalCurve(
        cumulative_hazard,
        inverse_cumulative_hazard,
        f"{curve.label} under hazard ratio {hazard_ratio:g}",
    )


def apply_acceleration_factor(curve: SurvivalCurve, acceleration_factor: float) -> SurvivalCurve:
    """Accelerated-failure-time transform: rescale the time axis.

    Stretches or compresses the curve in time,
    `survival(t) = curve.survival(t / acceleration_factor)`, so an
    acceleration factor above 1 pushes every event later and extends
    survival, and one below 1 pulls events earlier.

    Args:
        curve: The comparator curve.
        acceleration_factor: Multiplicative factor on event time. Must be
            positive.

    Returns:
        A new `SurvivalCurve` on the rescaled time axis.

    Example:
        >>> from heormodel.survival import apply_acceleration_factor, weibull
        >>> base = weibull(shape=1.2, scale=6.0)
        >>> stretched = apply_acceleration_factor(base, acceleration_factor=2.0)
        >>> round(float(stretched.survival(12.0)), 5) == round(float(base.survival(6.0)), 5)
        True
    """
    if acceleration_factor <= 0:
        raise ValueError("acceleration_factor must be positive.")

    def cumulative_hazard(time: _FloatArray) -> _FloatArray:
        return curve.cumulative_hazard(np.asarray(time, dtype=np.float64) / acceleration_factor)

    def inverse_cumulative_hazard(value: _FloatArray) -> _FloatArray:
        return acceleration_factor * curve.inverse_cumulative_hazard(
            np.asarray(value, dtype=np.float64)
        )

    return SurvivalCurve(
        cumulative_hazard,
        inverse_cumulative_hazard,
        f"{curve.label} under acceleration factor {acceleration_factor:g}",
    )


def mix(curves: Sequence[SurvivalCurve], weights: Sequence[float]) -> SurvivalCurve:
    """Population mixture: weight several curves' survival functions.

    The mixture survival function is the weighted average
    `sum(weight * curve.survival(t))`, the population curve for a cohort made
    up of subgroups that each follow their own curve, for example responders
    and non-responders to a treatment. The cumulative hazard implied by that
    survival function has no closed-form inverse in general, so sampling from
    the result uses `heormodel.survival.curve.numeric_inverse_cumulative_hazard`.

    Args:
        curves: At least two component curves.
        weights: Mixture weight for each curve, non-negative and summing to 1.

    Returns:
        A new `SurvivalCurve` for the mixture.

    Example:
        >>> from heormodel.survival import exponential, mix
        >>> curve = mix([exponential(rate=0.1), exponential(rate=0.5)], [0.5, 0.5])
        >>> round(float(curve.survival(5.0)), 5)
        0.34431
    """
    if len(curves) < 2:
        raise ValueError("mix requires at least two curves.")
    if len(curves) != len(weights):
        raise ValueError("curves and weights must have the same length.")
    weight_array = np.asarray(weights, dtype=np.float64)
    if np.any(weight_array < 0):
        raise ValueError("weights must be non-negative.")
    if not np.isclose(weight_array.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    pairs = list(zip(weight_array, curves, strict=True))

    def cumulative_hazard(time: _FloatArray) -> _FloatArray:
        time = np.asarray(time, dtype=np.float64)
        mixture_survival = sum(weight * curve.survival(time) for weight, curve in pairs)
        return -np.log(np.clip(mixture_survival, 1e-300, None))

    label = " + ".join(f"{weight:g} x {curve.label}" for weight, curve in pairs)
    return SurvivalCurve(
        cumulative_hazard,
        numeric_inverse_cumulative_hazard(cumulative_hazard),
        f"mixture({label})",
    )


def splice(early: SurvivalCurve, late: SurvivalCurve, cutpoint: float) -> SurvivalCurve:
    """Extrapolation: follow one curve to a cutpoint, then another's shape.

    Up to `cutpoint`, the result is `early`'s survival exactly. Beyond it, the
    result rescales `late`'s survival function to stay continuous at the
    cutpoint, `early.survival(cutpoint) * late.survival(t) / late.survival(cutpoint)`, so
    the tail follows `late`'s hazard shape from that point on. This is the
    standard construction for a long-term model built on an observed curve
    that only covers a follow-up window: `early` is fit to the trial data,
    `late` is a parametric or external curve chosen for the extrapolation.

    Args:
        early: Curve followed up to `cutpoint`.
        late: Curve whose shape drives the extrapolation beyond `cutpoint`.
        cutpoint: Time (years) at which the curve switches. Must be positive.

    Returns:
        A new `SurvivalCurve` that is continuous at `cutpoint`.

    Example:
        >>> from heormodel.survival import exponential, splice
        >>> curve = splice(exponential(rate=0.1), exponential(rate=0.3), cutpoint=5.0)
        >>> round(float(curve.survival(5.0)), 5)
        0.60653
    """
    if cutpoint <= 0:
        raise ValueError("cutpoint must be positive.")
    cut = np.asarray([cutpoint], dtype=np.float64)
    early_survival_at_cut = float(early.survival(cut)[0])
    late_survival_at_cut = float(late.survival(cut)[0])

    def cumulative_hazard(time: _FloatArray) -> _FloatArray:
        time = np.asarray(time, dtype=np.float64)
        survival = np.where(
            time <= cutpoint,
            early.survival(time),
            early_survival_at_cut * late.survival(time) / late_survival_at_cut,
        )
        return -np.log(np.clip(survival, 1e-300, None))

    label = f"{early.label} to {cutpoint:g}, then {late.label}"
    return SurvivalCurve(
        cumulative_hazard, numeric_inverse_cumulative_hazard(cumulative_hazard), label
    )
