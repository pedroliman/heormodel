"""The survival curve value object and the three distribution families.

A `SurvivalCurve` wraps a cumulative hazard function of time, the quantity
that determines everything else about the curve: the survival function is
`exp(-cumulative_hazard(t))`, and the inverse cumulative hazard drives event-time
sampling. `exponential`, `weibull`, and `gompertz` build a `SurvivalCurve` from
distribution parameters; `heormodel.survival.algebra` and
`heormodel.survival.lifelines_adapter` build curves from other curves or from a
fitted model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]


def numeric_inverse_cumulative_hazard(
    cumulative_hazard: Callable[[_FloatArray], _FloatArray],
) -> Callable[[_FloatArray], _FloatArray]:
    """Invert a monotone cumulative hazard function by bisection.

    Every `SurvivalCurve` needs an inverse cumulative hazard for event-time
    sampling. The three families below have a closed form; a curve built by
    `heormodel.survival.algebra.mix` or `heormodel.survival.algebra.splice`, or
    read from a fitted model by `heormodel.survival.lifelines_adapter.from_lifelines`,
    generally does not, so this bisects instead. The search first doubles an
    upper bound until the cumulative hazard there meets or exceeds the target,
    then bisects 60 times, accurate to about 1 part in `2**60` of that bound.

    Args:
        cumulative_hazard: A function mapping non-negative times to a
            non-decreasing cumulative hazard, `0` at `t=0` and unbounded as
            `t` grows (every proper survival curve has one).

    Returns:
        A function mapping a target cumulative hazard value to the time at
        which the curve reaches it.

    Example:
        >>> import numpy as np
        >>> from heormodel.survival.curve import numeric_inverse_cumulative_hazard
        >>> inverse = numeric_inverse_cumulative_hazard(lambda t: 0.2 * t)
        >>> round(float(inverse(np.array([1.0]))[0]), 5)
        5.0
    """

    def inverse(value: _FloatArray) -> _FloatArray:
        value = np.asarray(value, dtype=np.float64)
        low = np.zeros_like(value)
        high = np.ones_like(value)
        for _ in range(200):
            too_low = cumulative_hazard(high) < value
            if not too_low.any():
                break
            high = np.where(too_low, high * 2.0, high)
        else:  # pragma: no cover - only a non-diverging cumulative hazard reaches this
            raise RuntimeError(
                "Could not bracket the inverse cumulative hazard; the curve's "
                "cumulative hazard must diverge as t grows."
            )
        for _ in range(60):
            mid = 0.5 * (low + high)
            too_low = cumulative_hazard(mid) < value
            low = np.where(too_low, mid, low)
            high = np.where(too_low, high, mid)
        return 0.5 * (low + high)

    return inverse


@dataclass(frozen=True)
class SurvivalCurve:
    """A survival curve as its cumulative hazard function, plus how to sample from it.

    The families below (`exponential`, `weibull`, `gompertz`), the curve
    algebra in `heormodel.survival.algebra`, and
    `heormodel.survival.lifelines_adapter.from_lifelines` are the ways to build
    one; nothing about the analysis code constructs a `SurvivalCurve` directly.

    Args:
        cumulative_hazard: The curve's cumulative hazard `H(t)`, mapping an
            array of non-negative times to `H` at each one. `survival` is
            `exp(-H(t))`.
        inverse_cumulative_hazard: The inverse of `cumulative_hazard`, mapping
            a target hazard value to the time at which the curve reaches it.
            `sample_time` uses it for inverse-transform sampling.
        label: A short description of the curve for `repr` and for reading
            back a model built from several curves, for example
            `"weibull(shape=1.2, scale=6)"`.

    Example:
        >>> from heormodel.survival import weibull
        >>> curve = weibull(shape=1.2, scale=6.0)
        >>> round(float(curve.survival(6.0)), 5)
        0.36788
    """

    cumulative_hazard: Callable[[_FloatArray], _FloatArray]
    inverse_cumulative_hazard: Callable[[_FloatArray], _FloatArray]
    label: str = "curve"

    def survival(self, time: _FloatArray | float) -> _FloatArray:
        """Survival probability `S(t) = exp(-cumulative_hazard(t))`.

        Args:
            time: Time (years), scalar or array.

        Returns:
            Survival probability at each time.

        Example:
            >>> from heormodel.survival import exponential
            >>> curve = exponential(rate=0.2)
            >>> round(float(curve.survival(5.0)), 5)
            0.36788
        """
        return np.exp(-self.cumulative_hazard(np.asarray(time, dtype=np.float64)))

    def sample_time(self, rng: np.random.Generator, size: int) -> _FloatArray:
        """Sample event times by inverse-transform sampling.

        The cumulative hazard evaluated at a curve's own event time follows a
        standard exponential distribution, so drawing an exponential variate
        and inverting the cumulative hazard samples an event time directly,
        without inverting the survival function on the probability scale.

        Args:
            rng: A `numpy` random generator, typically the one an engine's
                model function receives.
            size: Number of event times to draw.

        Returns:
            Array of `size` sampled event times.

        Example:
            >>> import numpy as np
            >>> from heormodel.survival import exponential
            >>> curve = exponential(rate=0.2)
            >>> rng = np.random.default_rng(0)
            >>> times = curve.sample_time(rng, 5)
            >>> times.shape
            (5,)
        """
        return self.inverse_cumulative_hazard(rng.standard_exponential(size))

    def __repr__(self) -> str:
        return f"SurvivalCurve({self.label!r})"


def exponential(rate: float) -> SurvivalCurve:
    """Exponential survival curve with a constant hazard.

    The special case of `weibull` with `shape=1`, kept as its own family
    because a constant hazard has a closed-form discounted life expectancy,
    `1 / (discount_rate + rate)`, used to check `heormodel.survival.algebra.apply_hazard_ratio`.

    Args:
        rate: Constant hazard (events per year). Must be positive.

    Returns:
        A `SurvivalCurve` with `survival(t) = exp(-rate * t)`.

    Example:
        >>> from heormodel.survival import exponential
        >>> curve = exponential(rate=0.2)
        >>> round(float(curve.survival(5.0)), 5)
        0.36788
    """
    if rate <= 0:
        raise ValueError("rate must be positive.")

    def cumulative_hazard(time: _FloatArray) -> _FloatArray:
        return rate * np.asarray(time, dtype=np.float64)

    def inverse_cumulative_hazard(value: _FloatArray) -> _FloatArray:
        return np.asarray(value, dtype=np.float64) / rate

    return SurvivalCurve(
        cumulative_hazard, inverse_cumulative_hazard, f"exponential(rate={rate:g})"
    )


def weibull(shape: float, scale: float) -> SurvivalCurve:
    """Weibull survival curve, `S(t) = exp(-(t / scale) ** shape)`.

    The accelerated-failure-time parameterization: `scale` sets the time
    axis, in the curve's time unit (typically years), and `shape` controls
    whether the hazard rises (`shape > 1`), falls (`shape < 1`), or stays
    constant (`shape == 1`, equivalent to `exponential(1 / scale)`).

    Args:
        shape: Weibull shape parameter. Must be positive.
        scale: Weibull scale parameter, in the curve's time unit. Must be
            positive.

    Returns:
        A `SurvivalCurve` with `survival(t) = exp(-(t / scale) ** shape)`.

    Example:
        >>> from heormodel.survival import weibull
        >>> curve = weibull(shape=1.2, scale=6.0)
        >>> round(float(curve.survival(6.0)), 5)
        0.36788
    """
    if shape <= 0:
        raise ValueError("shape must be positive.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    def cumulative_hazard(time: _FloatArray) -> _FloatArray:
        return (np.asarray(time, dtype=np.float64) / scale) ** shape

    def inverse_cumulative_hazard(value: _FloatArray) -> _FloatArray:
        return scale * np.asarray(value, dtype=np.float64) ** (1.0 / shape)

    return SurvivalCurve(
        cumulative_hazard, inverse_cumulative_hazard, f"weibull(shape={shape:g}, scale={scale:g})"
    )


def gompertz(shape: float, rate: float) -> SurvivalCurve:
    """Gompertz survival curve with an exponentially increasing hazard.

    The hazard is `rate * exp(shape * t)`, the standard model for mortality
    that accelerates with age. `shape` must be positive so the cumulative
    hazard diverges as `t` grows; a curve with a hazard that eventually falls
    or flattens is not one this family represents.

    Args:
        shape: Exponential growth rate of the hazard. Must be positive.
        rate: Hazard at time 0. Must be positive.

    Returns:
        A `SurvivalCurve` with cumulative hazard `(rate / shape) * (exp(shape * t) - 1)`.

    Example:
        >>> from heormodel.survival import gompertz
        >>> curve = gompertz(shape=0.1, rate=0.01)
        >>> round(float(curve.survival(10.0)), 5)
        0.84212
    """
    if shape <= 0:
        raise ValueError("shape must be positive.")
    if rate <= 0:
        raise ValueError("rate must be positive.")

    def cumulative_hazard(time: _FloatArray) -> _FloatArray:
        return (rate / shape) * (np.exp(shape * np.asarray(time, dtype=np.float64)) - 1.0)

    def inverse_cumulative_hazard(value: _FloatArray) -> _FloatArray:
        return np.log1p(shape * np.asarray(value, dtype=np.float64) / rate) / shape

    return SurvivalCurve(
        cumulative_hazard, inverse_cumulative_hazard, f"gompertz(shape={shape:g}, rate={rate:g})"
    )
