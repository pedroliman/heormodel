"""The survival curve value object and the parametric families.

`SurvivalCurve` is one composed value object, not a family hierarchy. It holds a
cumulative-hazard function of time and, optionally, an analytic hazard and an
analytic quantile of the cumulative hazard. Everything a model needs, the
survival function, the hazard, the conditional probability of an event in a
window, event-time sampling, and per-cycle transition probabilities, derives
from those functions. Families (`exponential`, `weibull`, `gompertz`) and curve
algebra are plain functions that return a new `SurvivalCurve`, so curves compose
without inheritance.

Event-time sampling is inverse-transform: draw a unit-exponential threshold and
solve ``cumulative_hazard(t) = threshold``. When a family supplies an analytic
quantile the solve is closed-form; otherwise `SurvivalCurve` inverts the
cumulative hazard by vectorized bisection, which is exact to machine precision
and needs nothing but the cumulative-hazard function, so mixtures, splices, and
adapted fits all sample without special cases.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
TimeFunction = Callable[[Array], Array]

_TINY = 1e-300  # floor for a survival probability before taking its logarithm


@dataclass(frozen=True)
class SurvivalCurve:
    """A survival curve expressed as composable functions of time.

    Args:
        cumulative_hazard: The cumulative hazard ``H(t)``, a vectorized function
            of a time array. The survival function is ``exp(-H(t))``.
        hazard: Optional analytic hazard ``h(t)``. When omitted, `hazard_at`
            falls back to a numeric derivative of the cumulative hazard.
        quantile: Optional inverse of the cumulative hazard: given a threshold
            ``e`` it returns the time ``t`` with ``H(t) = e``. When omitted,
            sampling inverts the cumulative hazard by bisection.
        name: A label carried into provenance and repr.

    Example:
        >>> from heormodel.survival import weibull
        >>> curve = weibull(shape=1.2, scale=6.0)
        >>> round(float(curve.survival(6.0)), 4)  # survival at the scale
        0.3679
        >>> round(float(curve.conditional_probability(0.0, 1.0)), 5)
        0.10994
    """

    cumulative_hazard: TimeFunction
    hazard: TimeFunction | None = None
    quantile: TimeFunction | None = None
    name: str = "survival curve"

    def survival(self, t: ArrayLike) -> Array:
        """Survival probability ``S(t) = exp(-H(t))``."""
        return np.exp(-self.cumulative_hazard(np.asarray(t, dtype=float)))

    def hazard_at(self, t: ArrayLike) -> Array:
        """Instantaneous hazard ``h(t)``, analytic when supplied, else numeric."""
        time = np.asarray(t, dtype=float)
        if self.hazard is not None:
            return np.asarray(self.hazard(time), dtype=float)
        step = 1e-6
        forward = self.cumulative_hazard(time + step)
        backward = self.cumulative_hazard(np.maximum(time - step, 0.0))
        return np.asarray((forward - backward) / (time + step - np.maximum(time - step, 0.0)))

    def conditional_probability(self, t0: ArrayLike, t1: ArrayLike) -> Array:
        """Probability of the event in ``(t0, t1]`` given survival to ``t0``.

        This is ``1 - S(t1) / S(t0)``, the per-window transition probability a
        cohort model consumes.
        """
        survival_0 = self.survival(t0)
        survival_1 = self.survival(t1)
        with np.errstate(invalid="ignore", divide="ignore"):
            probability = 1.0 - np.where(survival_0 > 0, survival_1 / survival_0, 0.0)
        return np.clip(probability, 0.0, 1.0)

    def cycle_transition_probabilities(
        self, n_cycles: int, cycle_length: float = 1.0
    ) -> Array:
        """Per-cycle event probabilities over a cycle grid.

        Returns a length-``n_cycles`` array whose entry ``k`` is the probability
        of the event during cycle ``k``, ``1 - S((k+1) h) / S(k h)`` for cycle
        length ``h``. Stacked into a two-state transition array this is the
        age-varying input `heormodel.models.MarkovModel` accepts.

        Example:
            >>> from heormodel.survival import weibull
            >>> probs = weibull(1.2, 6.0).cycle_transition_probabilities(5)
            >>> [round(float(p), 5) for p in probs]
            [0.10994, 0.14025, 0.15439, 0.16428, 0.17201]
        """
        if n_cycles < 1:
            raise ValueError("n_cycles must be at least one.")
        edges = np.arange(n_cycles + 1, dtype=float) * cycle_length
        survival = self.survival(edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs = 1.0 - np.where(survival[:-1] > 0, survival[1:] / survival[:-1], 0.0)
        return np.clip(probs, 0.0, 1.0)

    def sample_time(self, rng: np.random.Generator, size: int) -> Array:
        """Sample ``size`` event times by inverse-transform sampling.

        Draw a unit-exponential threshold ``e`` and return the time ``t`` with
        ``H(t) = e``, using the analytic quantile when available and bisection
        otherwise. This is the competing-times sampler
        `heormodel.models.MicrosimModel.continuous` calls per destination state.
        """
        threshold = rng.exponential(size=size)
        if self.quantile is not None:
            return np.asarray(self.quantile(threshold), dtype=float)
        return self._invert(threshold)

    def _invert(self, threshold: Array) -> Array:
        """Invert the cumulative hazard by vectorized bisection."""
        threshold = np.asarray(threshold, dtype=float)
        if threshold.size == 0:
            return threshold
        upper = 1.0
        target = float(threshold.max())
        while float(self.cumulative_hazard(np.array([upper]))[0]) < target and upper < 1e15:
            upper *= 2.0
        low = np.zeros_like(threshold)
        high = np.full_like(threshold, upper)
        for _ in range(60):  # 60 halvings resolve the root to machine precision
            mid = 0.5 * (low + high)
            earlier = self.cumulative_hazard(mid) > threshold
            high = np.where(earlier, mid, high)
            low = np.where(earlier, low, mid)
        return 0.5 * (low + high)


def _positive(value: float, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}.")
    return number


def exponential(rate: float) -> SurvivalCurve:
    """Constant-hazard curve ``S(t) = exp(-rate t)``.

    Example:
        >>> from heormodel.survival import exponential
        >>> round(float(exponential(0.1).survival(10.0)), 4)
        0.3679
    """
    rate = _positive(rate, "rate")
    return SurvivalCurve(
        cumulative_hazard=lambda t: rate * np.asarray(t, dtype=float),
        hazard=lambda t: np.full_like(np.asarray(t, dtype=float), rate),
        quantile=lambda e: np.asarray(e, dtype=float) / rate,
        name=f"Exponential(rate={rate:g})",
    )


def weibull(shape: float, scale: float) -> SurvivalCurve:
    """Weibull curve ``S(t) = exp(-(t / scale) ** shape)`` (accelerated-failure-time).

    Example:
        >>> from heormodel.survival import weibull
        >>> round(float(weibull(1.2, 6.0).survival(6.0)), 4)
        0.3679
    """
    shape = _positive(shape, "shape")
    scale = _positive(scale, "scale")
    return SurvivalCurve(
        cumulative_hazard=lambda t: (np.asarray(t, dtype=float) / scale) ** shape,
        hazard=lambda t: (shape / scale) * (np.asarray(t, dtype=float) / scale) ** (shape - 1.0),
        quantile=lambda e: scale * np.asarray(e, dtype=float) ** (1.0 / shape),
        name=f"Weibull(shape={shape:g}, scale={scale:g})",
    )


def gompertz(shape: float, rate: float) -> SurvivalCurve:
    """Gompertz curve with hazard ``rate * exp(shape t)``, the background-mortality form.

    ``shape`` may be any sign; ``rate`` is the hazard at time zero and must be
    positive. The cumulative hazard is ``(rate / shape) (exp(shape t) - 1)``.

    Example:
        >>> from heormodel.survival import gompertz
        >>> round(float(gompertz(0.1, 0.02).hazard_at(0.0)), 4)
        0.02
    """
    rate = _positive(rate, "rate")
    shape = float(shape)
    if shape == 0.0:
        return exponential(rate)

    def cumulative_hazard(t: Array) -> Array:
        return (rate / shape) * (np.exp(shape * np.asarray(t, dtype=float)) - 1.0)

    return SurvivalCurve(
        cumulative_hazard=cumulative_hazard,
        hazard=lambda t: rate * np.exp(shape * np.asarray(t, dtype=float)),
        quantile=lambda e: np.log1p(shape * np.asarray(e, dtype=float) / rate) / shape,
        name=f"Gompertz(shape={shape:g}, rate={rate:g})",
    )
