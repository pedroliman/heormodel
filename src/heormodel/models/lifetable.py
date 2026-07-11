"""Age-dependent mortality from a life table, with exact event-time sampling.

`LifeTable` holds all-cause mortality rates by age band and samples the time to
death for an individual of a given age, optionally under a hazard ratio. The
rate is piecewise constant: each band runs from its age to the next, and the
last band extends indefinitely so death is certain. Sampling inverts the
cumulative hazard exactly, so a continuous-time engine can draw a death time
conditional on the individual's current age at every state entry, the standard
construction for age-dependent background mortality in a discrete-event
simulation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LifeTable:
    """Piecewise-constant mortality rates by age, sampled by inversion.

    Args:
        ages: Start age of each band, strictly increasing. Band ``i`` runs from
            ``ages[i]`` to ``ages[i + 1]``; the last band has no upper end.
        rates: Annual mortality rate in each band, positive, same length as
            ``ages``.

    Example:
        >>> import numpy as np
        >>> from heormodel.models import LifeTable
        >>> table = LifeTable(ages=[0.0, 60.0], rates=[0.01, 0.1])
        >>> round(table.life_expectancy(60.0), 1)
        10.0
        >>> rng = np.random.default_rng(7)
        >>> t = table.sample_time_to_death(rng, np.full(4000, 60.0))
        >>> bool(abs(t.mean() - 10.0) < 0.5)
        True
    """

    def __init__(self, ages: ArrayLike, rates: ArrayLike) -> None:
        ages_arr = np.asarray(ages, dtype=np.float64)
        rates_arr = np.asarray(rates, dtype=np.float64)
        if ages_arr.ndim != 1 or ages_arr.shape != rates_arr.shape or len(ages_arr) == 0:
            raise ValueError("ages and rates must be 1-D arrays of the same nonzero length.")
        if np.any(np.diff(ages_arr) <= 0):
            raise ValueError("ages must be strictly increasing.")
        if not np.all(np.isfinite(rates_arr)) or np.any(rates_arr <= 0):
            raise ValueError("rates must be finite and positive.")
        self._ages = ages_arr
        self._rates = rates_arr
        widths = np.diff(ages_arr)
        # Cumulative hazard at each band start, measured from ages[0].
        self._cumhaz = np.concatenate([[0.0], np.cumsum(rates_arr[:-1] * widths)])

    def rate(self, age: ArrayLike) -> NDArray[np.float64]:
        """Annual mortality rate at each age.

        Example:
            >>> from heormodel.models import LifeTable
            >>> LifeTable(ages=[0.0, 60.0], rates=[0.01, 0.1]).rate([30.0, 75.0]).tolist()
            [0.01, 0.1]
        """
        idx = self._band(age)
        return self._rates[idx]

    def cumulative_hazard(self, age: ArrayLike) -> NDArray[np.float64]:
        """Cumulative mortality hazard from the first table age to each age.

        Example:
            >>> from heormodel.models import LifeTable
            >>> float(LifeTable(ages=[0.0, 60.0], rates=[0.01, 0.1]).cumulative_hazard(70.0))
            1.6
        """
        age_arr = np.asarray(age, dtype=np.float64)
        idx = self._band(age_arr)
        return self._cumhaz[idx] + self._rates[idx] * (age_arr - self._ages[idx])

    def sample_time_to_death(
        self,
        rng: np.random.Generator,
        age: ArrayLike,
        *,
        hazard_ratio: ArrayLike = 1.0,
    ) -> NDArray[np.float64]:
        """Sample years until death for individuals of the given ages.

        Draws by inverting the cumulative hazard: the time to death ``t``
        solves ``hr * (H(age + t) - H(age)) = e`` with ``e`` standard
        exponential, so each draw is conditional on having survived to
        ``age``. The hazard ratio scales the whole remaining hazard, the
        proportional-hazards form used for excess disease mortality.

        Args:
            rng: Random generator supplying the exponential draws.
            age: Current age of each individual, at or above the first table
                age.
            hazard_ratio: Multiplier on the mortality rate, scalar or one
                value per individual.

        Returns:
            Years from ``age`` to death, one value per individual.

        Example:
            >>> import numpy as np
            >>> from heormodel.models import LifeTable
            >>> table = LifeTable(ages=[0.0], rates=[0.05])
            >>> t = table.sample_time_to_death(
            ...     np.random.default_rng(0), np.zeros(4000), hazard_ratio=5.0)
            >>> bool(abs(t.mean() - 4.0) < 0.2)  # exponential with rate 0.25
            True
        """
        age_arr = np.atleast_1d(np.asarray(age, dtype=np.float64))
        hr = np.broadcast_to(np.asarray(hazard_ratio, dtype=np.float64), age_arr.shape)
        if np.any(hr <= 0):
            raise ValueError("hazard_ratio must be positive.")
        target = self.cumulative_hazard(age_arr) + rng.exponential(size=age_arr.shape) / hr
        idx = np.clip(np.searchsorted(self._cumhaz, target, side="right") - 1, 0, None)
        age_at_death = self._ages[idx] + (target - self._cumhaz[idx]) / self._rates[idx]
        return age_at_death - age_arr

    def life_expectancy(self, age: float, *, hazard_ratio: float = 1.0) -> float:
        """Remaining life expectancy at an age, exact for the piecewise rates.

        Integrates the survival function band by band, so it is the analytic
        mean of `sample_time_to_death` and a direct check on simulated death
        times.

        Example:
            >>> from heormodel.models import LifeTable
            >>> LifeTable(ages=[0.0], rates=[0.02]).life_expectancy(30.0)
            50.0
        """
        start = float(age)
        idx = int(self._band(start))
        expectancy = 0.0
        survival = 1.0
        bands = len(self._ages)
        for i in range(idx, bands):
            m = float(self._rates[i]) * float(hazard_ratio)
            lo = max(start, float(self._ages[i]))
            if i + 1 < bands:
                width = float(self._ages[i + 1]) - lo
                expectancy += survival * (1.0 - np.exp(-m * width)) / m
                survival *= float(np.exp(-m * width))
            else:
                expectancy += survival / m
        return float(expectancy)

    def _band(self, age: ArrayLike) -> NDArray[np.int64]:
        age_arr = np.asarray(age, dtype=np.float64)
        if np.any(age_arr < self._ages[0]):
            raise ValueError(f"age below the first table age {self._ages[0]}.")
        return np.clip(np.searchsorted(self._ages, age_arr, side="right") - 1, 0, None)
