"""Distribution specifications for probabilistic sensitivity analysis.

Each distribution is a small immutable spec object backed by ``scipy.stats``.
Specs expose the quantile function (``ppf``), which is what the correlated
Gaussian-copula sampler in `heormodel.params.sampling` needs, plus direct
sampling and moments for convenience.

Method-of-moments constructors (``from_mean_se``) build the distribution
that matches a published point estimate and standard error, the everyday
workflow when parameterising a probabilistic sensitivity analysis from the literature.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


class Distribution(ABC):
    """Abstract base class for univariate parameter distributions.

    Subclasses wrap a frozen ``scipy.stats`` distribution and are the scalar
    building blocks of a `ParameterSet`.
    """

    @abstractmethod
    def _frozen(self) -> Any:
        """Return the frozen ``scipy.stats`` distribution."""

    def ppf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Quantile function (inverse CDF) evaluated at ``u`` in (0, 1)."""
        return np.asarray(self._frozen().ppf(u), dtype=np.float64)

    def sample(self, n: int, rng: np.random.Generator | int | None = None) -> NDArray[np.float64]:
        """Draw ``n`` independent samples.

        Example:
            >>> from heormodel.params import Beta
            >>> Beta(2, 5).sample(3, rng=1).shape
            (3,)
        """
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        return np.asarray(self._frozen().rvs(size=n, random_state=gen), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean."""
        return float(self._frozen().mean())

    def sd(self) -> float:
        """Distribution standard deviation."""
        return float(self._frozen().std())

    def __repr__(self) -> str:
        """Spec string with floats at 6 significant digits.

        Keeps provenance records readable: method-of-moments constructors
        produce floats whose full repr is noise.

        Example:
            >>> from heormodel.params import Beta
            >>> Beta.from_mean_se(0.2, 0.05)
            Beta(alpha=12.6, beta=50.4)
        """
        parts = []
        for name in getattr(self, "__dataclass_fields__", {}):
            value = getattr(self, name)
            text = format(value, ".6g") if isinstance(value, float) else repr(value)
            parts.append(f"{name}={text}")
        return f"{type(self).__name__}({', '.join(parts)})"


@dataclass(frozen=True, repr=False)
class Beta(Distribution):
    """Beta distribution, for probabilities and utilities on [0, 1].

    Example:
        >>> from heormodel.params import Beta
        >>> d = Beta.from_mean_se(0.2, 0.05)
        >>> round(d.mean(), 3)
        0.2
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta requires alpha > 0 and beta > 0.")

    @classmethod
    def from_mean_se(cls, mean: float, se: float) -> Beta:
        """Method-of-moments Beta from a mean and standard error.

        Requires ``0 < mean < 1`` and ``se**2 < mean * (1 - mean)``.
        """
        if not 0.0 < mean < 1.0:
            raise ValueError("Beta mean must lie strictly in (0, 1).")
        nu = mean * (1.0 - mean) / se**2 - 1.0
        if nu <= 0.0:
            raise ValueError("se too large for a Beta with this mean: need se^2 < mean*(1-mean).")
        return cls(alpha=mean * nu, beta=(1.0 - mean) * nu)

    def _frozen(self) -> Any:
        return stats.beta(self.alpha, self.beta)


@dataclass(frozen=True, repr=False)
class Gamma(Distribution):
    """Gamma distribution, for non-negative quantities such as costs.

    Parameterised by ``shape`` (k) and ``scale`` (theta); mean = shape * scale.

    Example:
        >>> from heormodel.params import Gamma
        >>> d = Gamma.from_mean_se(1000.0, 100.0)
        >>> round(d.sd(), 1)
        100.0
    """

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("Gamma requires shape > 0 and scale > 0.")

    @classmethod
    def from_mean_se(cls, mean: float, se: float) -> Gamma:
        """Method-of-moments Gamma from a mean and standard error."""
        if mean <= 0 or se <= 0:
            raise ValueError("Gamma mean and se must be positive.")
        return cls(shape=(mean / se) ** 2, scale=se**2 / mean)

    def _frozen(self) -> Any:
        return stats.gamma(self.shape, scale=self.scale)


@dataclass(frozen=True, repr=False)
class LogNormal(Distribution):
    """Lognormal distribution, for relative risks and skewed costs.

    Parameterised on the log scale: ``mu`` and ``sigma`` are the mean and
    standard deviation of ``log(X)``.

    Example:
        >>> from heormodel.params import LogNormal
        >>> d = LogNormal.from_mean_se(2.0, 0.5)
        >>> round(d.mean(), 3)
        2.0
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("LogNormal requires sigma > 0.")

    @classmethod
    def from_mean_se(cls, mean: float, se: float) -> LogNormal:
        """Method-of-moments LogNormal matching the natural-scale mean and SE."""
        if mean <= 0 or se <= 0:
            raise ValueError("LogNormal mean and se must be positive.")
        sigma2 = math.log(1.0 + (se / mean) ** 2)
        return cls(mu=math.log(mean) - sigma2 / 2.0, sigma=math.sqrt(sigma2))

    def _frozen(self) -> Any:
        return stats.lognorm(self.sigma, scale=math.exp(self.mu))


@dataclass(frozen=True, repr=False)
class Normal(Distribution):
    """Normal distribution.

    Example:
        >>> from heormodel.params import Normal
        >>> Normal(0.0, 1.0).mean()
        0.0
    """

    mean_: float
    sd_: float

    def __post_init__(self) -> None:
        if self.sd_ <= 0:
            raise ValueError("Normal requires sd > 0.")

    def _frozen(self) -> Any:
        return stats.norm(self.mean_, self.sd_)


@dataclass(frozen=True, repr=False)
class Uniform(Distribution):
    """Uniform distribution on [low, high].

    Example:
        >>> from heormodel.params import Uniform
        >>> Uniform(0.0, 2.0).mean()
        1.0
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("Uniform requires high > low.")

    def _frozen(self) -> Any:
        return stats.uniform(self.low, self.high - self.low)


@dataclass(frozen=True, repr=False)
class Fixed(Distribution):
    """Degenerate distribution: a parameter held constant across iterations.

    Example:
        >>> from heormodel.params import Fixed
        >>> Fixed(3.5).sample(2).tolist()
        [3.5, 3.5]
    """

    value: float

    def _frozen(self) -> Any:
        raise NotImplementedError  # never used; all methods overridden

    def ppf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Constant quantile function."""
        return np.full_like(np.asarray(u, dtype=np.float64), self.value)

    def sample(self, n: int, rng: np.random.Generator | int | None = None) -> NDArray[np.float64]:
        """Return ``n`` copies of the fixed value."""
        return np.full(n, self.value, dtype=np.float64)

    def mean(self) -> float:
        """The fixed value."""
        return self.value

    def sd(self) -> float:
        """Zero."""
        return 0.0


@dataclass(frozen=True)
class Dirichlet:
    """Dirichlet distribution: a vector of transition probabilities summing to 1.

    Multivariate: inside a `ParameterSet` a
    Dirichlet named ``p`` with component names ``("a", "b")`` expands to draw
    columns ``p[a]`` and ``p[b]``. Sampling uses the standard construction of
    independent Gamma(alpha_i, 1) marginals normalised to sum to one, which
    makes each component compatible with the copula machinery. Leave the
    correlation entries for Dirichlet component columns at zero; correlating
    the underlying gammas with other parameters distorts the Dirichlet.

    Example:
        >>> from heormodel.params import Dirichlet
        >>> d = Dirichlet((80, 15, 5), names=("stay", "progress", "die"))
        >>> d.mean().round(2).tolist()
        [0.8, 0.15, 0.05]
    """

    alpha: tuple[float, ...]
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.alpha) < 2:
            raise ValueError("Dirichlet requires at least two components.")
        if any(a <= 0 for a in self.alpha):
            raise ValueError("Dirichlet concentrations must be positive.")
        if self.names is not None and len(self.names) != len(self.alpha):
            raise ValueError("names must match the number of concentrations.")

    @property
    def n_components(self) -> int:
        """Number of components in the probability vector."""
        return len(self.alpha)

    def component_labels(self, base: str) -> list[str]:
        """Expanded column names, e.g. ``base[stay]`` or ``base[0]``."""
        keys = (
            self.names
            if self.names is not None
            else tuple(str(i) for i in range(self.n_components))
        )
        return [f"{base}[{k}]" for k in keys]

    def component_gammas(self) -> list[Gamma]:
        """The independent Gamma(alpha_i, 1) marginals used for sampling."""
        return [Gamma(shape=a, scale=1.0) for a in self.alpha]

    def mean(self) -> NDArray[np.float64]:
        """Component means alpha_i / sum(alpha)."""
        a = np.asarray(self.alpha, dtype=np.float64)
        return a / a.sum()

    def sample(self, n: int, rng: np.random.Generator | int | None = None) -> NDArray[np.float64]:
        """Draw ``n`` probability vectors, shape ``(n, n_components)``."""
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        return gen.dirichlet(np.asarray(self.alpha, dtype=np.float64), size=n)
