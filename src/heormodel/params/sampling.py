"""Correlated probabilistic sampling of parameter sets.

A `ParameterSet` maps parameter names to distribution specs and
produces the **parameter draw matrix**: a tidy ``pandas.DataFrame`` with one
row per iteration (index named ``"iteration"``) and one column per scalar
parameter. That matrix is the shared currency of the package: model engines
consume it, and value-of-information analyses trace outcomes back to it
through the shared iteration index.

Correlation between parameters is induced with a Gaussian copula on
Spearman rank correlations, so each parameter keeps its exact marginal
distribution while the joint rank correlation matches the target.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr

from heormodel._util import as_rng
from heormodel.params.distributions import Dirichlet, Distribution

AnyDistribution: TypeAlias = Distribution | Dirichlet
CorrelationSpec: TypeAlias = pd.DataFrame | Mapping[tuple[str, str], float] | None

_U_EPS = 1e-12


def _nearest_correlation(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip eigenvalues to make ``r`` positive semi-definite, keep unit diagonal."""
    vals, vecs = np.linalg.eigh(r)
    if vals.min() >= 1e-10:
        return r
    vals = np.clip(vals, 1e-10, None)
    fixed = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


class ParameterSet:
    """A named collection of parameter distributions with optional correlation.

    Args:
        distributions: Mapping of parameter name to a univariate
            `Distribution` or a
            `Dirichlet` (which expands to
            one column per component, named ``name[component]``).
        correlation: Target Spearman rank correlations between scalar
            parameter columns. Either a symmetric ``DataFrame`` labelled by
            column names, or a mapping of ``(name_a, name_b) -> rho`` pairs
            (unlisted pairs are independent). ``None`` means independent.

    Example:
        >>> from heormodel.params import Beta, Gamma, ParameterSet
        >>> ps = ParameterSet(
        ...     {"p_sick": Beta.from_mean_se(0.2, 0.05),
        ...      "c_sick": Gamma.from_mean_se(1000, 150)},
        ...     correlation={("p_sick", "c_sick"): 0.5},
        ... )
        >>> draws = ps.sample(1000, seed=42)
        >>> list(draws.columns)
        ['p_sick', 'c_sick']
        >>> draws.index.name
        'iteration'
    """

    def __init__(
        self,
        distributions: Mapping[str, AnyDistribution],
        correlation: CorrelationSpec = None,
    ) -> None:
        if not distributions:
            raise ValueError("ParameterSet requires at least one distribution.")
        self.distributions: dict[str, AnyDistribution] = dict(distributions)
        self._columns: list[str] = []
        self._marginals: list[Distribution] = []
        self._dirichlet_groups: list[tuple[int, int]] = []  # [start, stop) column slices
        for name, dist in self.distributions.items():
            if isinstance(dist, Dirichlet):
                start = len(self._columns)
                self._columns.extend(dist.component_labels(name))
                self._marginals.extend(dist.component_gammas())
                self._dirichlet_groups.append((start, len(self._columns)))
            else:
                self._columns.append(name)
                self._marginals.append(dist)
        self._corr = self._build_correlation(correlation)

    @property
    def names(self) -> list[str]:
        """Expanded scalar column names of the draw matrix."""
        return list(self._columns)

    def correlation_matrix(self) -> pd.DataFrame:
        """The target Spearman correlation matrix over scalar columns."""
        return pd.DataFrame(self._corr.copy(), index=self.names, columns=self.names)

    def _build_correlation(self, spec: CorrelationSpec) -> NDArray[np.float64]:
        k = len(self._columns)
        r = np.eye(k, dtype=np.float64)
        if spec is None:
            return r
        idx = {name: i for i, name in enumerate(self._columns)}
        if isinstance(spec, pd.DataFrame):
            for a in spec.index:
                for b in spec.columns:
                    if a not in idx or b not in idx:
                        raise KeyError(f"Unknown parameter in correlation spec: {a!r}/{b!r}")
                    r[idx[a], idx[b]] = float(spec.loc[a, b])
        else:
            for (a, b), rho in spec.items():
                if a not in idx:
                    raise KeyError(f"Unknown parameter in correlation spec: {a!r}")
                if b not in idx:
                    raise KeyError(f"Unknown parameter in correlation spec: {b!r}")
                r[idx[a], idx[b]] = float(rho)
                r[idx[b], idx[a]] = float(rho)
        np.fill_diagonal(r, 1.0)
        if not np.allclose(r, r.T):
            raise ValueError("Correlation matrix must be symmetric.")
        if np.abs(r).max() > 1.0:
            raise ValueError("Correlations must lie in [-1, 1].")
        return _nearest_correlation(r)

    def sample(self, n: int, seed: int | np.random.Generator | None = None) -> pd.DataFrame:
        """Draw the parameter matrix for ``n`` iterations.

        Uses a Gaussian copula: correlated standard normals are mapped to
        uniforms and pushed through each marginal quantile function, so
        marginals are exact and rank correlations approximate the target
        (the Spearman target is converted to the equivalent normal
        correlation via ``2 * sin(pi * rho / 6)``).

        Args:
            n: Number of iterations (rows).
            seed: Integer seed or ``numpy`` Generator for reproducibility.

        Returns:
            DataFrame with ``RangeIndex`` named ``"iteration"`` and one
            column per scalar parameter.

        Example:
            >>> from heormodel.params import Normal, ParameterSet
            >>> ps = ParameterSet({"x": Normal(0, 1)})
            >>> ps.sample(5, seed=7).shape
            (5, 1)
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        rng = as_rng(seed)
        k = len(self._columns)
        pearson = 2.0 * np.sin(np.pi * self._corr / 6.0)
        pearson = _nearest_correlation(pearson)
        chol = np.linalg.cholesky(pearson)
        z = rng.standard_normal((n, k)) @ chol.T
        u = np.clip(ndtr(z), _U_EPS, 1.0 - _U_EPS)
        values = np.empty((n, k), dtype=np.float64)
        for j, marginal in enumerate(self._marginals):
            values[:, j] = marginal.ppf(u[:, j])
        for start, stop in self._dirichlet_groups:
            block = values[:, start:stop]
            values[:, start:stop] = block / block.sum(axis=1, keepdims=True)
        return pd.DataFrame(values, columns=self._columns, index=pd.RangeIndex(n, name="iteration"))

    def means(self) -> pd.Series:
        """Analytic means of each scalar column (Dirichlet components included).

        Example:
            >>> from heormodel.params import Fixed, ParameterSet
            >>> float(ParameterSet({"a": Fixed(2.0)}).means()["a"])
            2.0
        """
        out: dict[str, float] = {}
        for name, dist in self.distributions.items():
            if isinstance(dist, Dirichlet):
                for label, m in zip(dist.component_labels(name), dist.mean(), strict=True):
                    out[label] = float(m)
            else:
                out[name] = dist.mean()
        return pd.Series(out, name="mean")

    def at_means(self) -> pd.DataFrame:
        """Wrap the analytic means as a one-row base-case draw matrix.

        Equivalent to ``single_draw(self.means().to_dict())``: the
        deterministic run at point values that sits next to the probabilistic analysis.

        Example:
            >>> from heormodel.params import Fixed, ParameterSet
            >>> ParameterSet({"a": Fixed(2.0)}).at_means().shape
            (1, 1)
        """
        from heormodel.params.inputs import single_draw

        return single_draw(self.means().to_dict())

    def spec(self) -> dict[str, str]:
        """Human-readable provenance record of each distribution spec.

        Example:
            >>> from heormodel.params import Normal, ParameterSet
            >>> ParameterSet({"x": Normal(0, 1)}).spec()
            {'x': 'Normal(mean_=0, sd_=1)'}
        """
        return {name: repr(dist) for name, dist in self.distributions.items()}
