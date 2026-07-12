"""Model calibration via approximate Bayesian computation (pyabc).

Bridges ``heormodel`` parameter specs to ``pyabc`` priors, runs ABC-SMC against
observed calibration targets, and returns the posterior as an
equally-weighted parameter draw matrix carrying the standard ``iteration``
index, so calibrated draws flow through `heormodel.run.run_psa` and the
analysis layer exactly like draws from `heormodel.params.ParameterSet.sample`.

``pyabc`` is an optional dependency: install with ``uv pip install
'heormodel[calibration]'``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heormodel._util import require_optional, resample_to_iterations
from heormodel.params.distributions import (
    Beta,
    Dirichlet,
    Distribution,
    Gamma,
    LogNormal,
    Normal,
    Uniform,
)

#: A calibration simulator: parameter values -> simulated calibration targets.
TargetSimulator = Callable[[dict[str, float]], dict[str, float]]


def _require_pyabc() -> Any:
    return require_optional("pyabc", feature="Calibration", extra="calibration")


def to_pyabc_prior(distributions: Mapping[str, Distribution | Dirichlet]) -> Any:
    """Translate ``heormodel`` distribution specs into a ``pyabc`` prior.

    Supported: `Beta`, `Gamma`, `LogNormal`,
    `Normal`, `Uniform`. Dirichlet and Fixed parameters cannot
    be calibrated directly; hold them constant inside the simulator instead.

    Example:
        >>> from heormodel.calibrate import to_pyabc_prior  # doctest: +SKIP
        >>> from heormodel.params import Beta
        >>> prior = to_pyabc_prior({"p": Beta(2, 8)})  # doctest: +SKIP
    """
    pyabc = _require_pyabc()
    rvs: dict[str, Any] = {}
    for name, dist in distributions.items():
        if isinstance(dist, Beta):
            rvs[name] = pyabc.RV("beta", dist.alpha, dist.beta)
        elif isinstance(dist, Gamma):
            rvs[name] = pyabc.RV("gamma", dist.shape, scale=dist.scale)
        elif isinstance(dist, LogNormal):
            rvs[name] = pyabc.RV("lognorm", dist.sigma, scale=float(np.exp(dist.mu)))
        elif isinstance(dist, Normal):
            rvs[name] = pyabc.RV("norm", dist.mean_, dist.sd_)
        elif isinstance(dist, Uniform):
            rvs[name] = pyabc.RV("uniform", dist.low, dist.high - dist.low)
        else:
            raise TypeError(
                f"Parameter {name!r}: {type(dist).__name__} priors are not supported "
                "for ABC calibration; hold it constant inside the simulator."
            )
    return pyabc.Distribution(**rvs)


@dataclass
class CalibrationResult:
    """Posterior draws and diagnostics from an ABC-SMC calibration.

    Attributes:
        posterior: Equally-weighted posterior draw matrix with a
            ``RangeIndex`` named ``iteration``, ready for
            `heormodel.run.run_psa`.
        weighted: The raw weighted particle population (columns = parameters,
            plus a ``weight`` column).
        n_populations: Number of ABC-SMC populations run.
        final_epsilon: Acceptance threshold of the final population.
    """

    posterior: pd.DataFrame
    weighted: pd.DataFrame
    n_populations: int
    final_epsilon: float


def abc_calibrate(
    simulator: TargetSimulator,
    priors: Mapping[str, Distribution | Dirichlet],
    observed: Mapping[str, float],
    *,
    population_size: int = 200,
    max_populations: int = 8,
    min_epsilon: float = 0.0,
    n_posterior: int | None = None,
    seed: int | None = None,
    db_path: str | Path | None = None,
) -> CalibrationResult:
    """Calibrate model parameters to observed targets with ABC-SMC.

    Args:
        simulator: Maps a parameter dict to simulated calibration targets
            (same keys as ``observed``).
        priors: Parameter priors as ``heormodel`` distribution specs.
        observed: Observed calibration target values.
        population_size: Particles per ABC-SMC population.
        max_populations: Maximum number of populations.
        min_epsilon: Stop once the acceptance threshold reaches this value.
        n_posterior: Rows in the returned equally-weighted posterior matrix
            (default: the final population size).
        seed: Seed for the weighted-to-equal resampling step. (The ABC run
            itself uses pyabc's internal randomness.)
        db_path: Where to store pyabc's bookkeeping database (default: a
            temporary file).

    Returns:
        A `CalibrationResult` whose ``posterior`` plugs directly into
        the probabilistic analysis.

    Example:
        >>> from heormodel.calibrate import abc_calibrate  # doctest: +SKIP
        >>> from heormodel.params import Uniform
        >>> result = abc_calibrate(  # doctest: +SKIP
        ...     simulator=lambda p: {"prevalence": p["risk"] * 0.5},
        ...     priors={"risk": Uniform(0.0, 1.0)},
        ...     observed={"prevalence": 0.15},
        ... )
    """
    pyabc = _require_pyabc()
    from pyabc.sampler import SingleCoreSampler

    prior = to_pyabc_prior(priors)
    keys = sorted(observed)

    def model(parameter: Mapping[str, float]) -> dict[str, float]:
        return dict(simulator(dict(parameter)))

    distance = pyabc.PNormDistance(p=2)
    abc = pyabc.ABCSMC(
        model,
        prior,
        distance,
        population_size=population_size,
        sampler=SingleCoreSampler(),
    )
    if db_path is None:
        db_file = Path(tempfile.mkdtemp()) / "heormodel_abc.db"
    else:
        db_file = Path(db_path)
    abc.new("sqlite:///" + str(db_file), {k: float(observed[k]) for k in keys})
    history = abc.run(minimum_epsilon=min_epsilon, max_nr_populations=max_populations)

    particles, weights = history.get_distribution()
    weighted = particles.copy()
    weighted["weight"] = weights
    rng = np.random.default_rng(seed)
    n_out = n_posterior or len(particles)
    picks = rng.choice(len(particles), size=n_out, p=np.asarray(weights) / np.sum(weights))
    posterior = resample_to_iterations(particles, picks)
    posterior.columns.name = None
    epsilons = history.get_all_populations()["epsilon"]
    return CalibrationResult(
        posterior=posterior,
        weighted=weighted,
        n_populations=history.max_t + 1,
        final_epsilon=float(epsilons.iloc[-1]),
    )
