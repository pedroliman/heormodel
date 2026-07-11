"""heormodel: health economic evaluation in Python.

One parameter draw matrix flows through swappable model engines into a
shared analysis layer. Engines differ internally but share a contract on
their outputs, the `Outcomes` structure indexed by
``(intervention, iteration)``, which makes cost-effectiveness and
value-of-information analysis engine-agnostic. Outputs from any external
model enter the same pipeline via `as_outcomes`.

Subpackages:
    - `heormodel.params`: distributions and correlated probabilistic sampling
    - `heormodel.models`: engines behind the output contract, plus state occupancy
      over time from an event history
    - `heormodel.run`: seeds, run loop, bring-your-own-outputs ingestion
    - `heormodel.cea`: incremental analysis, frontier, NMB/NHB, CEAC/CEAF, expected loss
    - `heormodel.dsa`: one-way, one-at-a-time, and grid deterministic sensitivity designs
    - `heormodel.voi`: EVPI, EVPPI, EVSI
    - `heormodel.calibrate`: ABC calibration (optional ``pyabc`` extra)
    - `heormodel.report`: plots and reproducibility scaffolding
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from heormodel.models import Intervention, ModelEngine, ModelFn, Outcomes
from heormodel.params import ParameterSet, mix_draws
from heormodel.run import RunResult, SeedManager, as_outcomes, run_psa

try:
    __version__ = _version("heormodel")
except PackageNotFoundError:  # pragma: no cover - not installed, e.g. running from source
    __version__ = "0.0.0"

__all__ = [
    "ModelEngine",
    "ModelFn",
    "Outcomes",
    "ParameterSet",
    "RunResult",
    "SeedManager",
    "Intervention",
    "__version__",
    "as_outcomes",
    "mix_draws",
    "run_psa",
]
