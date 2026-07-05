"""heval: health economic evaluation in Python.

One parameter draw matrix flows through swappable model engines into a
shared analysis layer. Engines differ internally but share a contract on
their outputs, the `Outcomes` schema indexed by
``(strategy, iteration)``, which makes cost-effectiveness and
value-of-information analysis engine-agnostic. Outputs from any external
model enter the same pipeline via `as_outcomes`.

Subpackages:
    - `heval.params`: distributions and correlated PSA sampling
    - `heval.models`: engines behind the output contract
    - `heval.run`: seeds, run loop, bring-your-own-outputs ingestion
    - `heval.cea`: incremental analysis, frontier, NMB/NHB, CEAC/CEAF
    - `heval.dsa`: one-way, one-at-a-time, and grid deterministic sensitivity designs
    - `heval.voi`: EVPI, EVPPI, EVSI
    - `heval.calibrate`: ABC calibration (optional ``pyabc`` extra)
    - `heval.report`: plots and reproducibility scaffolding
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from heval.models import ModelEngine, ModelFn, Outcomes
from heval.params import ParameterSet, mix_draws
from heval.run import SeedManager, as_outcomes, run_psa

try:
    __version__ = _version("heval")
except PackageNotFoundError:  # pragma: no cover - not installed, e.g. running from source
    __version__ = "0.0.0"

__all__ = [
    "ModelEngine",
    "ModelFn",
    "Outcomes",
    "ParameterSet",
    "SeedManager",
    "__version__",
    "as_outcomes",
    "mix_draws",
    "run_psa",
]
