"""heval: health economic evaluation in Python.

One parameter draw matrix flows through swappable model engines into a
shared analysis layer. Engines differ internally but share a contract on
their outputs, the :class:`~heval.models.Outcomes` schema indexed by
``(strategy, iteration)``, which makes cost-effectiveness and
value-of-information analysis engine-agnostic. Outputs from any external
model enter the same pipeline via :func:`~heval.run.as_outcomes`.

Subpackages:
    - :mod:`heval.params`: distributions and correlated PSA sampling
    - :mod:`heval.models`: engines behind the output contract
    - :mod:`heval.run`: seeds, run loop, bring-your-own-outputs ingestion
    - :mod:`heval.cea`: incremental analysis, frontier, NMB/NHB, CEAC/CEAF
    - :mod:`heval.voi`: EVPI, EVPPI, EVSI
    - :mod:`heval.calibrate`: ABC calibration (optional ``pyabc`` extra)
    - :mod:`heval.report`: plots and reproducibility scaffolding
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from heval.models import ModelEngine, ModelFn, Outcomes
from heval.params import ParameterSet
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
    "run_psa",
]
