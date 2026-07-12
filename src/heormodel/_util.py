"""Small cross-cutting helpers shared across subpackages.

Internal conveniences with no public surface: a generator resolver, an
optional-dependency import guard, the shared-iteration-index check, and the
resample-to-iteration-index step the draw-mixing and calibration paths share.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def as_rng(seed: np.random.Generator | int | None) -> np.random.Generator:
    """A generator is returned unchanged; anything else seeds a fresh one."""
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def require_optional(module: str, *, feature: str, extra: str) -> Any:
    """Import an optional dependency ``module`` or raise with an install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            f"{feature} requires {module}; install it with uv pip install 'heormodel[{extra}]'."
        ) from err


def require_shared_index(
    index: pd.Index, iterations: pd.Index, label: str, *, detail: str = ""
) -> None:
    """Raise unless ``index`` matches the outcomes iteration index."""
    if not pd.Index(index).equals(pd.Index(iterations)):
        message = f"{label} index must equal the outcomes iteration index"
        raise ValueError(f"{message}; {detail}" if detail else f"{message}.")


def resample_to_iterations(frame: pd.DataFrame, picks: NDArray[np.intp]) -> pd.DataFrame:
    """Select rows ``picks`` and reindex as a fresh ``iteration`` range."""
    out = frame.iloc[picks].reset_index(drop=True)
    out.index = pd.RangeIndex(len(picks), name="iteration")
    return out
