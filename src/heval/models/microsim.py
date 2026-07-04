"""Individual-level (microsimulation) engines (planned, not yet implemented).

Planned design (phase 3):

- ``DiscreteTimeMicrosimEngine``: individual state-transition simulation on a
  fixed cycle grid with individual-level heterogeneity and history-dependent
  transition probabilities.
- ``ContinuousTimeMicrosimEngine``: competing time-to-event sampling from
  parametric hazards, no cycle grid.
- Both simulate a population per PSA iteration (with seeds spawned by
  `heval.run.SeedManager`), average within iteration, and emit the
  standard `Outcomes` schema. Internals are
  deliberately not shared with the cohort engine; the output contract is
  the only common surface.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from heval.models.outcomes import Outcomes


class DiscreteTimeMicrosimEngine:
    """Discrete-time individual-level simulation engine (stub).

    Raises:
        NotImplementedError: The engine body arrives in a later phase.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("DiscreteTimeMicrosimEngine is scheduled for a later phase.")

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate the microsimulation on a parameter draw matrix (stub)."""
        raise NotImplementedError


class ContinuousTimeMicrosimEngine:
    """Continuous-time individual-level simulation engine (stub).

    Raises:
        NotImplementedError: The engine body arrives in a later phase.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("ContinuousTimeMicrosimEngine is scheduled for a later phase.")

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate the microsimulation on a parameter draw matrix (stub)."""
        raise NotImplementedError
