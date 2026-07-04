"""Cohort state-transition (Markov) engine (planned, not yet implemented).

Planned design (phase 2):

- Strategies defined by a transition-probability matrix builder
  ``build_matrix(params: pd.Series) -> ndarray`` plus per-state cost and
  utility payoffs, cycle length, horizon, discounting, and optional
  half-cycle correction.
- Vectorised evaluation across PSA iterations: one matrix-power sweep per
  iteration chunk, accumulating discounted costs and QALYs per strategy.
- Output: the standard `Outcomes` schema,
  optionally with per-state cost components. Nothing beyond that schema is
  exposed to the analysis layer.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from heval.models.outcomes import Outcomes


class MarkovCohortEngine:
    """Cohort state-transition model engine (stub).

    Raises:
        NotImplementedError: The engine body arrives in a later phase; only
            the output contract is fixed now.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "MarkovCohortEngine is scheduled for a later phase. Use a callable "
            "conforming to heval.models.ModelFn, or bring your own outputs via "
            "heval.run.as_outcomes, in the meantime."
        )

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate the cohort model on a parameter draw matrix (stub)."""
        raise NotImplementedError
