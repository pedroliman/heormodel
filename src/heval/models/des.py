"""Discrete-event simulation engine (planned, not yet implemented).

Planned design (phase 4): a thin wrapper around SimPy rather than a new DES
kernel. ``heval`` adds only what SimPy does not provide for health economic
models:

- patient trajectory recording (event log per entity),
- resource-constraint helpers (queues, capacities) mapped to costs,
- continuous cost and utility accrual between events with discounting,
- per-iteration seeding via `heval.run.SeedManager`, and
- aggregation of entity-level accruals into the standard
  `Outcomes` schema.

The SimPy environment and process functions remain the user's own code;
this engine never hides or re-implements the simulation loop.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from heval.models.outcomes import Outcomes


class DESEngine:
    """SimPy-wrapping discrete-event simulation engine (stub).

    Raises:
        NotImplementedError: The engine body arrives in a later phase.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "DESEngine is scheduled for a later phase; it will wrap SimPy with "
            "trajectory, resource-constraint, and accrual helpers only."
        )

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate the DES model on a parameter draw matrix (stub)."""
        raise NotImplementedError
