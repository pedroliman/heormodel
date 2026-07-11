"""The model engine protocol: a contract on outputs, not internals.

Engines differ radically inside (cohort matrix algebra, individual-level
simulation, discrete-event simulation) and deliberately do **not** share an
implementation API. What they share is this: given a parameter draw matrix,
an engine returns an `Outcomes` object whose
iteration index matches the draws it was given. Analysis code depends only
on that contract and never reaches into engine internals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

import pandas as pd

from heormodel.models.outcomes import Outcomes

if TYPE_CHECKING:
    from heormodel.run.seeds import SeedManager


@runtime_checkable
class ModelEngine(Protocol):
    """Anything that turns parameter draws into standardized outcomes.

    Implementations must satisfy two invariants:

    1. The returned `Outcomes` iteration index equals ``draws.index``
       (same values, same order); this preserves the parameter/outcome
       linkage required by EVPPI and EVSI.
    2. Every intervention is evaluated on every iteration (a balanced panel);
       the `Outcomes` constructor enforces this.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import ModelEngine, Outcomes
        >>> class TwoInterventionModel:
        ...     def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        ...         costs = pd.DataFrame({"A": draws["c"], "B": draws["c"] * 2})
        ...         effects = pd.DataFrame({"A": draws["e"], "B": draws["e"] * 1.5})
        ...         return Outcomes.from_wide(costs, effects)
        >>> isinstance(TwoInterventionModel(), ModelEngine)
        True
    """

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate all interventions for every row of the draw matrix."""
        ...


#: A plain function is also an acceptable model: ``draws -> Outcomes``.
ModelFn: TypeAlias = Callable[[pd.DataFrame], Outcomes]


@dataclass(frozen=True)
class EngineResult:
    """What a stochastic engine returns for one batch of draws.

    The ``outcomes`` panel is always present; ``events`` and ``individuals`` hold
    the optional log channels and stay ``None`` unless the runner asked for them
    through ``collect``.

    Args:
        outcomes: The `Outcomes` panel for this batch.
        events: The state-change (or resource) history, or ``None``.
        individuals: Per-individual accruals, or ``None``.
    """

    outcomes: Outcomes
    events: pd.DataFrame | None = None
    individuals: pd.DataFrame | None = None


@runtime_checkable
class StochasticEngine(Protocol):
    """An engine whose randomness the runner drives through per-iteration streams.

    Deterministic engines satisfy `ModelEngine` alone. A stochastic engine
    (individual-level or discrete-event) additionally accepts a `SeedManager` of
    per-iteration streams and an optional ``collect`` channel, returning an
    `EngineResult`. `run_psa` builds the streams from its ``seed`` argument, so
    the engine holds no seed of its own and reruns under a new seed without
    reconstruction. Because streams are keyed by iteration index, the numbers do
    not depend on how a run is split across workers.
    """

    def evaluate_streamed(
        self, draws: pd.DataFrame, *, streams: SeedManager, collect: str | None = None
    ) -> EngineResult:
        """Evaluate a batch under the given streams, collecting ``collect`` logs."""
        ...
