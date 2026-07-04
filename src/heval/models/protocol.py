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
from typing import Protocol, TypeAlias, runtime_checkable

import pandas as pd

from heval.models.outcomes import Outcomes


@runtime_checkable
class ModelEngine(Protocol):
    """Anything that turns parameter draws into standardized outcomes.

    Implementations must satisfy two invariants:

    1. The returned `Outcomes` iteration index equals ``draws.index``
       (same values, same order); this preserves the parameter/outcome
       linkage required by EVPPI and EVSI.
    2. Every strategy is evaluated on every iteration (a balanced panel);
       the `Outcomes` constructor enforces this.

    Example:
        >>> import pandas as pd
        >>> from heval.models import ModelEngine, Outcomes
        >>> class TwoStrategyModel:
        ...     def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        ...         costs = pd.DataFrame({"A": draws["c"], "B": draws["c"] * 2})
        ...         effects = pd.DataFrame({"A": draws["e"], "B": draws["e"] * 1.5})
        ...         return Outcomes.from_wide(costs, effects)
        >>> isinstance(TwoStrategyModel(), ModelEngine)
        True
    """

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate all strategies for every row of the draw matrix."""
        ...


#: A plain function is also an acceptable model: ``draws -> Outcomes``.
ModelFn: TypeAlias = Callable[[pd.DataFrame], Outcomes]
