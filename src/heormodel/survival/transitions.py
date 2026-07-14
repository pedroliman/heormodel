"""Turn cause-specific hazards into a per-cycle transition array.

`to_transition_matrix` builds the age-varying transition array
`heormodel.models.MarkovModel` accepts from a set of cause-specific survival
curves. Each cycle's transition-intensity matrix is assembled from the
cumulative-hazard increment of every cause over that cycle, and its matrix
exponential gives a transition-probability matrix whose rows sum to one. For a
single decrement this reproduces ``1 - S(k+1) / S(k)`` exactly; for competing
causes it is the standard continuous-time approximation that holds the relative
cause mix fixed within a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from heormodel.survival.curve import SurvivalCurve


def to_transition_matrix(
    causes: Mapping[tuple[str, str], SurvivalCurve],
    states: Sequence[str],
    n_cycles: int,
    cycle_length: float = 1.0,
) -> NDArray[np.float64]:
    """Build a per-cycle transition array from cause-specific survival curves.

    Args:
        causes: Map from a ``(from_state, to_state)`` pair to the cause-specific
            `SurvivalCurve` for that transition. A state with no outgoing cause
            is absorbing.
        states: State labels in the order that fixes the array's axes.
        n_cycles: Number of cycles in the horizon.
        cycle_length: Years per cycle.

    Returns:
        An array of shape ``(n_cycles, n_states, n_states)``; entry ``[k, i, j]``
        is the probability of moving from state ``i`` to state ``j`` during cycle
        ``k``. Each row sums to one.

    Example:
        >>> from heormodel.survival import weibull, to_transition_matrix
        >>> causes = {("alive", "dead"): weibull(1.2, 6.0)}
        >>> transition = to_transition_matrix(causes, ("alive", "dead"), 5)
        >>> [round(float(p), 5) for p in transition[:, 0, 1]]
        [0.10994, 0.14025, 0.15439, 0.16428, 0.17201]
    """
    if n_cycles < 1:
        raise ValueError("n_cycles must be at least one.")
    order = tuple(states)
    index = {state: position for position, state in enumerate(order)}
    n_states = len(order)
    edges = np.arange(n_cycles + 1, dtype=float) * cycle_length

    increments: dict[tuple[int, int], NDArray[np.float64]] = {}
    for (source, target), curve in causes.items():
        if source not in index or target not in index:
            raise ValueError(f"transition {(source, target)} names a state not in {order}.")
        cumulative = np.asarray(curve.cumulative_hazard(edges), dtype=float)
        increments[(index[source], index[target])] = np.diff(cumulative)

    transition = np.empty((n_cycles, n_states, n_states), dtype=np.float64)
    for cycle in range(n_cycles):
        intensity = np.zeros((n_states, n_states), dtype=np.float64)
        for (row, column), increment in increments.items():
            rate = increment[cycle]
            intensity[row, column] += rate
            intensity[row, row] -= rate
        transition[cycle] = expm(intensity)
    return transition
