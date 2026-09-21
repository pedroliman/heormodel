"""Turning survival curves into per-cycle transition probabilities.

`to_transition_matrix` is the bridge from a `heormodel.survival.SurvivalCurve`
to `heormodel.models.MarkovModel`'s per-cycle transition array. One curve
gives a two-state alive-and-dead model; several curves for the same alive
state give a competing-risks model, one absorbing state per cause.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from heormodel.survival.curve import SurvivalCurve

_PROB_TOL = 1e-8


def to_transition_matrix(
    curves: SurvivalCurve | Sequence[SurvivalCurve],
    *,
    n_cycles: int,
    cycle_length: float = 1.0,
) -> NDArray[np.float64]:
    """Per-cycle transition array from one or more cause-specific survival curves.

    With one curve, builds a two-state (alive, dead) array whose per-cycle
    death probability is `1 - survival(t + cycle_length) / survival(t)`,
    ready for `heormodel.models.CohortSpec.transition`.

    With several curves, treats each as a cause-specific hazard out of one
    shared alive state and builds a `(1 + len(curves))`-state array: state 0
    is alive, state `1 + k` is dead from cause `k`, absorbing. Within a cycle,
    the total event probability is split across causes in proportion to each
    cause's cumulative hazard increment over that cycle, `1 - exp(-total
    hazard increment)` overall. This construction is exact when every curve's
    hazard is constant within a cycle (an all-exponential competing-risks
    model), the standard result that the chance of cause `k` occurring first
    is proportional to its share of the total hazard.

    Args:
        curves: One `SurvivalCurve`, or a sequence of curves for competing
            risks out of the same starting state.
        n_cycles: Number of cycles in the model horizon.
        cycle_length: Years per cycle.

    Returns:
        Transition array, shape `(n_cycles, n_states, n_states)`, each row
        summing to 1.

    Example:
        >>> from heormodel.survival import to_transition_matrix, weibull
        >>> curve = weibull(shape=1.2, scale=6.0)
        >>> transition = to_transition_matrix(curve, n_cycles=5)
        >>> [round(float(p), 5) for p in transition[:, 0, 1]]
        [0.10994, 0.14025, 0.15439, 0.16428, 0.17201]
    """
    if n_cycles < 1:
        raise ValueError("n_cycles must be at least one.")
    curve_list = [curves] if isinstance(curves, SurvivalCurve) else list(curves)
    if not curve_list:
        raise ValueError("to_transition_matrix requires at least one survival curve.")

    times = np.arange(n_cycles + 1, dtype=np.float64) * cycle_length
    hazards = np.stack([curve.cumulative_hazard(times) for curve in curve_list], axis=0)
    increment = hazards[:, 1:] - hazards[:, :-1]  # shape (n_causes, n_cycles)
    if np.any(increment < -_PROB_TOL):
        raise ValueError(
            "A curve's cumulative hazard is not non-decreasing; check the curve's "
            "cumulative_hazard function."
        )
    increment = np.clip(increment, 0.0, None)
    total_increment = increment.sum(axis=0)
    event_probability = 1.0 - np.exp(-total_increment)
    share = np.divide(
        increment, total_increment, out=np.zeros_like(increment), where=total_increment > 0
    )

    n_causes = len(curve_list)
    n_states = n_causes + 1
    transition = np.zeros((n_cycles, n_states, n_states), dtype=np.float64)
    transition[:, 0, 0] = 1.0 - event_probability
    for cause in range(n_causes):
        transition[:, 0, 1 + cause] = event_probability * share[cause]
        transition[:, 1 + cause, 1 + cause] = 1.0  # dead from this cause is absorbing
    return transition
