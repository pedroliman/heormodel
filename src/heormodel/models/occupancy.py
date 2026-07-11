"""State occupancy over time from an individual-level event history."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from heormodel.models.outcomes import ITERATION_LEVEL, STRATEGY_LEVEL

_EVENT_COLUMNS = (STRATEGY_LEVEL, ITERATION_LEVEL, "individual", "time", "from_state", "to_state")


def state_occupancy(
    events: pd.DataFrame,
    *,
    states: Sequence[str],
    initial_state: str,
    n_individuals: int,
    times: ArrayLike,
) -> pd.DataFrame:
    """Proportion of individuals in each state at each time.

    Counts, for every requested time, how many individuals occupy each state:
    everyone starts in ``initial_state``, each event row moves one individual
    at its ``time``, and an event at exactly a requested time counts as having
    happened. Individuals appear in the log only when they move, so the
    initial state and the population size are explicit arguments rather than
    read from the log.

    Survival is one minus the dead-state column, and prevalence among the
    alive is the summed disease-state columns divided by that survival, both
    one-line derivations of this table.

    Args:
        events: Event history with columns ``strategy``, ``iteration``,
            ``individual``, ``time``, ``from_state``, ``to_state``, as
            returned by ``evaluate(draws, trace="events")``.
        states: Every state label, in the order the columns should take.
        initial_state: State every individual occupies at time zero.
        n_individuals: Number of simulated individuals per strategy and
            iteration.
        times: Times at which to evaluate occupancy.

    Returns:
        DataFrame indexed by ``(strategy, iteration, time)`` with one
        proportion column per state; rows sum to 1.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import state_occupancy
        >>> events = pd.DataFrame({
        ...     "strategy": "care", "iteration": 0, "individual": [0, 0, 1],
        ...     "time": [1.0, 3.0, 2.0], "from_state": ["H", "S", "H"],
        ...     "to_state": ["S", "D", "D"]})
        >>> occ = state_occupancy(events, states=("H", "S", "D"),
        ...     initial_state="H", n_individuals=4, times=[0.0, 2.5])
        >>> float(occ.loc[("care", 0, 2.5), "H"])
        0.5
    """
    missing = [c for c in _EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events is missing columns {missing}.")
    state_list = list(states)
    if initial_state not in state_list:
        raise ValueError(f"initial_state {initial_state!r} is not in states.")
    known = events["from_state"].isin(state_list) & events["to_state"].isin(state_list)
    if not known.all():
        raise ValueError("events contains states not listed in states.")
    if n_individuals <= 0:
        raise ValueError("n_individuals must be positive.")
    grid = np.atleast_1d(np.asarray(times, dtype=np.float64))
    frames = []
    for (strategy, iteration), group in events.groupby(
        [STRATEGY_LEVEL, ITERATION_LEVEL], sort=False
    ):
        counts = np.zeros((len(grid), len(state_list)), dtype=np.float64)
        for j, state in enumerate(state_list):
            entries = np.sort(group.loc[group["to_state"] == state, "time"].to_numpy())
            exits = np.sort(group.loc[group["from_state"] == state, "time"].to_numpy())
            start = float(n_individuals) if state == initial_state else 0.0
            counts[:, j] = (
                start
                + np.searchsorted(entries, grid, side="right")
                - np.searchsorted(exits, grid, side="right")
            )
        index = pd.MultiIndex.from_arrays(
            [np.repeat(strategy, len(grid)), np.repeat(iteration, len(grid)), grid],
            names=[STRATEGY_LEVEL, ITERATION_LEVEL, "time"],
        )
        frames.append(pd.DataFrame(counts / n_individuals, index=index, columns=state_list))
    if not frames:
        raise ValueError("events is empty.")
    return pd.concat(frames)
