"""State occupancy over time from an individual-level event history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from heormodel.models.outcomes import INTERVENTION_LEVEL, ITERATION_LEVEL

_EVENT_COLUMNS = (
    INTERVENTION_LEVEL, ITERATION_LEVEL, "individual", "time", "from_state", "to_state"
)


def state_occupancy(
    events: pd.DataFrame,
    *,
    states: Sequence[str],
    initial_state: str,
    n_individuals: int,
    times: ArrayLike,
    interventions: Sequence[str],
    iterations: Sequence[Any],
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
        events: Event history with columns ``intervention``, ``iteration``,
            ``individual``, ``time``, ``from_state``, ``to_state``, as
            returned by ``evaluate(draws, trace="events")``.
        states: Every state label, in the order the columns should take.
        initial_state: State every individual occupies at time zero.
        n_individuals: Number of simulated individuals per intervention and
            iteration.
        times: Times at which to evaluate occupancy.
        interventions: Every intervention actually run, in output order.
            An intervention in which nobody moves during a whole iteration
            leaves no rows in ``events`` at all, so it cannot be recovered
            from the log; naming it here is what keeps it in the result.
        iterations: Every iteration actually run, e.g. the index of the
            parameter draws passed to ``run_psa``. An iteration in which
            nobody moves leaves no rows in ``events`` either; naming it
            here is what keeps it in the result, entirely in
            ``initial_state`` at every requested time.

    Returns:
        DataFrame indexed by ``(intervention, iteration, time)`` with one
        proportion column per state; rows sum to 1. Covers every pair
        formed by crossing ``interventions`` with ``iterations``, plus any
        additional pair already present in ``events``. A pair with no
        event rows of its own occupies ``initial_state`` at every
        requested time.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import state_occupancy
        >>> events = pd.DataFrame({
        ...     "intervention": "care", "iteration": 0, "individual": [0, 0, 1],
        ...     "time": [1.0, 3.0, 2.0], "from_state": ["H", "S", "H"],
        ...     "to_state": ["S", "D", "D"]})
        >>> occ = state_occupancy(events, states=("H", "S", "D"),
        ...     initial_state="H", n_individuals=4, times=[0.0, 2.5],
        ...     interventions=["care"], iterations=[0, 1])
        >>> float(occ.loc[("care", 0, 2.5), "H"])
        0.5

        Iteration 1 has no rows in ``events``, since nobody moved under it,
        so it occupies ``initial_state`` at every requested time:

        >>> occ.loc[("care", 1, 2.5)].tolist()
        [1.0, 0.0, 0.0]
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

    def block(intervention: str, iteration: Any, proportions: np.ndarray) -> pd.DataFrame:
        index = pd.MultiIndex.from_arrays(
            [np.repeat(intervention, len(grid)), np.repeat(iteration, len(grid)), grid],
            names=[INTERVENTION_LEVEL, ITERATION_LEVEL, "time"],
        )
        return pd.DataFrame(proportions, index=index, columns=state_list)

    frames = []
    seen: dict[tuple[str, Any], None] = {}
    for (intervention, iteration), group in events.groupby(
        [INTERVENTION_LEVEL, ITERATION_LEVEL], sort=False
    ):
        seen[(intervention, iteration)] = None
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
        frames.append(block(intervention, iteration, counts / n_individuals))

    # Fill in every (intervention, iteration) pair the caller ran but that has
    # no event rows: nobody moved, so everyone stays in initial_state throughout.
    initial_proportions = np.zeros((len(grid), len(state_list)), dtype=np.float64)
    initial_proportions[:, state_list.index(initial_state)] = 1.0
    for intervention in dict.fromkeys(interventions):
        for iteration in dict.fromkeys(iterations):
            if (intervention, iteration) not in seen:
                frames.append(block(intervention, iteration, initial_proportions))

    if not frames:
        raise ValueError(
            "No (intervention, iteration) pairs to report: events is empty "
            "and interventions or iterations is empty."
        )
    return pd.concat(frames)
