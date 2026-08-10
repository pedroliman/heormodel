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
    interventions: Sequence[str] | None = None,
    iterations: Sequence[Any] | None = None,
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
        interventions: Every intervention that was run, in output order.
            An intervention in which nobody moves during a whole iteration
            leaves no rows in ``events`` at all, so pass this explicitly
            to keep such an intervention in the result; otherwise it
            cannot be recovered from the log. Only takes effect together
            with ``iterations``, or on its own when ``events`` already
            contains at least one row for every relevant iteration under
            some other intervention.
        iterations: Every iteration that was run, e.g. the index of the
            parameter draws passed to ``run_psa``. An iteration in which
            nobody moves leaves no rows in ``events`` and is otherwise
            dropped from the result without warning; pass this explicitly
            to keep it, entirely in ``initial_state`` at every requested
            time. Only takes effect together with ``interventions``, or on
            its own when ``events`` already contains at least one row for
            every relevant intervention under some other iteration.

    Returns:
        DataFrame indexed by ``(intervention, iteration, time)`` with one
        proportion column per state; rows sum to 1. Always includes every
        ``(intervention, iteration)`` pair found in ``events``. When
        ``interventions`` or ``iterations`` is given, also includes every
        pair formed by crossing it with the other axis, which defaults to
        what ``events`` already contains; a pair added this way, with no
        event rows of its own, occupies ``initial_state`` at every
        requested time.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import state_occupancy
        >>> events = pd.DataFrame({
        ...     "intervention": "care", "iteration": 0, "individual": [0, 0, 1],
        ...     "time": [1.0, 3.0, 2.0], "from_state": ["H", "S", "H"],
        ...     "to_state": ["S", "D", "D"]})
        >>> occ = state_occupancy(events, states=("H", "S", "D"),
        ...     initial_state="H", n_individuals=4, times=[0.0, 2.5])
        >>> float(occ.loc[("care", 0, 2.5), "H"])
        0.5

        Iteration 1 has no rows in ``events``, so it is absent from ``occ``
        above. Passing ``iterations`` adds it, entirely in
        ``initial_state``:

        >>> occ = state_occupancy(events, states=("H", "S", "D"),
        ...     initial_state="H", n_individuals=4, times=[0.0, 2.5],
        ...     iterations=[0, 1])
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

    # Fill in every requested (intervention, iteration) pair with no event
    # rows: nobody moved, so everyone is still in initial_state throughout.
    # Only runs when the caller opts in via interventions or iterations;
    # otherwise the result is exactly the pairs found in events, as before.
    if interventions is not None or iterations is not None:
        full_interventions = (
            list(dict.fromkeys(interventions))
            if interventions is not None
            else list(dict.fromkeys(pair[0] for pair in seen))
        )
        full_iterations = (
            list(dict.fromkeys(iterations))
            if iterations is not None
            else list(dict.fromkeys(pair[1] for pair in seen))
        )
        initial_proportions = np.zeros((len(grid), len(state_list)), dtype=np.float64)
        initial_proportions[:, state_list.index(initial_state)] = 1.0
        for intervention in full_interventions:
            for iteration in full_iterations:
                if (intervention, iteration) not in seen:
                    frames.append(block(intervention, iteration, initial_proportions))

    if not frames:
        raise ValueError("events is empty.")
    return pd.concat(frames)
