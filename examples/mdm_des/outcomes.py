"""Epidemiological outcomes from the event history: survival, prevalence, dwell times.

These derive from `heormodel.models.state_occupancy`, which turns the event
history into the proportion of the population in each state over time. Survival
is one minus the dead-state column and prevalence among the alive is the summed
disease-state columns over survival, so neither needs its own maintained
function.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from heormodel.models import state_occupancy


def survival_and_prevalence(
    events: pd.DataFrame,
    *,
    states: Sequence[str],
    strategies: Sequence[str],
    initial_state: str,
    dead_state: str,
    disease_states: Sequence[str],
    n_individuals: int,
    horizon: float,
    n_points: int = 151,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Survival and disease-prevalence curves by strategy over a time grid.

    Input: the event history, the model structure, the population size, and the
    horizon. Output: a tuple ``(survival, prevalence)`` of DataFrames indexed by
    time (years since the start) with one column per strategy, in the given
    strategy order.
    """
    grid = np.linspace(0.0, horizon, n_points)
    occupancy = state_occupancy(
        events, states=states, initial_state=initial_state,
        n_individuals=n_individuals, times=grid,
    ).droplevel("iteration")
    alive = 1.0 - occupancy[dead_state]
    survival = alive.unstack("strategy")[list(strategies)]
    prevalence = (
        occupancy[list(disease_states)].sum(axis=1) / alive
    ).unstack("strategy")[list(strategies)]
    return survival, prevalence


def dwell_times(events: pd.DataFrame) -> pd.DataFrame:
    """Mean completed sojourn per state and strategy.

    Input: the event history. Output: a DataFrame of the mean dwell years by
    strategy and originating state, each completed sojourn being the gap between
    consecutive event times within an individual.
    """
    ev = events.sort_values(["strategy", "individual", "time"])
    entered = ev.groupby(["strategy", "individual"])["time"].shift(fill_value=0.0)
    ev = ev.assign(dwell=ev["time"] - entered)
    return ev.groupby(["strategy", "from_state"])["dwell"].mean().unstack("from_state")
