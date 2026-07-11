"""The continuous-time Sick-Sicker model: its event-time and valuation functions.

Each factory binds the model configuration (the life table, the starting age,
the state labels) and returns the plain function `MicrosimModel` calls. This is
how configuration is passed in rather than read from module-level constants.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from heormodel.models import LifeTable, MicrosimModel
from heormodel.run import SeedManager

EventTimes = Callable[..., np.ndarray]
Valuation = Callable[..., tuple[np.ndarray, np.ndarray]]


def make_event_times(
    life_table: LifeTable, age_start: float, states: tuple[str, ...]
) -> EventTimes:
    """Build the continuous-clock event-time sampler.

    Input: the background-mortality `LifeTable`, the starting age, and the state
    labels. Output: a function ``fn(params, state, attrs, rng)`` returning an
    ``(n, n_states)`` array of the sampled time to each competing transition
    (``inf`` where a transition cannot occur), the ``event_times`` argument
    `MicrosimModel` expects on the continuous clock. Competing times are redrawn
    at every state entry, so the Weibull progression draw needs no truncation
    and each death time reflects the individual's current age and state.
    """
    index = {label: i for i, label in enumerate(states)}
    n_states = len(states)
    healthy, sick, sicker, dead = index["H"], index["S1"], index["S2"], index["D"]

    def event_times(
        params: pd.Series, state: np.ndarray, attrs: pd.DataFrame, rng: np.random.Generator
    ) -> np.ndarray:
        n = len(state)
        times = np.full((n, n_states), np.inf)
        age = age_start + attrs["time"].to_numpy()
        in_h = state == healthy
        if in_h.any():
            times[in_h, sick] = rng.exponential(1.0 / params["r_HS1"], int(in_h.sum()))
            times[in_h, dead] = life_table.sample_time_to_death(rng, age[in_h])
        in_s1 = state == sick
        if in_s1.any():
            times[in_s1, healthy] = rng.exponential(1.0 / params["r_S1H"], int(in_s1.sum()))
            # Weibull in proportional-hazards form: treatment B multiplies the
            # scale; sampling uses scale ** (-1/shape), the accelerated-failure
            # -time scale.
            scale_ph = params["r_S1S2_scale"] * (params["hr_S1S2_trtB"] if params["trtB"] else 1.0)
            aft_scale = scale_ph ** (-1.0 / params["r_S1S2_shape"])
            times[in_s1, sicker] = aft_scale * rng.weibull(params["r_S1S2_shape"], int(in_s1.sum()))
            times[in_s1, dead] = life_table.sample_time_to_death(
                rng, age[in_s1], hazard_ratio=params["hr_S1"]
            )
        in_s2 = state == sicker
        if in_s2.any():
            times[in_s2, dead] = life_table.sample_time_to_death(
                rng, age[in_s2], hazard_ratio=params["hr_S2"]
            )
        return times

    return event_times


def make_state_costs_and_utilities(
    states: tuple[str, ...], treatment_a_utility_gain: float = 0.20
) -> Valuation:
    """Build the per-state cost and utility function.

    Input: the state labels and the fixed utility gain treatment A adds in Sick.
    Output: a function ``fn(params, state, attrs) -> (cost_rate, utility_rate)``,
    the ``state_costs_and_utilities`` argument `MicrosimModel` expects: the
    annual cost and utility of each individual's current state and strategy.
    Treatment costs apply in both disease states because Sick and Sicker are
    indistinguishable in practice.
    """
    index = {label: i for i, label in enumerate(states)}
    healthy, sick, sicker = index["H"], index["S1"], index["S2"]

    def state_costs_and_utilities(
        params: pd.Series, state: np.ndarray, attrs: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(state)
        cost = np.zeros(n)
        utility = np.zeros(n)
        on_a, on_b = bool(params["trtA"]), bool(params["trtB"])
        treatment_cost = on_a * params["c_trtA"] + on_b * params["c_trtB"]
        in_h = state == healthy
        cost[in_h], utility[in_h] = params["c_H"], params["u_H"]
        in_s1 = state == sick
        cost[in_s1] = params["c_S1"] + treatment_cost
        utility[in_s1] = (params["u_S1"] + treatment_a_utility_gain) if on_a else params["u_S1"]
        in_s2 = state == sicker
        cost[in_s2] = params["c_S2"] + treatment_cost
        utility[in_s2] = params["u_S2"]
        return cost, utility

    return state_costs_and_utilities


def build_engine(
    *,
    life_table: LifeTable,
    states: tuple[str, ...],
    strategies: Mapping[str, Mapping[str, Any]],
    age_start: float,
    horizon: float,
    discount_rate: float,
    population: int,
    seed_manager: SeedManager,
    treatment_a_utility_gain: float = 0.20,
) -> MicrosimModel:
    """Assemble the continuous-clock `MicrosimModel` from the model functions.

    Input: the life table, the model structure and horizon, the discount rate,
    the population size, and a seed manager. Output: a configured
    `MicrosimModel` ready to ``evaluate`` a draw matrix.
    """
    return MicrosimModel(
        states=states,
        clock="continuous",
        event_times=make_event_times(life_table, age_start, states),
        state_costs_and_utilities=make_state_costs_and_utilities(
            states, treatment_a_utility_gain
        ),
        population=population,
        strategies=strategies,
        horizon=horizon,
        discount_rate=discount_rate,
        seed_manager=seed_manager,
    )
