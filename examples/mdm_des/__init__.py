"""Continuous-time Sick-Sicker discrete-event replication (Lopez-Mendez and others, 2026).

The building blocks live in sibling modules: `mortality` loads the life table,
`model` defines the event-time and valuation functions and assembles the engine,
`transitions` reconstructs the sojourn-accrued transition costs and utilities,
`parameters` holds the model structure and distributions, and `outcomes` and
`plots` read the event history. `run` composes them into the full replication.
Import the pieces from this package, for example
``from mdm_des import build_engine, load_life_table``.
"""

from mdm_des.model import build_engine, make_event_times, make_state_reward_rates
from mdm_des.mortality import load_life_table
from mdm_des.outcomes import dwell_times, survival_and_prevalence
from mdm_des.parameters import base_case, interventions, parameter_set, states
from mdm_des.plots import plot_epidemiology
from mdm_des.transitions import (
    costs_and_utilities_model,
    transition_costs_and_utilities,
    with_transition_costs_and_utilities,
)

__all__ = [
    "base_case",
    "build_engine",
    "costs_and_utilities_model",
    "dwell_times",
    "load_life_table",
    "make_event_times",
    "make_state_reward_rates",
    "parameter_set",
    "plot_epidemiology",
    "states",
    "interventions",
    "survival_and_prevalence",
    "transition_costs_and_utilities",
    "with_transition_costs_and_utilities",
]
