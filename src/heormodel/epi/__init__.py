"""Epidemiological outcomes from an event history (`heormodel.epi`).

An individual-level engine can return its event history, one row per state
change (`MicrosimModel.evaluate` with ``trace="events"``). This package turns
that log into the epidemiological outcomes a state-transition analysis reports
alongside costs and effects: state occupancy over time, survival probabilities,
and disease prevalence among the alive.
"""

from heormodel.epi.occupancy import prevalence, state_occupancy, survival

__all__ = [
    "prevalence",
    "state_occupancy",
    "survival",
]
