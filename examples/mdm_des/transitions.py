"""Transition costs and utilities accrued over the sojourn that ends in each event.

The companion cost function does not pay the transition amounts once at the
event. It adds each (the onset cost and disutility, and the cost of dying) to
the annual flow of the sojourn that ends in the transition, then multiplies by
the discounted length of that sojourn, so a $2,000 cost of dying enters as
$2,000 per year over the whole final sojourn. These functions reconstruct that
arithmetic from the event history, which is why the amounts are not part of the
engine's per-year valuation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from heormodel.models import Outcomes


def transition_costs_and_utilities(
    events: pd.DataFrame, draws: pd.DataFrame, *, n_individuals: int, discount_rate: float
) -> pd.DataFrame:
    """Per-person cost and utility from the transition amounts, accrued over the sojourn.

    Input: the event history from ``evaluate(trace="events")``, the draw matrix
    (for the onset and death amounts per iteration), the population size, and
    the annual discount rate. Output: a DataFrame indexed by ``(strategy,
    iteration)`` with a ``cost`` and a ``qaly`` column to add to each strategy's
    outcomes.
    """
    ev = events.sort_values(["strategy", "iteration", "individual", "time"])
    start = ev.groupby(["strategy", "iteration", "individual"])["time"].shift(fill_value=0.0)
    stop = ev["time"].to_numpy()
    # Discounted sojourn length: the integral of the continuous discount factor
    # over the sojourn that this transition ends.
    discounted_years = (
        np.exp(-discount_rate * start.to_numpy()) - np.exp(-discount_rate * stop)
    ) / discount_rate
    row_params = draws.loc[ev["iteration"].to_numpy()]
    onset = (ev["from_state"].to_numpy() == "H") & (ev["to_state"].to_numpy() == "S1")
    death = ev["to_state"].to_numpy() == "D"
    cost_rate = onset * row_params["ic_HS1"].to_numpy() + death * row_params["ic_D"].to_numpy()
    utility_rate = -(onset * row_params["du_HS1"].to_numpy())
    contribution = pd.DataFrame(
        {
            "strategy": ev["strategy"].to_numpy(),
            "iteration": ev["iteration"].to_numpy(),
            "cost": cost_rate * discounted_years,
            "qaly": utility_rate * discounted_years,
        }
    )
    grouped = contribution.groupby(["strategy", "iteration"])[["cost", "qaly"]].sum()
    return grouped / n_individuals


def with_transition_costs_and_utilities(
    outcomes: Outcomes,
    events: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    n_individuals: int,
    discount_rate: float,
) -> Outcomes:
    """Add the sojourn-accrued transition costs and utilities to an outcomes panel.

    Input: the engine's `Outcomes`, the event history, the draw matrix, the
    population size, and the discount rate. Output: a new `Outcomes` with the
    transition amounts folded into cost and QALYs.
    """
    totals = transition_costs_and_utilities(
        events, draws, n_individuals=n_individuals, discount_rate=discount_rate
    )
    data = outcomes.data.add(totals.reindex(outcomes.data.index, fill_value=0.0), fill_value=0.0)
    return Outcomes(data, effect=outcomes.effect)


def costs_and_utilities_model(
    engine: object, *, n_individuals: int, discount_rate: float
) -> Callable[[pd.DataFrame], Outcomes]:
    """Wrap the engine as a ``draws -> Outcomes`` model that adds the transition amounts.

    ``run_psa`` drives the returned function over the draw matrix, so the sojourn
    accrual runs inside the framework's per-iteration seeding. Input: a
    continuous-clock engine, the population size, and the discount rate. Output:
    a function ``model(draws) -> Outcomes``.
    """

    def model(draws: pd.DataFrame) -> Outcomes:
        outcomes, events = engine.evaluate(draws, trace="events")  # type: ignore[attr-defined]
        return with_transition_costs_and_utilities(
            outcomes, events, draws, n_individuals=n_individuals, discount_rate=discount_rate
        )

    return model
