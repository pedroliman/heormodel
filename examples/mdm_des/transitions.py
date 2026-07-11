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

import numpy as np
import pandas as pd

from heormodel.models import EngineResult, Outcomes, StochasticEngine
from heormodel.run import SeedManager


def transition_costs_and_utilities(
    events: pd.DataFrame, draws: pd.DataFrame, *, n_individuals: int, discount_rate: float
) -> pd.DataFrame:
    """Per-person cost and utility from the transition amounts, accrued over the sojourn.

    Input: the event history from ``run_psa(engine, draws, collect="events")``,
    the draw matrix (for the onset and death amounts per iteration), the population size, and
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


class _WithTransitionAccrual:
    """A stochastic engine that folds the sojourn-accrued transition amounts in.

    It wraps a continuous-clock engine, collects the event history one batch at a
    time, adds the transition amounts, and returns outcomes only, so the runner
    never gathers the whole event history. Because it exposes ``evaluate_streamed``
    it is a `heormodel.models.StochasticEngine`: `run_psa` hands it the
    per-iteration streams, so the sojourn accrual runs under the run's seed.
    """

    def __init__(self, engine: StochasticEngine, *, n_individuals: int, discount_rate: float):
        self._engine = engine
        self._n_individuals = n_individuals
        self._discount_rate = discount_rate

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        return self.evaluate_streamed(draws, streams=SeedManager(0)).outcomes

    def evaluate_streamed(
        self, draws: pd.DataFrame, *, streams: SeedManager, collect: str | None = None
    ) -> EngineResult:
        result = self._engine.evaluate_streamed(draws, streams=streams, collect="events")
        outcomes = with_transition_costs_and_utilities(
            result.outcomes, result.events, draws,
            n_individuals=self._n_individuals, discount_rate=self._discount_rate,
        )
        return EngineResult(outcomes, events=result.events if collect == "events" else None)


def costs_and_utilities_model(
    engine: StochasticEngine, *, n_individuals: int, discount_rate: float
) -> _WithTransitionAccrual:
    """Wrap the engine so its outcomes carry the sojourn-accrued transition amounts.

    ``run_psa`` drives the returned engine over the draw matrix, so the sojourn
    accrual runs inside the framework's per-iteration seeding. Input: a
    continuous-clock engine, the population size, and the discount rate. Output:
    a stochastic engine whose ``evaluate`` folds the transition amounts in.
    """
    return _WithTransitionAccrual(
        engine, n_individuals=n_individuals, discount_rate=discount_rate
    )
