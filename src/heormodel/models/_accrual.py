"""Cost and utility accrual, discounting, and aggregation to `Outcomes`.

Internal helpers shared by the microsimulation engines (and, later, the
discrete-event engine). They are the one piece of implementation the engines
hold in common: everything else about how each engine simulates is its own.

Two accrual styles live here. `accrue` discounts per-cycle amounts on a shared
cycle grid, for the discrete-time engine. `integrate_flow` integrates a
continuous flow between event times, for the continuous-time engine. Both call
`discount_factor`, which offers a per-cycle and a continuous variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from heormodel.models.outcomes import INTERVENTION_LEVEL, ITERATION_LEVEL


def discount_factor(
    t: float | NDArray[np.float64], rate: float, *, continuous: bool = False
) -> NDArray[np.float64]:
    """Discount factor at time ``t`` (years) for an annual ``rate``.

    The per-cycle variant (default) uses ``(1 + rate) ** -t``, the convention
    for cycle-based cohort and microsimulation models. The continuous variant
    uses ``exp(-rate * t)``, matching continuous-time accrual.

    Args:
        t: Time in years, scalar or array.
        rate: Annual discount rate (0.03 for 3%).
        continuous: Use continuous ``exp(-rate * t)`` instead of per-cycle.

    Example:
        >>> from heormodel.models._accrual import discount_factor
        >>> round(float(discount_factor(1.0, 0.03)), 6)
        0.970874
        >>> round(float(discount_factor(1.0, 0.03, continuous=True)), 6)
        0.970446
    """
    t = np.asarray(t, dtype=np.float64)
    if continuous:
        return np.exp(-rate * t)
    return (1.0 + rate) ** (-t)


def accrue(
    amounts: NDArray[np.float64],
    times: NDArray[np.float64],
    rate: float,
    *,
    weights: NDArray[np.float64] | None = None,
    continuous: bool = False,
) -> NDArray[np.float64]:
    """Discount per-cycle amounts on a shared time grid and sum, per individual.

    Args:
        amounts: Per-individual per-cycle payoffs, shape ``(n, T)``.
        times: Accrual time of each cycle in years, shape ``(T,)``.
        rate: Annual discount rate.
        weights: Optional per-cycle weights, shape ``(T,)``, for half-cycle
            correction (0.5 on the first and last cycle).
        continuous: Passed through to `discount_factor`.

    Returns:
        Discounted total per individual, shape ``(n,)``.

    Example:
        >>> import numpy as np
        >>> from heormodel.models._accrual import accrue
        >>> amounts = np.ones((1, 3))  # one individual, three cycles
        >>> times = np.array([0.0, 1.0, 2.0])
        >>> float(accrue(amounts, times, 0.0)[0])
        3.0
    """
    factors = discount_factor(times, rate, continuous=continuous)
    if weights is not None:
        factors = factors * np.asarray(weights, dtype=np.float64)
    return np.asarray(amounts, dtype=np.float64) @ factors


def integrate_flow(
    t0: float | NDArray[np.float64],
    duration: float | NDArray[np.float64],
    rate: float,
) -> NDArray[np.float64]:
    """Integrate a unit continuous flow over ``[t0, t0 + duration]``, discounted.

    Returns the integral of ``exp(-rate * u)`` from ``t0`` to ``t0 + duration``.
    Multiply by a per-year flow rate (a cost or utility rate) to accrue over a
    segment between events in the continuous-time engine.

    Args:
        t0: Segment start times in years.
        duration: Segment lengths in years.
        rate: Annual discount rate; ``0`` gives the undiscounted length.

    Example:
        >>> from heormodel.models._accrual import integrate_flow
        >>> round(float(integrate_flow(0.0, 1.0, 0.03)), 6)
        0.985149
        >>> round(float(integrate_flow(0.0, 1.0, 0.0)), 6)
        1.0
    """
    t0 = np.asarray(t0, dtype=np.float64)
    duration = np.asarray(duration, dtype=np.float64)
    if rate == 0.0:
        return duration.astype(np.float64)
    return (np.exp(-rate * t0) - np.exp(-rate * (t0 + duration))) / rate


def aggregate(
    per_individual: dict[str, NDArray[np.float64]],
    intervention: str,
    iteration: object,
) -> pd.DataFrame:
    """Average per-individual accruals into one ``(intervention, iteration)`` row.

    Population averaging happens here: `Outcomes` stays at the intervention and
    iteration grain, and individual-level detail never crosses into it.

    Args:
        per_individual: Column name to per-individual accrual array.
        intervention: Intervention label for the row.
        iteration: Iteration index value for the row.

    Example:
        >>> import numpy as np
        >>> from heormodel.models._accrual import aggregate
        >>> row = aggregate(
        ...     {"cost": np.array([1.0, 3.0]), "qaly": np.array([0.5, 0.7])},
        ...     "Tx", 0)
        >>> float(row.loc[("Tx", 0), "cost"])
        2.0
    """
    means = {col: float(np.mean(v)) for col, v in per_individual.items()}
    index = pd.MultiIndex.from_tuples(
        [(intervention, iteration)], names=[INTERVENTION_LEVEL, ITERATION_LEVEL]
    )
    return pd.DataFrame([means], index=index)
