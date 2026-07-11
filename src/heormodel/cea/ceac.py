"""Acceptability curves, frontier, expected loss curves, and CE-plane data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from heormodel.models.outcomes import Outcomes


def ceac(
    outcomes: Outcomes, wtp: ArrayLike | Sequence[float], *, effect: str | None = None
) -> pd.DataFrame:
    """Cost-effectiveness acceptability curve for every intervention.

    For each willingness-to-pay value, the probability (share of
    iterations) that each intervention has the highest net monetary benefit.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        wtp: Grid of willingness-to-pay values.
        effect: Effect column (default: the primary effect).

    Returns:
        DataFrame indexed by ``wtp`` with one probability column per
        intervention; rows sum to 1.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.cea import ceac
        >>> c = pd.DataFrame({"A": [0.0, 0.0], "B": [10.0, 10.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, -1.0]})
        >>> float(ceac(Outcomes.from_wide(c, e), wtp=[100.0]).loc[100.0, "B"])
        0.5
    """
    wtp_grid = np.atleast_1d(np.asarray(wtp, dtype=np.float64))
    costs = outcomes.costs_wide().to_numpy(dtype=np.float64)
    effects = outcomes.effects_wide(effect).to_numpy(dtype=np.float64)
    n, d = costs.shape
    probs = np.empty((len(wtp_grid), d), dtype=np.float64)
    for i, lam in enumerate(wtp_grid):
        winners = np.argmax(lam * effects - costs, axis=1)
        probs[i] = np.bincount(winners, minlength=d) / n
    return pd.DataFrame(probs, index=pd.Index(wtp_grid, name="wtp"), columns=outcomes.interventions)


def expected_loss(
    outcomes: Outcomes, wtp: ArrayLike | Sequence[float], *, effect: str | None = None
) -> pd.DataFrame:
    """Expected loss curve: mean foregone net benefit per intervention.

    In each iteration, an intervention's loss is the gap between its net monetary
    benefit and the best intervention's in that iteration; the curve is the mean
    loss over iterations at each willingness-to-pay value. The intervention with
    the lowest expected loss is the optimal choice, and its expected loss
    equals the expected value of perfect information, so the curves show both
    the ranking and the cost of decision uncertainty on one money scale.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        wtp: Grid of willingness-to-pay values.
        effect: Effect column (default: the primary effect).

    Returns:
        DataFrame indexed by ``wtp`` with one expected-loss column per
        intervention.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.cea import expected_loss
        >>> c = pd.DataFrame({"A": [0.0, 0.0], "B": [10.0, 10.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, -1.0]})
        >>> curves = expected_loss(Outcomes.from_wide(c, e), wtp=[10.0])
        >>> curves.loc[10.0].round(1).tolist()  # loss of A, loss of B
        [0.0, 10.0]
    """
    wtp_grid = np.atleast_1d(np.asarray(wtp, dtype=np.float64))
    costs = outcomes.costs_wide().to_numpy(dtype=np.float64)
    effects = outcomes.effects_wide(effect).to_numpy(dtype=np.float64)
    losses = np.empty((len(wtp_grid), costs.shape[1]), dtype=np.float64)
    for i, lam in enumerate(wtp_grid):
        nb = lam * effects - costs
        losses[i] = (nb.max(axis=1, keepdims=True) - nb).mean(axis=0)
    return pd.DataFrame(
        losses, index=pd.Index(wtp_grid, name="wtp"), columns=outcomes.interventions
    )


def ceaf(
    outcomes: Outcomes, wtp: ArrayLike | Sequence[float], *, effect: str | None = None
) -> pd.DataFrame:
    """Cost-effectiveness acceptability frontier.

    At each willingness-to-pay value, identifies the intervention with the
    highest **expected** NMB (the optimal choice for a risk-neutral decision
    maker) and reports its acceptability-curve probability.

    Returns:
        DataFrame indexed by ``wtp`` with columns ``intervention`` (the optimal
        intervention) and ``prob`` (its probability of being cost-effective).

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.cea import ceaf
        >>> c = pd.DataFrame({"A": [0.0, 0.0], "B": [10.0, 10.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, 1.0]})
        >>> ceaf(Outcomes.from_wide(c, e), wtp=[100.0]).loc[100.0, "intervention"]
        'B'
    """
    wtp_grid = np.atleast_1d(np.asarray(wtp, dtype=np.float64))
    curve = ceac(outcomes, wtp_grid, effect=effect)
    costs = outcomes.costs_wide().to_numpy(dtype=np.float64)
    effects = outcomes.effects_wide(effect).to_numpy(dtype=np.float64)
    mean_cost = costs.mean(axis=0)
    mean_eff = effects.mean(axis=0)
    interventions = np.asarray(outcomes.interventions, dtype=object)
    optimal = [interventions[int(np.argmax(lam * mean_eff - mean_cost))] for lam in wtp_grid]
    prob = [curve.iloc[i][opt] for i, opt in enumerate(optimal)]
    idx = pd.Index(wtp_grid, name="wtp")
    return pd.DataFrame({"intervention": optimal, "prob": prob}, index=idx)


def ce_plane(
    outcomes: Outcomes, *, comparator: str | None = None, effect: str | None = None
) -> pd.DataFrame:
    """Incremental cost and effect per iteration versus a comparator.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        comparator: Reference intervention (default: the intervention flagged
            ``is_comparator=True`` at construction, via ``outcomes.comparator``,
            or the first intervention if none was flagged).
        effect: Effect column (default: the primary effect).

    Returns:
        Tidy DataFrame with columns ``intervention``, ``iteration``,
        ``inc_cost`` and ``inc_effect`` for every non-comparator intervention,
        ready to scatter on the cost-effectiveness plane.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.cea import ce_plane
        >>> c = pd.DataFrame({"A": [0.0], "B": [10.0]})
        >>> e = pd.DataFrame({"A": [0.0], "B": [0.5]})
        >>> float(ce_plane(Outcomes.from_wide(c, e))["inc_cost"][0])
        10.0
    """
    ref = comparator or outcomes.comparator or outcomes.interventions[0]
    if ref not in outcomes.interventions:
        raise KeyError(f"Unknown comparator intervention: {ref!r}.")
    costs = outcomes.costs_wide()
    effects = outcomes.effects_wide(effect)
    frames = []
    for s in outcomes.interventions:
        if s == ref:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "intervention": s,
                    "iteration": costs.index,
                    "inc_cost": (costs[s] - costs[ref]).to_numpy(),
                    "inc_effect": (effects[s] - effects[ref]).to_numpy(),
                }
            )
        )
    if not frames:
        raise ValueError("ce_plane needs at least two interventions.")
    return pd.concat(frames, ignore_index=True)
