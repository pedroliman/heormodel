"""Incremental cost-effectiveness analysis on the efficiency frontier.

Implements the standard decision-analytic algorithm: order interventions by
cost, remove strongly dominated interventions (more costly and no more
effective than another), then iteratively remove extendedly dominated
interventions (whose ICER exceeds that of the next more effective intervention)
until ICERs increase monotonically along the frontier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heormodel.models.outcomes import Outcomes

#: Status labels: not dominated (on the frontier), dominated, extendedly dominated.
STATUS_ND = "ND"
STATUS_D = "D"
STATUS_ED = "ED"


def _mean_table(source: Outcomes | pd.DataFrame, effect: str | None) -> pd.DataFrame:
    if isinstance(source, Outcomes):
        eff = effect or source.effect
        summary = source.summary()
        return pd.DataFrame({"cost": summary["cost"], "effect": summary[eff]})
    df = source.copy()
    eff = effect or ("effect" if "effect" in df.columns else "qaly")
    missing = [c for c in ("cost", eff) if c not in df.columns]
    if missing:
        raise ValueError(f"Mean outcome table is missing columns: {missing}.")
    return pd.DataFrame({"cost": df["cost"], "effect": df[eff]})


def icer_table(
    source: Outcomes | pd.DataFrame,
    *,
    effect: str | None = None,
    interval: float | None = 0.95,
) -> pd.DataFrame:
    """Full incremental analysis: dominance, extended dominance, and ICERs.

    Args:
        source: probabilistic `Outcomes` (means are taken per intervention) or a
            per-intervention mean table indexed by intervention with columns
            ``cost`` and the effect column.
        effect: Effect column name (defaults to the outcomes' primary
            effect, or ``"effect"`` for plain tables).
        interval: Central probability for the uncertainty interval, or ``None``
            to omit intervals. When ``source`` is a probabilistic `Outcomes`
            with more than one iteration, each estimate gains ``_lo``/``_hi``
            columns holding the two-sided interval across parameter draws (the
            default ``0.95`` gives a 95% interval, the 2.5th and 97.5th
            percentiles). A mean table or a single draw carries no intervals.

    Returns:
        DataFrame indexed by intervention, sorted by cost, with columns
        ``cost``, ``effect``, ``inc_cost``, ``inc_effect``, ``icer`` and
        ``status`` (``"ND"`` on the frontier, ``"D"`` strongly dominated,
        ``"ED"`` extendedly dominated). Every intervention except the cheapest
        carries an incremental cost and effect against its comparator, the
        cheapest frontier intervention still above it in cost order, so a
        dominated intervention shows the negative incremental effect or excess
        cost that marks it dominated. The ICER is a frontier quantity, filled
        between adjacent frontier interventions and left blank for dominated
        ones and for the cheapest frontier intervention.
        With intervals, each of ``cost``, ``effect``, ``inc_cost``,
        ``inc_effect`` and ``icer`` is followed by its ``_lo`` and ``_hi``
        bounds. Dominance and the frontier are settled once on the mean costs
        and effects; the intervals describe the spread of each measure for that
        fixed frontier. The incremental measures are differences between
        interventions, so their intervals are taken from the paired per-iteration
        difference (``cost`` of the intervention minus ``cost`` of its comparator
        in the same draw), not from the separate intervals of the two
        interventions. The ICER interval comes from the per-draw ratio of paired
        incremental cost to paired incremental effect; draws whose incremental
        effect approaches zero make that ratio unstable, so read the incremental
        cost and effect intervals alongside it.

    Example:
        >>> import pandas as pd
        >>> from heormodel.cea import icer_table
        >>> means = pd.DataFrame(
        ...     {"cost": [0.0, 100.0, 400.0], "effect": [0.0, 0.5, 1.0]},
        ...     index=["A", "B", "D"],
        ... )
        >>> t = icer_table(means)
        >>> float(t.loc["D", "icer"])
        600.0
    """
    means = _mean_table(source, effect)
    means = means.sort_values(["cost", "effect"], ascending=[True, False])
    cost = means["cost"].to_numpy(dtype=np.float64)
    eff = means["effect"].to_numpy(dtype=np.float64)
    n = len(means)
    status = np.array([STATUS_ND] * n, dtype=object)

    # strong dominance: another intervention is no more costly and no less
    # effective, strictly better on at least one axis; exact duplicates keep
    # only the first occurrence in cost order
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weak = cost[j] <= cost[i] and eff[j] >= eff[i]
            strict = cost[j] < cost[i] or eff[j] > eff[i]
            duplicate = j < i and cost[j] == cost[i] and eff[j] == eff[i]
            if (weak and strict) or duplicate:
                status[i] = STATUS_D
                break

    # extended dominance: remove interventions until ICERs are monotone
    while True:
        nd = [i for i in range(n) if status[i] == STATUS_ND]
        if len(nd) < 3:
            break
        icers = [
            (cost[nd[k]] - cost[nd[k - 1]]) / (eff[nd[k]] - eff[nd[k - 1]])
            for k in range(1, len(nd))
        ]
        for k in range(len(icers) - 1):
            if icers[k] > icers[k + 1]:
                status[nd[k + 1]] = STATUS_ED
                break
        else:
            break

    # Incremental cost and effect are shown for every intervention against the
    # cheapest frontier intervention still above it in cost order (its reference
    # for the dominance test), so dominated interventions carry the negative
    # incremental effect or excess cost that marks them dominated. The ICER is a
    # frontier quantity, so it is filled only between adjacent frontier
    # interventions; the ratio is left blank for dominated ones.
    inc_cost = np.full(n, np.nan)
    inc_eff = np.full(n, np.nan)
    icer = np.full(n, np.nan)
    comparators: dict[str, str] = {}
    last_nd = -1
    for i in range(n):
        if last_nd >= 0:
            inc_cost[i] = cost[i] - cost[last_nd]
            inc_eff[i] = eff[i] - eff[last_nd]
            comparators[str(means.index[i])] = str(means.index[last_nd])
            if status[i] == STATUS_ND:
                icer[i] = inc_cost[i] / inc_eff[i]
        if status[i] == STATUS_ND:
            last_nd = i

    result = means.copy()
    result["inc_cost"] = inc_cost
    result["inc_effect"] = inc_eff
    result["icer"] = icer
    result["status"] = status
    result.index.name = "intervention"

    if interval is not None and isinstance(source, Outcomes) and source.n_iterations > 1:
        frontier_labels = {str(means.index[i]) for i in range(n) if status[i] == STATUS_ND}
        result = _add_intervals(source, result, effect, interval, comparators, frontier_labels)
    return result


def _add_intervals(
    source: Outcomes,
    result: pd.DataFrame,
    effect: str | None,
    interval: float,
    comparators: dict[str, str],
    frontier_labels: set[str],
) -> pd.DataFrame:
    """Attach ``_lo``/``_hi`` interval columns to a mean incremental table.

    Per-strategy cost and effect intervals are the percentiles of each
    intervention's own draws. The incremental measures are differences between
    strategies, so their intervals are taken from the paired per-iteration
    difference between an intervention and its comparator, the cheapest frontier
    intervention still above it in cost order, not from the two interventions'
    separate intervals. The ICER interval is filled only for frontier
    interventions, matching the mean table.
    """
    if not 0.0 < interval < 1.0:
        raise ValueError(f"interval must be strictly between 0 and 1, got {interval}.")
    lo_q = 100.0 * (1.0 - interval) / 2.0
    hi_q = 100.0 * (1.0 + interval) / 2.0

    costs = source.costs_wide()
    effects = source.effects_wide(effect)
    labels = [str(s) for s in result.index]

    bounds: dict[str, dict[str, float]] = {label: {} for label in labels}
    for label in labels:
        bounds[label]["cost_lo"], bounds[label]["cost_hi"] = np.percentile(
            costs[label].to_numpy(dtype=np.float64), [lo_q, hi_q]
        )
        bounds[label]["effect_lo"], bounds[label]["effect_hi"] = np.percentile(
            effects[label].to_numpy(dtype=np.float64), [lo_q, hi_q]
        )

    for cur, prev in comparators.items():
        inc_cost = (costs[cur] - costs[prev]).to_numpy(dtype=np.float64)
        inc_effect = (effects[cur] - effects[prev]).to_numpy(dtype=np.float64)
        bounds[cur]["inc_cost_lo"], bounds[cur]["inc_cost_hi"] = np.percentile(
            inc_cost, [lo_q, hi_q]
        )
        bounds[cur]["inc_effect_lo"], bounds[cur]["inc_effect_hi"] = np.percentile(
            inc_effect, [lo_q, hi_q]
        )
        if cur in frontier_labels:
            icer = inc_cost / inc_effect
            bounds[cur]["icer_lo"], bounds[cur]["icer_hi"] = np.percentile(icer, [lo_q, hi_q])

    # interleave each estimate with its lower and upper bounds, status last
    ordered: list[str] = []
    for estimate in ("cost", "effect", "inc_cost", "inc_effect", "icer"):
        ordered += [estimate, f"{estimate}_lo", f"{estimate}_hi"]
    ordered.append("status")

    out = result.copy()
    for estimate in ("cost", "effect", "inc_cost", "inc_effect", "icer"):
        out[f"{estimate}_lo"] = [bounds[label].get(f"{estimate}_lo", np.nan) for label in labels]
        out[f"{estimate}_hi"] = [bounds[label].get(f"{estimate}_hi", np.nan) for label in labels]
    return out[ordered]


def frontier(source: Outcomes | pd.DataFrame, *, effect: str | None = None) -> list[str]:
    """Interventions on the cost-effectiveness efficiency frontier, cheapest first.

    Example:
        >>> import pandas as pd
        >>> from heormodel.cea import frontier
        >>> means = pd.DataFrame(
        ...     {"cost": [0.0, 10.0, 5.0], "effect": [0.0, 1.0, -1.0]},
        ...     index=["A", "B", "C"],
        ... )
        >>> frontier(means)
        ['A', 'B']
    """
    table = icer_table(source, effect=effect)
    return [str(s) for s in table.index[table["status"] == STATUS_ND]]
