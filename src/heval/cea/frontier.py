"""Incremental cost-effectiveness analysis on the efficiency frontier.

Implements the standard decision-analytic algorithm: order strategies by
cost, remove strongly dominated strategies (more costly and no more
effective than another), then iteratively remove extendedly dominated
strategies (whose ICER exceeds that of the next more effective strategy)
until ICERs increase monotonically along the frontier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heval.models.outcomes import Outcomes

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


def icer_table(source: Outcomes | pd.DataFrame, *, effect: str | None = None) -> pd.DataFrame:
    """Full incremental analysis: dominance, extended dominance, and ICERs.

    Args:
        source: PSA `Outcomes` (means are taken per strategy) or a
            per-strategy mean table indexed by strategy with columns
            ``cost`` and the effect column.
        effect: Effect column name (defaults to the outcomes' primary
            effect, or ``"effect"`` for plain tables).

    Returns:
        DataFrame indexed by strategy, sorted by cost, with columns
        ``cost``, ``effect``, ``inc_cost``, ``inc_effect``, ``icer`` and
        ``status`` (``"ND"`` on the frontier, ``"D"`` strongly dominated,
        ``"ED"`` extendedly dominated). ICERs are computed between adjacent
        frontier strategies; the cheapest frontier strategy has no ICER.

    Example:
        >>> import pandas as pd
        >>> from heval.cea import icer_table
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

    # strong dominance: another strategy is no more costly and no less
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

    # extended dominance: remove strategies until ICERs are monotone
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

    inc_cost = np.full(n, np.nan)
    inc_eff = np.full(n, np.nan)
    icer = np.full(n, np.nan)
    nd = [i for i in range(n) if status[i] == STATUS_ND]
    for k in range(1, len(nd)):
        prev, cur = nd[k - 1], nd[k]
        inc_cost[cur] = cost[cur] - cost[prev]
        inc_eff[cur] = eff[cur] - eff[prev]
        icer[cur] = inc_cost[cur] / inc_eff[cur]

    result = means.copy()
    result["inc_cost"] = inc_cost
    result["inc_effect"] = inc_eff
    result["icer"] = icer
    result["status"] = status
    result.index.name = "strategy"
    return result


def frontier(source: Outcomes | pd.DataFrame, *, effect: str | None = None) -> list[str]:
    """Strategies on the cost-effectiveness efficiency frontier, cheapest first.

    Example:
        >>> import pandas as pd
        >>> from heval.cea import frontier
        >>> means = pd.DataFrame(
        ...     {"cost": [0.0, 10.0, 5.0], "effect": [0.0, 1.0, -1.0]},
        ...     index=["A", "B", "C"],
        ... )
        >>> frontier(means)
        ['A', 'B']
    """
    table = icer_table(source, effect=effect)
    return [str(s) for s in table.index[table["status"] == STATUS_ND]]
