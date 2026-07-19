"""Display formatting for the incremental cost-effectiveness ratio (ICER) table.

`heormodel.cea.icer_table` returns numbers so the value-of-information layer and
the plots can read them. `format_icer_table` turns that numeric table into a
table meant to be read: each estimate rounded and, when the outcomes are
probabilistic, written as ``point (low, high)`` with its uncertainty interval.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from heormodel.cea.frontier import icer_table
from heormodel.models.outcomes import Outcomes

#: Measures shown, in table order, with their default decimal places. Costs and
#: the ratio read as whole currency units; effects keep two decimals.
_MEASURES: tuple[str, ...] = ("cost", "effect", "inc_cost", "inc_effect", "icer")
_DEFAULT_DIGITS: dict[str, int] = {
    "cost": 0,
    "effect": 2,
    "inc_cost": 0,
    "inc_effect": 2,
    "icer": 0,
}

#: Sentence-case, spelled-out column headers for the reading table.
_DISPLAY_NAMES: dict[str, str] = {
    "cost": "Cost",
    "effect": "Effect",
    "inc_cost": "Incremental cost",
    "inc_effect": "Incremental effect",
    "icer": "ICER",
    "status": "Status",
}


def _cell(point: float, low: float | None, high: float | None, digits: int) -> str:
    """Format one estimate as ``point`` or ``point (low, high)``, or blank."""
    if pd.isna(point):
        return ""
    formatted = f"{point:,.{digits}f}"
    if low is None or pd.isna(low):
        return formatted
    return f"{formatted} ({low:,.{digits}f}, {high:,.{digits}f})"


def format_icer_table(
    source: Outcomes | pd.DataFrame,
    *,
    effect: str | None = None,
    interval: float | None = 0.95,
    digits: int | Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Render the ICER table for reading, with rounded numbers and intervals.

    Args:
        source: Same input as `heormodel.cea.icer_table`: a probabilistic
            `Outcomes` or a per-intervention mean table.
        effect: Effect column name, passed through to `icer_table`.
        interval: Central probability for the uncertainty interval, passed
            through to `icer_table`. With a probabilistic `Outcomes` of more
            than one iteration, each estimate is written ``point (low, high)``;
            otherwise only the point estimate is shown.
        digits: Decimal places. An integer applies to every measure; a mapping
            from measure name (``"cost"``, ``"effect"``, ``"inc_cost"``,
            ``"inc_effect"``, ``"icer"``) sets them one at a time and falls back
            to the default for any measure left out. The default rounds costs
            and the ratio to whole units and effects to two decimals.

    Returns:
        DataFrame of strings indexed by intervention, sorted by cost, with
        sentence-case columns ``Cost``, ``Effect``, ``Incremental cost``,
        ``Incremental effect``, ``ICER`` and ``Status``. Cells with no value (the
        cheapest frontier intervention's incremental columns, or any dominated
        intervention's ICER) are blank.

    Example:
        >>> import pandas as pd
        >>> from heormodel.report import format_icer_table
        >>> means = pd.DataFrame(
        ...     {"cost": [0.0, 100.0, 400.0], "effect": [0.0, 0.5, 1.0]},
        ...     index=["A", "B", "D"],
        ... )
        >>> format_icer_table(means).loc["D", "ICER"]
        '600'
    """
    if isinstance(digits, Mapping):
        places = {**_DEFAULT_DIGITS, **digits}
    elif digits is None:
        places = dict(_DEFAULT_DIGITS)
    else:
        places = {measure: digits for measure in _MEASURES}

    table = icer_table(source, effect=effect, interval=interval)
    out = pd.DataFrame(index=table.index)
    for measure in _MEASURES:
        point = table[measure]
        low = table.get(f"{measure}_lo")
        high = table.get(f"{measure}_hi")
        out[measure] = [
            _cell(
                point.iloc[i],
                None if low is None else low.iloc[i],
                None if high is None else high.iloc[i],
                places[measure],
            )
            for i in range(len(table))
        ]
    out["status"] = table["status"]
    out = out.rename(columns=_DISPLAY_NAMES)
    out.index = out.index.rename("Intervention")
    return out
