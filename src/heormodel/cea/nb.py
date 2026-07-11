"""Net monetary and net health benefit."""

from __future__ import annotations

import pandas as pd

from heormodel.models.outcomes import Outcomes


def nmb(outcomes: Outcomes, wtp: float, *, effect: str | None = None) -> pd.DataFrame:
    """Net monetary benefit per iteration and intervention: ``wtp * effect - cost``.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        wtp: Willingness to pay per unit of effect.
        effect: Effect column (default: the primary effect).

    Returns:
        DataFrame (iterations x interventions) of NMB values.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.cea import nmb
        >>> c = pd.DataFrame({"A": [100.0]})
        >>> e = pd.DataFrame({"A": [0.01]})
        >>> float(nmb(Outcomes.from_wide(c, e), wtp=50_000)["A"][0])
        400.0
    """
    return wtp * outcomes.effects_wide(effect) - outcomes.costs_wide()


def nhb(outcomes: Outcomes, wtp: float, *, effect: str | None = None) -> pd.DataFrame:
    """Net health benefit per iteration and intervention: ``effect - cost / wtp``.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.cea import nhb
        >>> c = pd.DataFrame({"A": [100.0]})
        >>> e = pd.DataFrame({"A": [0.01]})
        >>> float(nhb(Outcomes.from_wide(c, e), wtp=50_000)["A"][0])
        0.008
    """
    if wtp <= 0:
        raise ValueError("wtp must be positive for net health benefit.")
    return outcomes.effects_wide(effect) - outcomes.costs_wide() / wtp


def expected_nmb(outcomes: Outcomes, wtp: float, *, effect: str | None = None) -> pd.Series:
    """Expected (mean over iterations) NMB per intervention.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.cea import expected_nmb
        >>> c = pd.DataFrame({"A": [100.0, 200.0]})
        >>> e = pd.DataFrame({"A": [0.01, 0.01]})
        >>> float(expected_nmb(Outcomes.from_wide(c, e), wtp=50_000)["A"])
        350.0
    """
    return nmb(outcomes, wtp, effect=effect).mean(axis=0)
