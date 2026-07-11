"""Shared intervention handling for the engines.

An intervention is a name, optionally carrying decision levers merged into
each draw and a flag marking it the comparator. `Intervention` is the
canonical spelling; a bare string is shorthand for an intervention with no
decision levers and no comparator flag. Every engine accepts
``interventions`` as a sequence of either, normalises it here, and passes
the intervention name to its user-supplied model functions, so
branch-on-name and decision-lever styles work identically across engines.

Branch on the name when the arms differ in structure or in which model
function runs (a treatment that changes a utility, a comparator that skips a
step). Use decision levers for numeric scenario knobs the model already
reads as parameters (a server count, a capacity). Decision levers written as
flags (``{"on_treatment": 1.0}`` read back as a boolean) are the case
branch-on-name replaces: a flag standing in for a float is invisible to
`heormodel.voi.evppi_ranking` and to the deterministic sensitivity builders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Intervention:
    """A named intervention, optionally flagged as the comparator.

    Args:
        name: The intervention label, as it appears in `Outcomes`.
        decision_levers: Parameter values written over each draw for this
            intervention. Empty by default; reach for it only for numeric
            scenario knobs, not to encode which arm is which (branch on
            ``name`` for that).
        is_comparator: Marks this intervention as the PICOTS comparator, the
            reference arm incremental results are measured against. At most
            one intervention in a sequence may set this; when none does, the
            first intervention is the comparator.

    Example:
        >>> from heormodel.models import Intervention
        >>> Intervention("Expanded capacity", {"n_servers": 2}).decision_levers
        {'n_servers': 2}
        >>> Intervention("Standard care", is_comparator=True).is_comparator
        True
    """

    name: str
    decision_levers: Mapping[str, Any] = field(default_factory=dict)
    is_comparator: bool = False


#: A sequence of intervention names or `Intervention` objects.
InterventionSpec = Sequence[str | Intervention]


def normalize_interventions(interventions: InterventionSpec) -> dict[str, dict[str, Any]]:
    """Coerce an intervention sequence into an ordered name-to-decision-levers mapping.

    Args:
        interventions: A sequence whose items are intervention names (no
            decision levers) or `Intervention` objects.

    Returns:
        A dict keyed by intervention name, preserving order, whose values are
        the decision-lever dicts (empty for the bare-name form).

    Example:
        >>> from heormodel.models import Intervention
        >>> from heormodel.models._interventions import normalize_interventions
        >>> normalize_interventions(["A", "B"])
        {'A': {}, 'B': {}}
        >>> normalize_interventions([Intervention("Treatment", {"n_servers": 2})])
        {'Treatment': {'n_servers': 2}}
    """
    result: dict[str, dict[str, Any]] = {}
    for item in interventions:
        if isinstance(item, Intervention):
            name, decision_levers = item.name, dict(item.decision_levers)
        elif isinstance(item, str):
            name, decision_levers = item, {}
        else:
            raise TypeError("interventions must be a sequence of names or Intervention objects.")
        if name in result:
            raise ValueError(f"Duplicate intervention name {name!r}.")
        result[name] = decision_levers
    if not result:
        raise ValueError("Provide at least one intervention.")
    return result


def comparator_of(interventions: InterventionSpec) -> str | None:
    """Return the name of the intervention flagged ``is_comparator=True``, if any.

    Args:
        interventions: A sequence whose items are intervention names or
            `Intervention` objects.

    Returns:
        The comparator's name, or ``None`` if no item is flagged.

    Example:
        >>> from heormodel.models import Intervention
        >>> from heormodel.models._interventions import comparator_of
        >>> comparator_of(["A", Intervention("B", is_comparator=True)])
        'B'
        >>> comparator_of(["A", "B"]) is None
        True
    """
    flagged = [
        item.name for item in interventions if isinstance(item, Intervention) and item.is_comparator
    ]
    if len(flagged) > 1:
        raise ValueError(f"At most one intervention may set is_comparator=True; got {flagged}.")
    return flagged[0] if flagged else None


def merge_decision_levers(params: pd.Series, decision_levers: Mapping[str, Any]) -> pd.Series:
    """Return ``params`` with the intervention's decision levers written over it.

    The original series is left untouched when there are no decision levers,
    so the common case allocates nothing.

    Args:
        params: One draw-matrix row.
        decision_levers: The intervention's parameter decision levers.
    """
    if not decision_levers:
        return params
    merged = params.copy()
    for key, value in decision_levers.items():
        merged[key] = value
    return merged
