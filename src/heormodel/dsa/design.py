"""Scenario designs for deterministic sensitivity analysis.

Each builder takes a base case (a `pandas.Series` of point values, e.g. from
`heormodel.params.ParameterSet.means`) and returns a ``(design, descriptor)``
pair. The ``design`` is a draw matrix ready for `heormodel.run.run_psa`: rows are
scenarios, its index is named ``"iteration"``, and it carries every base
parameter as a column so any engine can evaluate it. The ``descriptor`` is a
tidy table with the same index, one row per scenario, recording what each
scenario varied so outcomes stay interpretable.

The descriptor always carries a ``scenario`` label column. One-way and
one-at-a-time designs add ``parameter`` and ``value`` columns naming the
single parameter each scenario varies (the base scenario uses the label
``"(base)"`` and a missing value). A grid design instead adds one column per
gridded parameter giving its value in the scenario, the form `heatmap_data`
reshapes.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import pandas as pd

#: A DSA design: the ``(design, descriptor)`` pair every builder returns.
Design: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]

SCENARIO_COL = "scenario"
PARAMETER_COL = "parameter"
VALUE_COL = "value"
BASE_LABEL = "(base)"


def _check_names(base: pd.Series, names: Sequence[str]) -> None:
    unknown = [n for n in names if n not in base.index]
    if unknown:
        raise KeyError(f"Parameters not in the base case: {unknown}.")


def _assemble(rows: list[pd.Series], descriptor_rows: list[dict[str, object]]) -> Design:
    index = pd.RangeIndex(len(rows), name="iteration")
    design = pd.DataFrame(rows).set_axis(index, axis=0)
    descriptor = pd.DataFrame(descriptor_rows).set_axis(index, axis=0)
    return design, descriptor


def one_way(base: pd.Series, parameter: str, values: Sequence[float]) -> Design:
    """Vary one parameter across ``values``, holding the rest at base.

    Args:
        base: Point values of every parameter, indexed by name.
        parameter: Name of the parameter to sweep.
        values: The values to set it to, one scenario each.

    Returns:
        A ``(design, descriptor)`` pair with ``len(values)`` scenarios. The
        descriptor names the swept ``parameter`` and its ``value`` per row.

    Example:
        >>> import pandas as pd
        >>> from heormodel.dsa import one_way
        >>> base = pd.Series({"a": 1.0, "b": 2.0})
        >>> design, descriptor = one_way(base, "a", [0.5, 1.5])
        >>> design["a"].tolist()
        [0.5, 1.5]
        >>> design["b"].tolist()
        [2.0, 2.0]
        >>> descriptor["value"].tolist()
        [0.5, 1.5]
    """
    _check_names(base, [parameter])
    values = list(values)
    if not values:
        raise ValueError("values must contain at least one value.")
    rows: list[pd.Series] = []
    descriptor_rows: list[dict[str, object]] = []
    for v in values:
        row = base.copy()
        row[parameter] = v
        rows.append(row[base.index])
        descriptor_rows.append(
            {SCENARIO_COL: f"{parameter}={v:g}", PARAMETER_COL: parameter, VALUE_COL: float(v)}
        )
    return _assemble(rows, descriptor_rows)


def one_at_a_time(base: pd.Series, ranges: Mapping[str, Sequence[float]]) -> Design:
    """Vary each parameter in ``ranges`` in turn, holding the rest at base.

    The union of one-way sweeps, with the base case included once as its own
    scenario. This is the design a tornado diagram reads.

    Args:
        base: Point values of every parameter, indexed by name.
        ranges: Maps a parameter name to the values to sweep it over (a
            ``(low, high)`` pair or any longer sequence).

    Returns:
        A ``(design, descriptor)`` pair with ``1 + sum(len(v))`` scenarios: the
        base case first, then each parameter's sweep. The descriptor names the
        single ``parameter`` each scenario varies and its ``value``.

    Example:
        >>> import pandas as pd
        >>> from heormodel.dsa import one_at_a_time
        >>> base = pd.Series({"a": 1.0, "b": 2.0})
        >>> design, descriptor = one_at_a_time(base, {"a": (0.5, 1.5), "b": (1.0, 3.0)})
        >>> len(design)
        5
        >>> descriptor["scenario"].tolist()
        ['(base)', 'a=0.5', 'a=1.5', 'b=1', 'b=3']
    """
    _check_names(base, list(ranges))
    rows: list[pd.Series] = [base.copy()]
    descriptor_rows: list[dict[str, object]] = [
        {SCENARIO_COL: BASE_LABEL, PARAMETER_COL: BASE_LABEL, VALUE_COL: float("nan")}
    ]
    for name, values in ranges.items():
        seq = list(values)
        if not seq:
            raise ValueError(f"Range for {name!r} is empty.")
        for v in seq:
            row = base.copy()
            row[name] = v
            rows.append(row[base.index])
            descriptor_rows.append(
                {SCENARIO_COL: f"{name}={v:g}", PARAMETER_COL: name, VALUE_COL: float(v)}
            )
    return _assemble(rows, descriptor_rows)


def grid(base: pd.Series, grids: Mapping[str, Sequence[float]]) -> Design:
    """Full-factorial design over the listed parameters, rest at base.

    Args:
        base: Point values of every parameter, indexed by name.
        grids: Maps each gridded parameter to the values it takes; the design
            is every combination of them.

    Returns:
        A ``(design, descriptor)`` pair with ``prod(len(v)) + 1`` scenarios: the
        base case first, then the factorial. The descriptor carries one column
        per gridded parameter with its value in that scenario.

    Example:
        >>> import pandas as pd
        >>> from heormodel.dsa import grid
        >>> base = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        >>> design, descriptor = grid(base, {"a": [0.0, 1.0], "b": [10.0, 20.0]})
        >>> len(design)
        5
        >>> descriptor["scenario"].iloc[0]
        '(base)'
        >>> design["c"].tolist()
        [3.0, 3.0, 3.0, 3.0, 3.0]
    """
    names = list(grids)
    if not names:
        raise ValueError("grids must list at least one parameter.")
    _check_names(base, names)
    value_lists = [list(grids[n]) for n in names]
    for name, seq in zip(names, value_lists, strict=True):
        if not seq:
            raise ValueError(f"Grid for {name!r} is empty.")

    base_descriptor: dict[str, object] = {SCENARIO_COL: BASE_LABEL}
    base_descriptor.update({n: float(base[n]) for n in names})
    rows: list[pd.Series] = [base.copy()]
    descriptor_rows: list[dict[str, object]] = [base_descriptor]

    for combo in itertools.product(*value_lists):
        row = base.copy()
        label = ", ".join(f"{n}={v:g}" for n, v in zip(names, combo, strict=True))
        descriptor_row: dict[str, object] = {SCENARIO_COL: label}
        for n, v in zip(names, combo, strict=True):
            row[n] = v
            descriptor_row[n] = float(v)
        rows.append(row[base.index])
        descriptor_rows.append(descriptor_row)
    return _assemble(rows, descriptor_rows)
