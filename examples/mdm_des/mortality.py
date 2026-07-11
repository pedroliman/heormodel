"""Read the background-mortality life table from a CSV of age-specific rates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from heormodel.models import LifeTable


def load_life_table(path: str | Path) -> LifeTable:
    """Read an ``age,rate`` CSV into a `LifeTable`.

    Input: a path to a CSV with one row per single year of age, in ascending
    order, giving the annual all-cause mortality rate at that age. Output: a
    `LifeTable` that samples the years until death conditional on the current
    age, optionally under a hazard ratio for the disease states.
    """
    table = pd.read_csv(path)
    return LifeTable(
        ages=table["age"].to_numpy(dtype=float),
        rates=table["rate"].to_numpy(dtype=float),
    )
