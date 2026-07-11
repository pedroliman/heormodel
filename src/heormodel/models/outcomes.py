"""The standardized outcome structure: the integration contract of ``heormodel``.

Every model engine, and every bring-your-own-outputs table, is normalised
into an `Outcomes` object: a tidy ``DataFrame`` indexed by
``(intervention, iteration)`` carrying a ``cost`` column, one or more effect
columns (e.g. QALYs), and optional disaggregated cost/effect components.
Every analysis in `heormodel.cea` and `heormodel.voi` consumes this object
and nothing else, which is what makes the analysis layer engine-agnostic.

The ``iteration`` level of the index is the same iteration index carried by
the parameter draw matrix from `heormodel.params`; EVPPI and EVSI rely on
that shared index to trace which parameter draw produced which outcome.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

COST_COL = "cost"
INTERVENTION_LEVEL = "intervention"
ITERATION_LEVEL = "iteration"


class Outcomes:
    """Probabilistic sensitivity analysis outcomes per intervention per iteration.

    Args:
        data: DataFrame indexed by a two-level ``MultiIndex`` named
            ``("intervention", "iteration")``, with a ``"cost"`` column and at
            least the primary effect column. Any additional numeric columns
            are carried along as disaggregated components.
        effect: Name of the primary effect column (default ``"qaly"``).
        comparator: Name of the reference intervention (the PICOTS
            comparator), or ``None`` if no intervention was flagged
            ``is_comparator=True``. `heormodel.cea.ce_plane` and
            `heormodel.report.tornado_data` fall back to this, then to the
            first intervention, when their own ``comparator`` argument is
            omitted.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> tidy = pd.DataFrame({
        ...     "intervention": ["A", "A", "B", "B"],
        ...     "iteration": [0, 1, 0, 1],
        ...     "cost": [100.0, 110.0, 200.0, 190.0],
        ...     "qaly": [1.0, 1.1, 1.4, 1.3],
        ... })
        >>> out = Outcomes.from_tidy(tidy)
        >>> out.interventions
        ['A', 'B']
        >>> out.n_iterations
        2
    """

    def __init__(
        self, data: pd.DataFrame, *, effect: str = "qaly", comparator: str | None = None
    ) -> None:
        if not isinstance(data.index, pd.MultiIndex) or list(data.index.names) != [
            INTERVENTION_LEVEL,
            ITERATION_LEVEL,
        ]:
            raise ValueError(
                "Outcomes data must be indexed by a MultiIndex named "
                f"('{INTERVENTION_LEVEL}', '{ITERATION_LEVEL}')."
            )
        if COST_COL not in data.columns:
            raise ValueError(f"Outcomes data must have a '{COST_COL}' column.")
        if effect not in data.columns:
            raise ValueError(f"Effect column {effect!r} not found in outcomes data.")
        if data.index.duplicated().any():
            raise ValueError("Duplicate (intervention, iteration) rows in outcomes data.")
        numeric = data.select_dtypes(include=[np.number])
        if numeric.shape[1] != data.shape[1]:
            bad = [c for c in data.columns if c not in numeric.columns]
            raise ValueError(f"Outcome columns must be numeric; offending columns: {bad}.")
        if not np.isfinite(data.to_numpy(dtype=np.float64)).all():
            raise ValueError("Outcomes data contains NaN or infinite values.")
        self.data = data.astype(np.float64)
        self.effect = effect
        interventions = list(dict.fromkeys(data.index.get_level_values(INTERVENTION_LEVEL)))
        self._interventions: list[str] = [str(s) for s in interventions]
        if comparator is not None and comparator not in self._interventions:
            raise KeyError(f"Unknown comparator intervention: {comparator!r}.")
        self.comparator = comparator
        cost_wide = self.data[COST_COL].unstack(INTERVENTION_LEVEL)
        if cost_wide.isna().any().any():
            raise ValueError(
                "Unbalanced panel: every intervention must be evaluated on the same iterations."
            )
        self._iterations = cost_wide.index

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_tidy(
        cls,
        df: pd.DataFrame,
        *,
        intervention: str = "intervention",
        iteration: str = "iteration",
        cost: str = "cost",
        effect: str = "qaly",
        comparator: str | None = None,
    ) -> Outcomes:
        """Build from a tidy long table (the bring-your-own-outputs entry point).

        Args:
            df: Long table with one row per (intervention, iteration).
            intervention: Column in ``df`` holding the intervention label.
            iteration: Column in ``df`` holding the iteration index.
            cost: Column in ``df`` holding the cost per iteration.
            effect: Column in ``df`` holding the effect (QALYs by default).
            comparator: Name of the reference intervention, or ``None``.

        Example:
            >>> import pandas as pd
            >>> from heormodel.models import Outcomes
            >>> df = pd.DataFrame({"arm": ["A", "B"], "iter": [0, 0],
            ...                    "cost": [1.0, 2.0], "qaly": [0.5, 0.6]})
            >>> Outcomes.from_tidy(df, intervention="arm", iteration="iter").n_iterations
            1
        """
        missing = [c for c in (intervention, iteration, cost, effect) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in tidy outcomes table: {missing}.")
        renames = {intervention: INTERVENTION_LEVEL, iteration: ITERATION_LEVEL, cost: COST_COL}
        data = df.rename(columns=renames).set_index([INTERVENTION_LEVEL, ITERATION_LEVEL])
        return cls(data, effect=effect, comparator=comparator)

    @classmethod
    def from_wide(
        cls,
        costs: pd.DataFrame,
        effects: pd.DataFrame,
        *,
        effect: str = "qaly",
        comparator: str | None = None,
    ) -> Outcomes:
        """Build from two wide tables (iterations x interventions).

        Args:
            costs: Costs, one column per intervention, indexed by iteration.
            effects: Effects with identical shape/labels.
            effect: Name to give the effect column.
            comparator: Name of the reference intervention, or ``None``.

        Example:
            >>> import pandas as pd
            >>> from heormodel.models import Outcomes
            >>> c = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
            >>> e = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]})
            >>> Outcomes.from_wide(c, e).interventions
            ['A', 'B']
        """
        if not costs.columns.equals(effects.columns) or not costs.index.equals(effects.index):
            raise ValueError("costs and effects must share identical index and columns.")
        c = costs.stack()
        e = effects.stack()
        data = pd.DataFrame({COST_COL: c, effect: e})
        data.index = data.index.set_names([ITERATION_LEVEL, INTERVENTION_LEVEL])
        data = data.swaplevel().sort_index(level=INTERVENTION_LEVEL, sort_remaining=False)
        # preserve the caller's intervention column order rather than sort order
        data = data.reindex(list(costs.columns), level=INTERVENTION_LEVEL)
        return cls(data, effect=effect, comparator=comparator)

    # -- accessors ---------------------------------------------------------

    @property
    def interventions(self) -> list[str]:
        """Intervention names in first-appearance order."""
        return list(self._interventions)

    @property
    def iterations(self) -> pd.Index:
        """The shared iteration index."""
        return self._iterations

    @property
    def n_iterations(self) -> int:
        """Number of iterations."""
        return len(self._iterations)

    @property
    def components(self) -> list[str]:
        """Names of disaggregated component columns beyond cost and primary effect."""
        return [c for c in self.data.columns if c not in (COST_COL, self.effect)]

    def costs_wide(self) -> pd.DataFrame:
        """Costs as an (iterations x interventions) matrix.

        Example:
            >>> import pandas as pd
            >>> from heormodel.models import Outcomes
            >>> c = pd.DataFrame({"A": [1.0], "B": [2.0]})
            >>> e = pd.DataFrame({"A": [0.1], "B": [0.2]})
            >>> Outcomes.from_wide(c, e).costs_wide().shape
            (1, 2)
        """
        return self.data[COST_COL].unstack(INTERVENTION_LEVEL)[self._interventions]

    def effects_wide(self, column: str | None = None) -> pd.DataFrame:
        """An effect column as an (iterations x interventions) matrix.

        Args:
            column: Effect column name; defaults to the primary effect.
        """
        col = column or self.effect
        return self.data[col].unstack(INTERVENTION_LEVEL)[self._interventions]

    def summary(self) -> pd.DataFrame:
        """Mean of every outcome column per intervention.

        Example:
            >>> import pandas as pd
            >>> from heormodel.models import Outcomes
            >>> c = pd.DataFrame({"A": [1.0, 3.0]})
            >>> e = pd.DataFrame({"A": [0.1, 0.3]})
            >>> float(Outcomes.from_wide(c, e).summary()["cost"]["A"])
            2.0
        """
        means = self.data.groupby(level=INTERVENTION_LEVEL, sort=False).mean()
        return means.reindex(self._interventions)

    def select(self, interventions: Sequence[str]) -> Outcomes:
        """Subset to the given interventions, preserving the iteration index."""
        unknown = [s for s in interventions if s not in self._interventions]
        if unknown:
            raise KeyError(f"Unknown interventions: {unknown}.")
        subset = self.data.loc[list(interventions)]
        comparator = self.comparator if self.comparator in interventions else None
        return Outcomes(subset, effect=self.effect, comparator=comparator)

    def __repr__(self) -> str:
        return (
            f"Outcomes(interventions={self._interventions}, n_iterations={self.n_iterations}, "
            f"effect={self.effect!r}, comparator={self.comparator!r}, "
            f"components={self.components})"
        )
