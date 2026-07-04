"""The standardized outcome schema: the integration contract of ``heval``.

Every model engine, and every bring-your-own-outputs table, is normalised
into an `Outcomes` object: a tidy ``DataFrame`` indexed by
``(strategy, iteration)`` carrying a ``cost`` column, one or more effect
columns (e.g. QALYs), and optional disaggregated cost/effect components.
Every analysis in `heval.cea` and `heval.voi` consumes this object
and nothing else, which is what makes the analysis layer engine-agnostic.

The ``iteration`` level of the index is the same iteration index carried by
the parameter draw matrix from `heval.params`; EVPPI and EVSI rely on
that shared index to trace which parameter draw produced which outcome.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

COST_COL = "cost"
STRATEGY_LEVEL = "strategy"
ITERATION_LEVEL = "iteration"


class Outcomes:
    """PSA outcomes per strategy per iteration, in the standard schema.

    Args:
        data: DataFrame indexed by a two-level ``MultiIndex`` named
            ``("strategy", "iteration")``, with a ``"cost"`` column and at
            least the primary effect column. Any additional numeric columns
            are carried along as disaggregated components.
        effect: Name of the primary effect column (default ``"qaly"``).

    Example:
        >>> import pandas as pd
        >>> from heval.models import Outcomes
        >>> tidy = pd.DataFrame({
        ...     "strategy": ["A", "A", "B", "B"],
        ...     "iteration": [0, 1, 0, 1],
        ...     "cost": [100.0, 110.0, 200.0, 190.0],
        ...     "qaly": [1.0, 1.1, 1.4, 1.3],
        ... })
        >>> out = Outcomes.from_tidy(tidy)
        >>> out.strategies
        ['A', 'B']
        >>> out.n_iterations
        2
    """

    def __init__(self, data: pd.DataFrame, *, effect: str = "qaly") -> None:
        if not isinstance(data.index, pd.MultiIndex) or list(data.index.names) != [
            STRATEGY_LEVEL,
            ITERATION_LEVEL,
        ]:
            raise ValueError(
                "Outcomes data must be indexed by a MultiIndex named "
                f"('{STRATEGY_LEVEL}', '{ITERATION_LEVEL}')."
            )
        if COST_COL not in data.columns:
            raise ValueError(f"Outcomes data must have a '{COST_COL}' column.")
        if effect not in data.columns:
            raise ValueError(f"Effect column {effect!r} not found in outcomes data.")
        if data.index.duplicated().any():
            raise ValueError("Duplicate (strategy, iteration) rows in outcomes data.")
        numeric = data.select_dtypes(include=[np.number])
        if numeric.shape[1] != data.shape[1]:
            bad = [c for c in data.columns if c not in numeric.columns]
            raise ValueError(f"Outcome columns must be numeric; offending columns: {bad}.")
        if not np.isfinite(data.to_numpy(dtype=np.float64)).all():
            raise ValueError("Outcomes data contains NaN or infinite values.")
        self.data = data.astype(np.float64)
        self.effect = effect
        strategies = list(dict.fromkeys(data.index.get_level_values(STRATEGY_LEVEL)))
        self._strategies: list[str] = [str(s) for s in strategies]
        cost_wide = self.data[COST_COL].unstack(STRATEGY_LEVEL)
        if cost_wide.isna().any().any():
            raise ValueError(
                "Unbalanced panel: every strategy must be evaluated on the same iterations."
            )
        self._iterations = cost_wide.index

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_tidy(
        cls,
        df: pd.DataFrame,
        *,
        strategy: str = "strategy",
        iteration: str = "iteration",
        cost: str = "cost",
        effect: str = "qaly",
    ) -> Outcomes:
        """Build from a tidy long table (the bring-your-own-outputs entry point).

        Args:
            df: Long table with one row per (strategy, iteration).
            strategy: Column in ``df`` holding the strategy label.
            iteration: Column in ``df`` holding the PSA iteration.
            cost: Column in ``df`` holding the cost per iteration.
            effect: Column in ``df`` holding the effect (QALYs by default).

        Example:
            >>> import pandas as pd
            >>> from heval.models import Outcomes
            >>> df = pd.DataFrame({"arm": ["A", "B"], "iter": [0, 0],
            ...                    "cost": [1.0, 2.0], "qaly": [0.5, 0.6]})
            >>> Outcomes.from_tidy(df, strategy="arm", iteration="iter").n_iterations
            1
        """
        missing = [c for c in (strategy, iteration, cost, effect) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in tidy outcomes table: {missing}.")
        renames = {strategy: STRATEGY_LEVEL, iteration: ITERATION_LEVEL, cost: COST_COL}
        data = df.rename(columns=renames).set_index([STRATEGY_LEVEL, ITERATION_LEVEL])
        return cls(data, effect=effect)

    @classmethod
    def from_wide(
        cls, costs: pd.DataFrame, effects: pd.DataFrame, *, effect: str = "qaly"
    ) -> Outcomes:
        """Build from two wide tables (iterations x strategies).

        Args:
            costs: Costs, one column per strategy, indexed by iteration.
            effects: Effects with identical shape/labels.
            effect: Name to give the effect column.

        Example:
            >>> import pandas as pd
            >>> from heval.models import Outcomes
            >>> c = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
            >>> e = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]})
            >>> Outcomes.from_wide(c, e).strategies
            ['A', 'B']
        """
        if not costs.columns.equals(effects.columns) or not costs.index.equals(effects.index):
            raise ValueError("costs and effects must share identical index and columns.")
        c = costs.stack()
        e = effects.stack()
        data = pd.DataFrame({COST_COL: c, effect: e})
        data.index = data.index.set_names([ITERATION_LEVEL, STRATEGY_LEVEL])
        data = data.swaplevel().sort_index(level=STRATEGY_LEVEL, sort_remaining=False)
        # preserve the caller's strategy column order rather than sort order
        data = data.reindex(list(costs.columns), level=STRATEGY_LEVEL)
        return cls(data, effect=effect)

    # -- accessors ---------------------------------------------------------

    @property
    def strategies(self) -> list[str]:
        """Strategy names in first-appearance order."""
        return list(self._strategies)

    @property
    def iterations(self) -> pd.Index:
        """The shared iteration index."""
        return self._iterations

    @property
    def n_iterations(self) -> int:
        """Number of PSA iterations."""
        return len(self._iterations)

    @property
    def components(self) -> list[str]:
        """Names of disaggregated component columns beyond cost and primary effect."""
        return [c for c in self.data.columns if c not in (COST_COL, self.effect)]

    def costs_wide(self) -> pd.DataFrame:
        """Costs as an (iterations x strategies) matrix.

        Example:
            >>> import pandas as pd
            >>> from heval.models import Outcomes
            >>> c = pd.DataFrame({"A": [1.0], "B": [2.0]})
            >>> e = pd.DataFrame({"A": [0.1], "B": [0.2]})
            >>> Outcomes.from_wide(c, e).costs_wide().shape
            (1, 2)
        """
        return self.data[COST_COL].unstack(STRATEGY_LEVEL)[self._strategies]

    def effects_wide(self, column: str | None = None) -> pd.DataFrame:
        """An effect column as an (iterations x strategies) matrix.

        Args:
            column: Effect column name; defaults to the primary effect.
        """
        col = column or self.effect
        return self.data[col].unstack(STRATEGY_LEVEL)[self._strategies]

    def summary(self) -> pd.DataFrame:
        """Mean of every outcome column per strategy.

        Example:
            >>> import pandas as pd
            >>> from heval.models import Outcomes
            >>> c = pd.DataFrame({"A": [1.0, 3.0]})
            >>> e = pd.DataFrame({"A": [0.1, 0.3]})
            >>> float(Outcomes.from_wide(c, e).summary()["cost"]["A"])
            2.0
        """
        means = self.data.groupby(level=STRATEGY_LEVEL, sort=False).mean()
        return means.reindex(self._strategies)

    def select(self, strategies: Sequence[str]) -> Outcomes:
        """Subset to the given strategies, preserving the iteration index."""
        unknown = [s for s in strategies if s not in self._strategies]
        if unknown:
            raise KeyError(f"Unknown strategies: {unknown}.")
        subset = self.data.loc[list(strategies)]
        return Outcomes(subset, effect=self.effect)

    def __repr__(self) -> str:
        return (
            f"Outcomes(strategies={self._strategies}, n_iterations={self.n_iterations}, "
            f"effect={self.effect!r}, components={self.components})"
        )
