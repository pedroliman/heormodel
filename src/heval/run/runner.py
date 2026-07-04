"""The run loop: drive a model over parameter draws, or ingest external outputs.

Two first-class entry points:

- `run_psa` evaluates a model engine (or plain callable) over the
  parameter draw matrix, optionally in parallel via ``joblib``, and
  guarantees the returned outcomes carry the draws' iteration index.
- `as_outcomes` normalises a bring-your-own-outputs PSA table (from
  any external simulator or spreadsheet export) into the standard
  `Outcomes` schema so it can flow straight
  into `heval.cea` and `heval.voi` without touching an engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from heval.models.outcomes import ITERATION_LEVEL, STRATEGY_LEVEL, Outcomes
from heval.models.protocol import ModelEngine, ModelFn


def as_outcomes(
    source: Outcomes | pd.DataFrame | str | Path,
    *,
    strategy: str = "strategy",
    iteration: str = "iteration",
    cost: str = "cost",
    effect: str = "qaly",
) -> Outcomes:
    """Normalise any costs/effects PSA table into the standard outcome schema.

    This is the bring-your-own-outputs entry point: feed a tidy table from
    any source directly into the analysis layer.

    Args:
        source: An `Outcomes` (returned unchanged), a tidy long
            ``DataFrame``, or a path to a CSV file of one.
        strategy: Column holding the strategy label.
        iteration: Column holding the PSA iteration.
        cost: Column holding the cost per iteration.
        effect: Column holding the effect (QALYs by default).

    Example:
        >>> import pandas as pd
        >>> from heval.run import as_outcomes
        >>> df = pd.DataFrame({"strategy": ["A", "B"], "iteration": [0, 0],
        ...                    "cost": [1.0, 2.0], "qaly": [0.5, 0.7]})
        >>> as_outcomes(df).strategies
        ['A', 'B']
    """
    if isinstance(source, Outcomes):
        return source
    if isinstance(source, (str, Path)):
        source = pd.read_csv(source)
    return Outcomes.from_tidy(
        source, strategy=strategy, iteration=iteration, cost=cost, effect=effect
    )


def _evaluate(model: ModelEngine | ModelFn, draws: pd.DataFrame) -> Outcomes:
    if isinstance(model, ModelEngine):
        return model.evaluate(draws)
    if callable(model):
        return model(draws)
    raise TypeError(
        "model must implement the ModelEngine protocol or be a callable draws -> Outcomes."
    )


def run_psa(
    model: ModelEngine | ModelFn,
    draws: pd.DataFrame,
    *,
    n_jobs: int = 1,
    batch_size: int | None = None,
) -> Outcomes:
    """Evaluate a model over the parameter draw matrix, preserving its index.

    The draw matrix's index is the canonical iteration index: the outcomes
    returned here carry exactly that index, keeping the parameter/outcome
    linkage intact for value-of-information analysis.

    Args:
        model: A `ModelEngine` or a callable
            ``draws -> Outcomes``.
        draws: Parameter draw matrix (rows = iterations), e.g. from
            `heval.params.ParameterSet.sample`.
        n_jobs: ``joblib`` worker count; 1 runs in-process.
        batch_size: Rows per parallel batch (default: split evenly across
            workers).

    Returns:
        Outcomes with iteration index equal to ``draws.index``.

    Example:
        >>> import pandas as pd
        >>> from heval.params import Normal, ParameterSet
        >>> from heval.models import Outcomes
        >>> from heval.run import run_psa
        >>> def model(d: pd.DataFrame) -> Outcomes:
        ...     costs = pd.DataFrame({"A": d["c"], "B": d["c"] + 10})
        ...     effects = pd.DataFrame({"A": d["c"] * 0, "B": d["c"] * 0 + 0.1})
        ...     return Outcomes.from_wide(costs, effects)
        >>> draws = ParameterSet({"c": Normal(100, 5)}).sample(50, seed=1)
        >>> run_psa(model, draws).n_iterations
        50
    """
    if draws.empty:
        raise ValueError("draws is empty.")
    if draws.index.duplicated().any():
        raise ValueError("draws index (the iteration index) must be unique.")

    if n_jobs == 1:
        result = _evaluate(model, draws)
    else:
        if batch_size is None:
            n_batches = max(1, min(len(draws), abs(n_jobs) * 4))
        else:
            n_batches = max(1, int(np.ceil(len(draws) / batch_size)))
        splits = np.array_split(np.arange(len(draws)), n_batches)
        batches = [draws.iloc[ix] for ix in splits if len(ix)]
        partials: list[Outcomes] = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate)(model, batch) for batch in batches
        )
        strategies = partials[0].strategies
        effect = partials[0].effect
        for p in partials[1:]:
            if p.strategies != strategies:
                raise ValueError("Model returned inconsistent strategies across batches.")
        data = pd.concat([p.data for p in partials])
        full_index = pd.MultiIndex.from_product(
            [strategies, draws.index], names=[STRATEGY_LEVEL, ITERATION_LEVEL]
        )
        result = Outcomes(data.reindex(full_index), effect=effect)

    returned = pd.Index(result.iterations)
    if not returned.equals(pd.Index(draws.index)):
        raise ValueError(
            "Model violated the output contract: outcome iteration index does not "
            "match the parameter draw index."
        )
    return result
