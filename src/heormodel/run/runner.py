"""The run loop: drive a model over parameter draws, or ingest external outputs.

Two first-class entry points:

- `run_psa` evaluates a model engine (or plain function) over the
  parameter draw matrix, optionally in parallel via ``joblib``, and
  guarantees the returned outcomes carry the draws' iteration index.
- `as_outcomes` normalises a bring-your-own-outputs table (from
  any external simulator or spreadsheet export) into the standard
  `Outcomes` structure so it can flow straight
  into `heormodel.cea` and `heormodel.voi` without touching an engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from heormodel.models.outcomes import ITERATION_LEVEL, STRATEGY_LEVEL, Outcomes
from heormodel.models.protocol import ModelEngine, ModelFn
from heormodel.run._progress import ProgressReporter, resolve_enabled


def as_outcomes(
    source: Outcomes | pd.DataFrame | str | Path,
    *,
    strategy: str = "strategy",
    iteration: str = "iteration",
    cost: str = "cost",
    effect: str = "qaly",
) -> Outcomes:
    """Normalise any costs/effects table into the standard outcome structure.

    This is the bring-your-own-outputs entry point: feed a tidy table from
    any source directly into the analysis layer.

    Args:
        source: An `Outcomes` (returned unchanged), a tidy long
            ``DataFrame``, or a path to a CSV file of one.
        strategy: Column holding the strategy label.
        iteration: Column holding the iteration index.
        cost: Column holding the cost per iteration.
        effect: Column holding the effect (QALYs by default).

    Example:
        >>> import pandas as pd
        >>> from heormodel.run import as_outcomes
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


def _split_batches(draws: pd.DataFrame, workers: int, batch_size: int | None) -> list[pd.DataFrame]:
    """Split draws into experiments (batches), preserving row order."""
    if batch_size is None:
        n_batches = max(1, min(len(draws), abs(workers) * 4))
    else:
        n_batches = max(1, int(np.ceil(len(draws) / batch_size)))
    splits = np.array_split(np.arange(len(draws)), n_batches)
    return [draws.iloc[ix] for ix in splits if len(ix)]


def _reassemble(partials: list[Outcomes], draws: pd.DataFrame) -> Outcomes:
    """Stitch per-batch outcomes back into one panel on the draw index."""
    strategies = partials[0].strategies
    effect = partials[0].effect
    for p in partials[1:]:
        if p.strategies != strategies:
            raise ValueError("Model returned inconsistent strategies across batches.")
    data = pd.concat([p.data for p in partials])
    if data.index.duplicated().any():
        raise ValueError(
            "Model violated the output contract: a batch returned outcomes whose "
            "iteration index does not match its input draws."
        )
    full_index = pd.MultiIndex.from_product(
        [strategies, draws.index], names=[STRATEGY_LEVEL, ITERATION_LEVEL]
    )
    return Outcomes(data.reindex(full_index), effect=effect)


class _ProgressParallel(Parallel):  # type: ignore[misc]
    """``joblib.Parallel`` that reports each batch as it completes.

    ``joblib`` calls `print_progress` from the parent process after every
    completed task, so the reporter advances as results return rather than
    waiting for the whole pool.
    """

    def __init__(self, reporter: ProgressReporter, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._reporter = reporter

    def print_progress(self) -> None:
        self._reporter.advance(self.n_completed_tasks)


def run_psa(
    model: ModelEngine | ModelFn,
    draws: pd.DataFrame,
    *,
    n_jobs: int = -1,
    sequential: bool = False,
    batch_size: int | None = None,
    progress: bool | None = None,
) -> Outcomes:
    """Evaluate a model over the parameter draw matrix, preserving its index.

    The draw matrix's index is the canonical iteration index: the outcomes
    returned here carry exactly that index, keeping the parameter/outcome
    linkage intact for value-of-information analysis.

    The run is parallel by default. Because each iteration draws a stream
    keyed by its index, the numbers are identical whether the run is
    parallel or sequential, and whatever the batch size; splitting only
    changes how work is dispatched, not the result.

    Args:
        model: A `ModelEngine` or a callable
            ``draws -> Outcomes``.
        draws: Parameter draw matrix (rows = iterations), e.g. from
            `heormodel.params.ParameterSet.sample`.
        n_jobs: ``joblib`` worker count; ``-1`` (default) uses all cores.
        sequential: Run in-process on one worker, the readable off switch
            for debugging and reproducibility checks. Forces sequential
            whatever ``n_jobs`` says. The run also falls back to sequential
            when there is one iteration or one available core.
        batch_size: Rows per experiment (default: split evenly across
            workers, four batches per worker). One experiment is one unit of
            work the loop dispatches, and the progress readout counts these.
        progress: Show a completed-count and time-remaining readout on
            ``stderr`` as experiments finish. ``None`` (default) shows it
            when ``stderr`` is a terminal and stays quiet otherwise, so CI
            logs and docs builds are silent; ``True`` forces it on, ``False``
            off. The remaining-time estimate uses only finished experiments,
            so early estimates are noisy and sharpen over the run.

    Returns:
        Outcomes with iteration index equal to ``draws.index``.

    Example:
        >>> import pandas as pd
        >>> from heormodel.params import Normal, ParameterSet
        >>> from heormodel.models import Outcomes
        >>> from heormodel.run import run_psa
        >>> def model(d: pd.DataFrame) -> Outcomes:
        ...     costs = pd.DataFrame({"A": d["c"], "B": d["c"] + 10})
        ...     effects = pd.DataFrame({"A": d["c"] * 0, "B": d["c"] * 0 + 0.1})
        ...     return Outcomes.from_wide(costs, effects)
        >>> draws = ParameterSet({"c": Normal(100, 5)}).sample(50, seed=1)
        >>> run_psa(model, draws, sequential=True).n_iterations
        50
    """
    if draws.empty:
        raise ValueError("draws is empty.")
    if draws.index.duplicated().any():
        raise ValueError("draws index (the iteration index) must be unique.")

    run_sequential = sequential or len(draws) == 1 or effective_n_jobs(n_jobs) == 1
    workers = 1 if run_sequential else n_jobs
    batches = _split_batches(draws, workers, batch_size)

    reporter = ProgressReporter(len(batches), enabled=resolve_enabled(progress, sys.stderr))
    partials: list[Outcomes]
    try:
        if run_sequential:
            partials = []
            for done, batch in enumerate(batches, start=1):
                partials.append(_evaluate(model, batch))
                reporter.advance(done)
        else:
            partials = _ProgressParallel(reporter, n_jobs=n_jobs, batch_size=1)(
                delayed(_evaluate)(model, batch) for batch in batches
            )
    finally:
        reporter.close()

    result = partials[0] if len(partials) == 1 else _reassemble(partials, draws)

    returned = pd.Index(result.iterations)
    if not returned.equals(pd.Index(draws.index)):
        raise ValueError(
            "Model violated the output contract: outcome iteration index does not "
            "match the parameter draw index."
        )
    return result
