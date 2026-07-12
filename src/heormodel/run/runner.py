"""The run loop: drive a model over parameter draws, or ingest external outputs.

Two first-class entry points:

- `run_psa` evaluates a model engine (or plain function) over the
  parameter draw matrix, optionally in parallel via ``joblib``, and
  returns a `RunResult` whose outcomes carry the draws' iteration index. It
  owns execution: it builds the per-iteration random streams from its ``seed``
  argument and gathers the optional event or individual logs, so engines stay
  seed-free descriptions of a model.
- `as_outcomes` normalises a bring-your-own-outputs table (from
  any external simulator or spreadsheet export) into the standard
  `Outcomes` structure so it can flow straight
  into `heormodel.cea` and `heormodel.voi` without touching an engine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from heormodel.models._engine import finalize_outcomes
from heormodel.models.outcomes import Outcomes
from heormodel.models.protocol import EngineResult, ModelEngine, ModelFn, StochasticEngine
from heormodel.run._progress import ProgressReporter, resolve_enabled
from heormodel.run.seeds import SeedManager


@dataclass(frozen=True)
class RunResult:
    """What `run_psa` returns: the outcomes panel plus any collected logs.

    ``outcomes`` is always present. ``events`` and ``individuals`` hold the
    optional log channels and are ``None`` unless ``collect`` asked for them.
    `RunResult` is deliberately not iterable, so it cannot be mistaken for the
    old ``(outcomes, trace)`` tuple.

    Args:
        outcomes: The `Outcomes` panel, indexed by ``(intervention, iteration)``.
        events: The state-change or resource history, or ``None``.
        individuals: Per-individual accruals, or ``None``.
    """

    outcomes: Outcomes
    events: pd.DataFrame | None = None
    individuals: pd.DataFrame | None = None


def as_outcomes(
    source: Outcomes | pd.DataFrame | str | Path,
    *,
    intervention: str = "intervention",
    iteration: str = "iteration",
    cost: str = "cost",
    effect: str = "qaly",
    comparator: str | None = None,
) -> Outcomes:
    """Normalise any costs/effects table into the standard outcome structure.

    This is the bring-your-own-outputs entry point: feed a tidy table from
    any source directly into the analysis layer.

    Args:
        source: An `Outcomes` (returned unchanged), a tidy long
            ``DataFrame``, or a path to a CSV file of one.
        intervention: Column holding the intervention label.
        iteration: Column holding the iteration index.
        cost: Column holding the cost per iteration.
        effect: Column holding the effect (QALYs by default).
        comparator: Name of the reference intervention, or ``None``. Ignored
            when ``source`` is already an `Outcomes`.

    Example:
        >>> import pandas as pd
        >>> from heormodel.run import as_outcomes
        >>> df = pd.DataFrame({"intervention": ["A", "B"], "iteration": [0, 0],
        ...                    "cost": [1.0, 2.0], "qaly": [0.5, 0.7]})
        >>> as_outcomes(df).interventions
        ['A', 'B']
    """
    if isinstance(source, Outcomes):
        return source
    if isinstance(source, (str, Path)):
        source = pd.read_csv(source)
    return Outcomes.from_tidy(
        source,
        intervention=intervention,
        iteration=iteration,
        cost=cost,
        effect=effect,
        comparator=comparator,
    )


def _evaluate(
    model: ModelEngine | ModelFn,
    draws: pd.DataFrame,
    streams: SeedManager,
    collect: str | None,
) -> EngineResult:
    if isinstance(model, StochasticEngine):
        return model.evaluate_streamed(draws, streams=streams, collect=collect)
    if collect is not None:
        raise ValueError(
            f"collect={collect!r} is only available for a stochastic engine; this model "
            "produces no event or individual log."
        )
    if isinstance(model, ModelEngine):
        return EngineResult(model.evaluate(draws))
    if callable(model):
        return EngineResult(model(draws))
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
    interventions = partials[0].interventions
    effect = partials[0].effect
    comparator = partials[0].comparator
    for p in partials[1:]:
        if p.interventions != interventions:
            raise ValueError("Model returned inconsistent interventions across batches.")
    data = pd.concat([p.data for p in partials])
    if data.index.duplicated().any():
        raise ValueError(
            "Model violated the output contract: a batch returned outcomes whose "
            "iteration index does not match its input draws."
        )
    return finalize_outcomes(data, interventions, draws.index, effect, comparator)


def _concat_logs(frames: list[pd.DataFrame | None]) -> pd.DataFrame | None:
    """Concatenate per-batch logs in batch (row) order, or None if unset."""
    present = [f for f in frames if f is not None]
    if not present:
        return None
    return pd.concat(present, ignore_index=True)


def _combine(partials: list[EngineResult], draws: pd.DataFrame) -> RunResult:
    """Combine per-batch engine results into one `RunResult`.

    The batches partition the draws in row order and return in that order, so
    concatenating their logs reconstructs the full row-major order whatever the
    batch boundaries, keeping the log invariant to worker count.
    """
    outcomes = (
        partials[0].outcomes
        if len(partials) == 1
        else _reassemble([p.outcomes for p in partials], draws)
    )
    return RunResult(
        outcomes=outcomes,
        events=_concat_logs([p.events for p in partials]),
        individuals=_concat_logs([p.individuals for p in partials]),
    )


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
    seed: int | None = None,
    collect: str | None = None,
    n_jobs: int = -1,
    sequential: bool = False,
    batch_size: int | None = None,
    progress: bool | None = None,
) -> RunResult:
    """Evaluate a model over the parameter draw matrix, preserving its index.

    This is the single execution point. It owns every execution concern: it
    builds the per-iteration random streams from ``seed`` and hands them to a
    stochastic engine, runs the batches in parallel, gathers any ``collect``
    log, and returns a `RunResult`. Engines themselves hold no seed and no side
    channel, so the same model object reruns under a new ``seed`` without
    reconstruction.

    The draw matrix's index is the canonical iteration index: the outcomes carry
    exactly that index, keeping the parameter/outcome linkage intact for
    value-of-information analysis.

    The run is parallel by default. Because each iteration draws a stream keyed
    by its index, the numbers, and any collected log, are identical whether the
    run is parallel or sequential, and whatever the batch size; splitting only
    changes how work is dispatched, not the result.

    Args:
        model: A `ModelEngine` or a callable ``draws -> Outcomes``.
        draws: Parameter draw matrix (rows = iterations), e.g. from
            `heormodel.params.ParameterSet.sample`.
        seed: Root seed for the per-iteration streams a stochastic engine uses.
            ``None`` (default) draws fresh entropy; pass an integer for a
            reproducible run. Ignored by deterministic engines.
        collect: ``None`` (default) returns outcomes only. ``"events"`` also
            gathers the state-change or resource history, and ``"individuals"``
            the per-individual accruals, into the `RunResult`. Only a stochastic
            engine produces these; asking a deterministic model for them raises.
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
        A `RunResult` whose ``outcomes`` iteration index equals ``draws.index``.

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
        >>> run_psa(model, draws, sequential=True).outcomes.n_iterations
        50
    """
    if draws.empty:
        raise ValueError("draws is empty.")
    if draws.index.duplicated().any():
        raise ValueError("draws index (the iteration index) must be unique.")

    streams = SeedManager(seed)
    run_sequential = sequential or len(draws) == 1 or effective_n_jobs(n_jobs) == 1
    workers = 1 if run_sequential else n_jobs
    batches = _split_batches(draws, workers, batch_size)

    reporter = ProgressReporter(len(batches), enabled=resolve_enabled(progress, sys.stderr))
    partials: list[EngineResult]
    try:
        if run_sequential:
            partials = []
            for done, batch in enumerate(batches, start=1):
                partials.append(_evaluate(model, batch, streams, collect))
                reporter.advance(done)
        else:
            partials = _ProgressParallel(reporter, n_jobs=n_jobs, batch_size=1)(
                delayed(_evaluate)(model, batch, streams, collect) for batch in batches
            )
    finally:
        reporter.close()

    result = _combine(partials, draws)

    returned = pd.Index(result.outcomes.iterations)
    if not returned.equals(pd.Index(draws.index)):
        raise ValueError(
            "Model violated the output contract: outcome iteration index does not "
            "match the parameter draw index."
        )
    return result
