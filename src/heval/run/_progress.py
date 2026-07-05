"""Minimal time-remaining reporter for the PSA run loop.

Writes a single-line completed-count and time-remaining readout to
``stderr`` as experiments finish. An experiment is one unit of work the
run loop dispatches: a batch, or an iteration when unbatched. The
remaining-time estimate uses only finished experiments, so early
estimates are noisy and sharpen as the run proceeds.

This module has no third-party dependency; a stderr writer is enough.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from datetime import timedelta
from typing import TextIO


def format_duration(seconds: float) -> str:
    """Format a duration in whole seconds as ``H:MM:SS``.

    Args:
        seconds: A non-negative duration. Fractions round down.

    Example:
        >>> from heval.run._progress import format_duration
        >>> format_duration(26.4)
        '0:00:26'
        >>> format_duration(3725)
        '1:02:05'
    """
    return str(timedelta(seconds=int(max(0.0, seconds))))


def resolve_enabled(progress: bool | None, stream: TextIO) -> bool:
    """Decide whether the reporter prints.

    ``None`` (the default) auto-detects: on when ``stream`` is a TTY, off
    otherwise, so CI logs and docs builds stay quiet. ``True`` forces the
    readout on regardless; ``False`` always silences it.

    Args:
        progress: The caller's ``progress`` argument.
        stream: The stream the reporter would write to.

    Example:
        >>> import io
        >>> from heval.run._progress import resolve_enabled
        >>> resolve_enabled(False, io.StringIO())
        False
        >>> resolve_enabled(True, io.StringIO())
        True
    """
    if progress is None:
        isatty = getattr(stream, "isatty", None)
        return bool(isatty()) if callable(isatty) else False
    return bool(progress)


class ProgressReporter:
    """Report completed experiments and an estimate of time remaining.

    The reporter overwrites one line on the stream while the run proceeds
    and terminates it with a newline on `close`. When disabled it writes
    nothing.

    Args:
        total: Number of experiments the run will dispatch.
        enabled: Whether to write anything.
        label: Line prefix, the run function's name by default.
        stream: Where to write; ``stderr`` by default.
        clock: Monotonic time source, injectable for testing.

    Example:
        >>> import io
        >>> from heval.run._progress import ProgressReporter
        >>> t = iter([0.0, 12.0])
        >>> buf = io.StringIO()
        >>> r = ProgressReporter(1000, enabled=True, stream=buf, clock=lambda: next(t))
        >>> r.advance(320)
        >>> buf.getvalue().strip()
        'running_psa: 320/1000 experiments, 0:00:12 elapsed, ~0:00:25 remaining'
        >>> r.remaining_seconds
        25.5
    """

    def __init__(
        self,
        total: int,
        *,
        enabled: bool,
        label: str = "running_psa",
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total = max(0, int(total))
        self.enabled = enabled
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self._clock = clock
        self.completed = 0
        self.remaining_seconds = float("inf")
        self._start = self._clock()

    def advance(self, completed: int) -> None:
        """Record ``completed`` finished experiments and redraw the line.

        ``completed`` is the cumulative count, not an increment, so it may
        be passed straight from a worker-pool completion hook.
        """
        self.completed = max(0, min(int(completed), self.total))
        elapsed = self._clock() - self._start
        if self.completed > 0:
            per = elapsed / self.completed
            self.remaining_seconds = per * (self.total - self.completed)
        else:
            self.remaining_seconds = float("inf")
        if not self.enabled:
            return
        remaining = "?:??:??" if self.completed == 0 else format_duration(self.remaining_seconds)
        line = (
            f"{self.label}: {self.completed}/{self.total} experiments, "
            f"{format_duration(elapsed)} elapsed, ~{remaining} remaining"
        )
        self.stream.write("\r" + line)
        self.stream.flush()

    def close(self) -> None:
        """Terminate the readout line, if anything was written."""
        if self.enabled and self.completed > 0:
            self.stream.write("\n")
            self.stream.flush()
