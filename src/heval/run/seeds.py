"""Reproducible seed management built on ``numpy.random.SeedSequence``.

A single `SeedManager` is the root of all randomness in a run:
parameter sampling, per-iteration engine streams, and any study-data
simulation for EVSI all derive from it, so one recorded entropy value
reproduces the whole analysis.
"""

from __future__ import annotations

import numpy as np


class SeedManager:
    """Root seed source that spawns independent child generators.

    Args:
        seed: Root seed. ``None`` draws fresh OS entropy; record
            `entropy` afterwards to reproduce the run.

    Example:
        >>> from heval.run import SeedManager
        >>> sm = SeedManager(42)
        >>> rng = sm.generator()
        >>> children = sm.spawn(3)
        >>> len(children)
        3
        >>> bool(SeedManager(42).spawn(3)[0].integers(100) == children[0].integers(100))
        True
    """

    def __init__(self, seed: int | None = None) -> None:
        self._sequence = np.random.SeedSequence(seed)
        self._spawned = 0

    @property
    def entropy(self) -> int:
        """The root entropy; persist this to reproduce the run exactly."""
        entropy = self._sequence.entropy
        assert entropy is not None
        return int(entropy)  # type: ignore[arg-type]

    def generator(self) -> np.random.Generator:
        """A generator for run-level randomness (e.g. parameter sampling)."""
        return np.random.default_rng(self._sequence)

    def spawn(self, n: int) -> list[np.random.Generator]:
        """Spawn ``n`` statistically independent child generators.

        Repeated calls continue the spawn sequence, so children never repeat
        within one manager.
        """
        children = self._sequence.spawn(n)
        self._spawned += n
        return [np.random.default_rng(child) for child in children]

    def __repr__(self) -> str:
        return f"SeedManager(entropy={self.entropy}, spawned={self._spawned})"
