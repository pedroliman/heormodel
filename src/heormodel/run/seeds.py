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
        >>> from heormodel.run import SeedManager
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

    def child_sequence(self, key: int) -> np.random.SeedSequence:
        """A child seed sequence addressed by a stable integer ``key``.

        Unlike `spawn`, the returned sequence depends only on ``key`` and
        this manager's seed, not on call order. Keying by iteration index
        gives per-iteration streams that stay identical however a run is
        chunked across workers. Spawn from the sequence for sub-streams
        (population sampling, per-intervention randomness).

        Example:
            >>> from heormodel.run import SeedManager
            >>> import numpy as np
            >>> a = SeedManager(7).child_sequence(3)
            >>> b = SeedManager(7).child_sequence(3)
            >>> int(np.random.default_rng(a).integers(1_000_000)) == int(
            ...     np.random.default_rng(b).integers(1_000_000))
            True
        """
        return np.random.SeedSequence(
            entropy=self._sequence.entropy,
            spawn_key=(*self._sequence.spawn_key, int(key)),
        )

    def __repr__(self) -> str:
        return f"SeedManager(entropy={self.entropy}, spawned={self._spawned})"
