"""Reproducibility scaffolding: seed logging, parameter provenance, model card.

A `RunRecord` snapshots everything needed to reproduce and audit an
analysis (the root seed entropy, the parameter specification, the outcome
dimensions, and package versions) and renders it as JSON (for archives) or
a markdown model card (for reports).
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from heval.models.outcomes import Outcomes
from heval.params.sampling import ParameterSet
from heval.run.seeds import SeedManager

_TRACKED_PACKAGES = ("heval", "numpy", "scipy", "pandas", "joblib", "scikit-learn")


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


@dataclass
class RunRecord:
    """A reproducibility record for one analysis run.

    Example:
        >>> from heval.report import capture_run
        >>> from heval.run import SeedManager
        >>> rec = capture_run(seed=SeedManager(42), note="demo")
        >>> rec.seed_entropy
        42
    """

    created: str
    seed_entropy: int | None
    n_iterations: int | None
    strategies: list[str] | None
    parameters: dict[str, str] | None
    draw_sources: dict[str, str] | None = None
    versions: dict[str, str] = field(default_factory=_versions)
    note: str = ""

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialise to JSON; optionally also write it to ``path``.

        Example:
            >>> from heval.report import capture_run
            >>> '"note": "x"' in capture_run(note="x").to_json()
            True
        """
        text = json.dumps(asdict(self), indent=2)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, text: str) -> RunRecord:
        """Rebuild a record from `to_json` output."""
        return cls(**json.loads(text))

    def model_card(self, title: str = "Model card") -> str:
        """Render a markdown model card for reports and repositories.

        Example:
            >>> from heval.report import capture_run
            >>> capture_run().model_card().startswith("# Model card")
            True
        """
        lines = [f"# {title}", "", f"- **Created:** {self.created}"]
        if self.note:
            lines.append(f"- **Note:** {self.note}")
        seed = "not recorded" if self.seed_entropy is None else str(self.seed_entropy)
        lines.append(f"- **Root seed entropy:** {seed}")
        if self.n_iterations is not None:
            lines.append(f"- **PSA iterations:** {self.n_iterations}")
        if self.strategies:
            lines.append(f"- **Strategies:** {', '.join(self.strategies)}")
        if self.parameters:
            lines += ["", "## Parameters", "", "| Parameter | Distribution |", "|---|---|"]
            lines += [f"| {name} | `{spec}` |" for name, spec in self.parameters.items()]
        if self.draw_sources:
            lines += ["", "## Draw sources", "", "| Parameter | Source |", "|---|---|"]
            lines += [f"| {name} | {src} |" for name, src in self.draw_sources.items()]
        lines += ["", "## Software versions", "", "| Package | Version |", "|---|---|"]
        lines += [f"| {pkg} | {ver} |" for pkg, ver in self.versions.items()]
        return "\n".join(lines) + "\n"


def capture_run(
    *,
    seed: SeedManager | int | None = None,
    params: ParameterSet | None = None,
    outcomes: Outcomes | None = None,
    draw_sources: Mapping[str, str] | None = None,
    note: str = "",
) -> RunRecord:
    """Snapshot a run's provenance into a `RunRecord`.

    Args:
        seed: The run's `SeedManager` (or raw seed).
        params: The sampled `ParameterSet`, if any.
        outcomes: The resulting outcomes, if available.
        draw_sources: Where each parameter's draws came from, for analyses
            that mix sources with `heval.params.mix_draws` (for example,
            ``{"beta": "ABC posterior", "u_healthy": "literature"}``).
        note: Free-text description of the analysis.

    Example:
        >>> from heval.report import capture_run
        >>> capture_run(seed=7).seed_entropy
        7
    """
    if isinstance(seed, SeedManager):
        entropy: int | None = seed.entropy
    elif seed is not None:
        entropy = int(seed)
    else:
        entropy = None
    return RunRecord(
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        seed_entropy=entropy,
        n_iterations=outcomes.n_iterations if outcomes is not None else None,
        strategies=outcomes.strategies if outcomes is not None else None,
        parameters=params.spec() if params is not None else None,
        draw_sources=dict(draw_sources) if draw_sources is not None else None,
        note=note,
    )
