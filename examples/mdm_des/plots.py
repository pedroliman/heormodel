"""Plots specific to the discrete-event replication (the article's figure 3)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from heormodel.report import intervention_colors


def plot_epidemiology(
    survival: pd.DataFrame,
    prevalence: pd.DataFrame,
    *,
    interventions: Sequence[str],
    age_start: float,
    path: str | Path,
) -> None:
    """Draw the survival and prevalence panels (the article's figure 3A and 3B).

    Input: the survival and prevalence curves (time-indexed, one column per
    intervention), the intervention order, the starting age, and an output path.
    Output: writes a PNG to ``path``. Intervention A shares standard of care's
    dynamics and AB shares B's, so their curves coincide; dashes keep both
    members of each pair visible.
    """
    colors = intervention_colors(list(interventions))
    dashes = {"Intervention A": (4, 2), "Intervention AB": (4, 2)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    panels = (
        (axes[0], survival, "Survival"),
        (axes[1], prevalence, "Prevalence of S1 and S2"),
    )
    for ax, curves, title in panels:
        for name in curves.columns:
            ax.plot(
                age_start + curves.index, curves[name], lw=1.8, color=colors[name],
                label=name, dashes=dashes.get(name, (None, None)),
            )
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Proportion")
        ax.set_title(title)
        ax.grid(True, color="0.88", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
