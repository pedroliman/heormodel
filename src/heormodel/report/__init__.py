"""Reporting (`heval.report`).

Standard cost-effectiveness plots (CE plane, CEAC/CEAF, efficiency
frontier, tornado) and reproducibility scaffolding (seed logging,
parameter provenance, run reports).
"""

from heormodel.report.plots import (
    PALETTE,
    heatmap_data,
    plot_ce_plane,
    plot_ceac,
    plot_expected_loss,
    plot_frontier,
    plot_tornado,
    strategy_colors,
    tornado_data,
)
from heormodel.report.provenance import RunRecord, capture_run

__all__ = [
    "PALETTE",
    "RunRecord",
    "capture_run",
    "heatmap_data",
    "plot_ce_plane",
    "plot_ceac",
    "plot_expected_loss",
    "plot_frontier",
    "plot_tornado",
    "strategy_colors",
    "tornado_data",
]
