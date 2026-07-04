"""Reporting (`heval.report`).

Standard cost-effectiveness plots (CE plane, CEAC/CEAF, efficiency
frontier, tornado) and reproducibility scaffolding (seed logging,
parameter provenance, run reports).
"""

from heval.report.plots import (
    PALETTE,
    plot_ce_plane,
    plot_ceac,
    plot_frontier,
    plot_tornado,
    strategy_colors,
    tornado_data,
)
from heval.report.provenance import RunRecord, capture_run

__all__ = [
    "PALETTE",
    "RunRecord",
    "capture_run",
    "plot_ce_plane",
    "plot_ceac",
    "plot_frontier",
    "plot_tornado",
    "strategy_colors",
    "tornado_data",
]
