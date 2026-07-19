"""Reporting (`heormodel.report`).

Standard cost-effectiveness plots (CE plane, CEAC/CEAF, efficiency
frontier, tornado) and reproducibility scaffolding (seed logging,
parameter provenance, run reports).
"""

from heormodel.report.plots import (
    PALETTE,
    heatmap_data,
    intervention_colors,
    plot_ce_plane,
    plot_ceac,
    plot_expected_loss,
    plot_frontier,
    plot_tornado,
    tornado_data,
)
from heormodel.report.provenance import RunRecord, capture_run
from heormodel.report.tables import format_icer_table

__all__ = [
    "PALETTE",
    "RunRecord",
    "capture_run",
    "format_icer_table",
    "heatmap_data",
    "plot_ce_plane",
    "plot_ceac",
    "plot_expected_loss",
    "plot_frontier",
    "plot_tornado",
    "intervention_colors",
    "tornado_data",
]
