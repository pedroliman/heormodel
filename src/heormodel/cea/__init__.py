"""Cost-effectiveness analysis (`heormodel.cea`).

Decision analysis on the standard outcome structure, whatever the engine:
incremental analysis with dominance and extended dominance, the efficiency
frontier, net monetary/health benefit, acceptability curves (CEAC), the
acceptability frontier (CEAF), expected loss curves, and
cost-effectiveness-plane data.
"""

from heormodel.cea.ceac import ce_plane, ceac, ceaf, expected_loss
from heormodel.cea.frontier import STATUS_D, STATUS_ED, STATUS_ND, frontier, icer_table
from heormodel.cea.nb import expected_nmb, nhb, nmb

__all__ = [
    "STATUS_D",
    "STATUS_ED",
    "STATUS_ND",
    "ce_plane",
    "ceac",
    "ceaf",
    "expected_loss",
    "expected_nmb",
    "frontier",
    "icer_table",
    "nhb",
    "nmb",
]
