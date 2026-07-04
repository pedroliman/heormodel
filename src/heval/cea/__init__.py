"""Cost-effectiveness analysis (`heval.cea`).

Engine-agnostic decision analysis on the standard outcome schema:
incremental analysis with dominance and extended dominance, the efficiency
frontier, net monetary/health benefit, acceptability curves (CEAC), the
acceptability frontier (CEAF), and cost-effectiveness-plane data.
"""

from heval.cea.ceac import ce_plane, ceac, ceaf
from heval.cea.frontier import STATUS_D, STATUS_ED, STATUS_ND, frontier, icer_table
from heval.cea.nb import expected_nmb, nhb, nmb

__all__ = [
    "STATUS_D",
    "STATUS_ED",
    "STATUS_ND",
    "ce_plane",
    "ceac",
    "ceaf",
    "expected_nmb",
    "frontier",
    "icer_table",
    "nhb",
    "nmb",
]
