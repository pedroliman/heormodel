"""Deterministic sensitivity analysis (`heormodel.dsa`).

Where probabilistic analysis answers how uncertain a decision is,
deterministic sensitivity analysis (DSA) answers which parameters move the
result and by how much. Both read the same model. This module builds
scenario designs that run through `heormodel.run.run_psa` unchanged: a design is
a draw matrix whose rows are scenarios instead of random draws, paired with a
descriptor table that names what each scenario varied.

Three builders cover the standard forms. `one_way` sweeps a single parameter
across a set of values. `one_at_a_time` sweeps each parameter in turn, the
union of one-way sweeps that feeds a tornado diagram. `grid` takes the full
factorial of two or more parameters, which feeds a two-way heatmap.
"""

from heormodel.dsa.design import Design, grid, one_at_a_time, one_way

__all__ = ["Design", "grid", "one_at_a_time", "one_way"]
