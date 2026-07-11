"""The model's states, strategies, base-case values, and probabilistic parameters.

Each is a function that returns its value, so the run script passes them to the
model building blocks rather than reading module-level constants.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from heormodel.params import Beta, Fixed, Gamma, LogNormal, ParameterSet


def states() -> tuple[str, ...]:
    """The four model states in order: Healthy, Sick, Sicker, Dead."""
    return ("H", "S1", "S2", "D")


def strategies() -> dict[str, dict[str, float]]:
    """The four strategies as the treatment flags the model functions read.

    Treatment A raises the Sick utility; treatment B slows the Sick-to-Sicker
    progression. Each strategy sets the two flags the event-time and valuation
    functions read.
    """
    return {
        "Standard of care": {"trtA": 0.0, "trtB": 0.0},
        "Strategy A": {"trtA": 1.0, "trtB": 0.0},
        "Strategy B": {"trtA": 0.0, "trtB": 1.0},
        "Strategy AB": {"trtA": 1.0, "trtB": 1.0},
    }


def base_case() -> dict[str, float]:
    """The article's Table 1 point estimates.

    ``c_trtB`` follows the text and the distribution mean, $13,000; the table's
    $12,000 entry is a misprint. Treatment A raises the Sick utility by a fixed
    0.20, so there is no separate treated-utility base value.
    """
    return dict(
        r_HS1=0.15, r_S1H=0.5, r_S1S2_scale=0.08, r_S1S2_shape=1.1,
        hr_S1=3.0, hr_S2=10.0, hr_S1S2_trtB=0.6,
        c_H=2000.0, c_S1=4000.0, c_S2=15000.0, c_trtA=12000.0, c_trtB=13000.0,
        u_H=1.0, u_S1=0.75, u_S2=0.5,
        du_HS1=0.01, ic_HS1=1000.0, ic_D=2000.0,
    )


def parameter_set() -> ParameterSet:
    """The probabilistic parameters, matching the companion code's active draws.

    The companion draws all of Table 1 but merges six columns into its model
    under names the model does not read, so it runs them at base case: the
    Weibull progression scale, both treatment costs, both transition costs, and
    the treated-utility increment (Sick utility is ``u_S1 + 0.20`` regardless of
    the drawn value). Those six are held fixed here. With the scale fixed, only
    the Weibull shape varies, so the progression hazard barely moves across
    draws and no scale-shape correlation applies.
    """
    distributions: dict[str, Any] = {
        "r_HS1": Gamma(30.0, 1.0 / 200.0),
        "r_S1H": Gamma(60.0, 1.0 / 120.0),
        "r_S1S2_scale": Fixed(0.08),
        "r_S1S2_shape": LogNormal.from_mean_se(1.10, 0.05),
        "hr_S1": LogNormal(np.log(3.0), 0.01),
        "hr_S2": LogNormal(np.log(10.0), 0.02),
        "hr_S1S2_trtB": LogNormal(np.log(0.6), 0.02),
        "c_H": Gamma(100.0, 20.0),
        "c_S1": Gamma(177.8, 22.5),
        "c_S2": Gamma(225.0, 66.7),
        "c_trtA": Fixed(12000.0),
        "c_trtB": Fixed(13000.0),
        "u_H": Beta(200.0, 3.0),
        "u_S1": Beta(130.0, 45.0),
        "u_S2": Beta(230.0, 230.0),
        "du_HS1": Beta(11.0, 1088.0),
        "ic_HS1": Fixed(1000.0),
        "ic_D": Fixed(2000.0),
    }
    return ParameterSet(distributions)
