"""Standard cost-effectiveness plots (matplotlib).

Every function draws on a provided or fresh ``Axes`` and returns it, so
plots compose into user figures. Styling follows a consistent scheme: a
fixed-order colorblind-validated categorical palette (one hue per strategy,
assigned in order, never cycled by rank), thin marks, and recessive grids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from heval.cea.ceac import ce_plane as _ce_plane_data
from heval.cea.frontier import STATUS_ND, icer_table
from heval.cea.nb import nmb
from heval.models.outcomes import Outcomes

#: Fixed-order categorical palette (colorblind-validated, light surface).
PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)


def _style(ax: Axes) -> None:
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def strategy_colors(strategies: list[str]) -> dict[str, str]:
    """Stable strategy -> color assignment in fixed palette order.

    Example:
        >>> from heval.report import strategy_colors
        >>> strategy_colors(["A", "B"])["A"]
        '#2a78d6'
    """
    if len(strategies) > len(PALETTE):
        raise ValueError(
            f"More than {len(PALETTE)} strategies; group the rest into 'Other' or "
            "use small multiples rather than extending the palette."
        )
    return {s: PALETTE[i] for i, s in enumerate(strategies)}


def plot_ce_plane(
    outcomes: Outcomes,
    *,
    comparator: str | None = None,
    wtp: float | None = None,
    effect: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Scatter of incremental cost vs incremental effect per PSA iteration.

    Args:
        outcomes: Standard PSA outcomes.
        comparator: Reference strategy (default: the first).
        wtp: If given, draw the willingness-to-pay threshold line.
        effect: Effect column (default: primary).
        ax: Existing axes to draw on.

    Example:
        >>> import pandas as pd
        >>> from heval.models import Outcomes
        >>> from heval.report import plot_ce_plane
        >>> c = pd.DataFrame({"A": [0.0, 1.0], "B": [5.0, 6.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.1], "B": [0.5, 0.4]})
        >>> ax = plot_ce_plane(Outcomes.from_wide(c, e))
        >>> ax.get_xlabel()
        'Incremental effect (qaly)'
    """
    ax = ax or plt.subplots()[1]
    data = _ce_plane_data(outcomes, comparator=comparator, effect=effect)
    colors = strategy_colors(outcomes.strategies)
    for s, grp in data.groupby("strategy", sort=False):
        ax.scatter(
            grp["inc_effect"],
            grp["inc_cost"],
            s=9,
            alpha=0.35,
            lw=0,
            color=colors[str(s)],
            label=str(s),
        )
    if wtp is not None:
        xs = np.array(ax.get_xlim())
        ax.plot(xs, wtp * xs, color="0.4", lw=1.2, ls="--", label=f"WTP = {wtp:g}")
        ax.set_xlim(*xs)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel(f"Incremental effect ({effect or outcomes.effect})")
    ax.set_ylabel("Incremental cost")
    ref = comparator or outcomes.strategies[0]
    ax.set_title(f"Cost-effectiveness plane vs {ref}")
    ax.legend(frameon=False)
    _style(ax)
    return ax


def plot_ceac(
    ceac_df: pd.DataFrame,
    *,
    ceaf_df: pd.DataFrame | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Cost-effectiveness acceptability curves, optionally with the frontier.

    Args:
        ceac_df: Output of `heval.cea.ceac`.
        ceaf_df: Optional output of `heval.cea.ceaf`; drawn as a bold
            trace over the optimal strategy's curve.
        ax: Existing axes to draw on.

    Example:
        >>> import pandas as pd
        >>> from heval.cea import ceac
        >>> from heval.models import Outcomes
        >>> from heval.report import plot_ceac
        >>> c = pd.DataFrame({"A": [0.0, 0.0], "B": [5.0, 5.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, -1.0]})
        >>> curve = ceac(Outcomes.from_wide(c, e), wtp=[0.0, 10.0, 20.0])
        >>> ax = plot_ceac(curve)
        >>> ax.get_ylabel()
        'Probability cost-effective'
    """
    ax = ax or plt.subplots()[1]
    colors = strategy_colors([str(c) for c in ceac_df.columns])
    for s in ceac_df.columns:
        ax.plot(ceac_df.index, ceac_df[s], lw=1.8, color=colors[str(s)], label=str(s))
    if ceaf_df is not None:
        ax.plot(
            ceaf_df.index,
            ceaf_df["prob"],
            lw=3.5,
            color="0.15",
            alpha=0.35,
            solid_capstyle="round",
            label="Frontier (CEAF)",
            zorder=1,
        )
    ax.set_xlabel("Willingness to pay")
    ax.set_ylabel("Probability cost-effective")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Cost-effectiveness acceptability")
    ax.legend(frameon=False)
    _style(ax)
    return ax


def plot_frontier(
    source: Outcomes | pd.DataFrame,
    *,
    effect: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Mean cost vs mean effect per strategy with the efficiency frontier.

    Frontier strategies are connected; dominated (D) and extendedly
    dominated (ED) strategies are shown hollow and annotated.

    Example:
        >>> import pandas as pd
        >>> from heval.report import plot_frontier
        >>> means = pd.DataFrame(
        ...     {"cost": [0.0, 100.0], "effect": [0.0, 1.0]}, index=["A", "B"])
        >>> ax = plot_frontier(means)
        >>> ax.get_ylabel()
        'Mean cost'
    """
    ax = ax or plt.subplots()[1]
    effect_name = effect or (source.effect if isinstance(source, Outcomes) else "effect")
    table = icer_table(source, effect=effect)
    colors = strategy_colors([str(s) for s in table.index])
    on = table[table["status"] == STATUS_ND]
    ax.plot(on["effect"], on["cost"], color="0.55", lw=1.4, zorder=1)
    for s, row in table.iterrows():
        nd = row["status"] == STATUS_ND
        ax.scatter(
            row["effect"],
            row["cost"],
            s=55,
            zorder=2,
            color=colors[str(s)],
            facecolors=colors[str(s)] if nd else "white",
            linewidths=1.6,
        )
        label = str(s) if nd else f"{s} ({row['status']})"
        ax.annotate(
            label,
            (row["effect"], row["cost"]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=9,
            color="0.25",
        )
    ax.set_xlabel(f"Mean effect ({effect_name})")
    ax.set_ylabel("Mean cost")
    ax.set_title("Efficiency frontier")
    _style(ax)
    return ax


def tornado_data(
    outcomes: Outcomes,
    draws: pd.DataFrame,
    wtp: float,
    *,
    strategy: str | None = None,
    comparator: str | None = None,
    effect: str | None = None,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    """One-way sensitivity of NMB to each parameter, estimated from the PSA.

    For each parameter, fits a univariate linear regression of the
    strategy's NMB (or incremental NMB versus ``comparator``) on that
    parameter and evaluates it at the parameter's outer quantiles. This is
    a PSA-based analogue of a deterministic one-way analysis.

    Returns:
        DataFrame indexed by parameter with columns ``low``, ``high`` and
        ``span``, sorted by descending span.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heval.models import Outcomes
        >>> from heval.report import tornado_data
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=500)
        >>> draws = pd.DataFrame({"x": x}, index=pd.RangeIndex(500, name="iteration"))
        >>> out = Outcomes.from_wide(
        ...     pd.DataFrame({"A": -x}), pd.DataFrame({"A": np.zeros(500)}))
        >>> td = tornado_data(out, draws, wtp=1.0, strategy="A")
        >>> td.index[0]
        'x'
    """
    if not pd.Index(draws.index).equals(pd.Index(outcomes.iterations)):
        raise ValueError("draws index must equal the outcomes iteration index.")
    nb = nmb(outcomes, wtp, effect=effect)
    target = strategy or outcomes.strategies[-1]
    y = nb[target]
    if comparator is not None:
        y = y - nb[comparator]
    yv = y.to_numpy(dtype=np.float64)
    rows = {}
    for p in draws.columns:
        x = draws[p].to_numpy(dtype=np.float64)
        if np.ptp(x) == 0.0:
            rows[p] = (float(yv.mean()), float(yv.mean()))
            continue
        slope, intercept = np.polyfit(x, yv, 1)
        lo_x, hi_x = np.quantile(x, quantiles)
        rows[p] = (intercept + slope * lo_x, intercept + slope * hi_x)
    td = pd.DataFrame(rows, index=["low", "high"]).T
    td.index.name = "parameter"
    td["span"] = (td["high"] - td["low"]).abs()
    return td.sort_values("span", ascending=False)


def plot_tornado(td: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Tornado diagram from `tornado_data`.

    Example:
        >>> import pandas as pd
        >>> from heval.report import plot_tornado
        >>> td = pd.DataFrame({"low": [0.0], "high": [2.0], "span": [2.0]},
        ...                   index=pd.Index(["x"], name="parameter"))
        >>> ax = plot_tornado(td)
        >>> ax.get_xlabel()
        'Net monetary benefit'
    """
    ax = ax or plt.subplots()[1]
    ordered = td.sort_values("span")  # widest bar on top
    base = float((ordered["low"] + ordered["high"]).mean() / 2)
    y = np.arange(len(ordered))
    left = ordered[["low", "high"]].min(axis=1).to_numpy(dtype=np.float64)
    width = ordered["span"].to_numpy(dtype=np.float64)
    ax.barh(y, width, left=left, height=0.55, color=PALETTE[0], alpha=0.85)
    ax.axvline(base, color="0.4", lw=1.0, ls="--")
    ax.set_yticks(y, [str(i) for i in ordered.index])
    ax.set_xlabel("Net monetary benefit")
    ax.set_title("Tornado diagram")
    _style(ax)
    ax.grid(axis="y", visible=False)
    return ax
