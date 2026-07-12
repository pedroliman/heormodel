"""Standard cost-effectiveness plots (matplotlib).

Every function draws on a provided or fresh ``Axes`` and returns it, so
plots compose into user figures. Styling follows a consistent scheme: a
fixed-order colorblind-validated categorical palette (one hue per intervention,
assigned in order, never cycled by rank), thin marks, and recessive grids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from heormodel._util import require_shared_index
from heormodel.cea.ceac import ce_plane as _ce_plane_data
from heormodel.cea.frontier import STATUS_ND, icer_table
from heormodel.cea.nb import nmb
from heormodel.dsa.design import BASE_LABEL, PARAMETER_COL, SCENARIO_COL, VALUE_COL, Design
from heormodel.models.outcomes import Outcomes

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


def _plot_columns(df: pd.DataFrame, ax: Axes) -> None:
    """Draw one colored line per column over the shared index."""
    colors = intervention_colors([str(c) for c in df.columns])
    for name in df.columns:
        ax.plot(df.index, df[name], lw=1.8, color=colors[str(name)], label=str(name))


def _tornado_table(rows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """A (low, high) mapping as a tornado table sorted by descending span."""
    table = pd.DataFrame(rows, index=["low", "high"]).T
    table.index.name = "parameter"
    table["span"] = (table["high"] - table["low"]).abs()
    return table.sort_values("span", ascending=False)


def intervention_colors(interventions: list[str]) -> dict[str, str]:
    """Stable intervention -> color assignment in fixed palette order.

    Example:
        >>> from heormodel.report import intervention_colors
        >>> intervention_colors(["A", "B"])["A"]
        '#2a78d6'
    """
    if len(interventions) > len(PALETTE):
        raise ValueError(
            f"More than {len(PALETTE)} interventions; group the rest into 'Other' or "
            "use small multiples rather than extending the palette."
        )
    return {s: PALETTE[i] for i, s in enumerate(interventions)}


def plot_ce_plane(
    outcomes: Outcomes,
    *,
    comparator: str | None = None,
    wtp: float | None = None,
    effect: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Scatter of incremental cost vs incremental effect per iteration.

    Args:
        outcomes: Outcomes from a probabilistic sensitivity analysis.
        comparator: Reference intervention (default: ``outcomes.comparator``,
            or the first intervention if none was flagged).
        wtp: If given, draw the willingness-to-pay threshold line.
        effect: Effect column (default: primary).
        ax: Existing axes to draw on.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.report import plot_ce_plane
        >>> c = pd.DataFrame({"A": [0.0, 1.0], "B": [5.0, 6.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.1], "B": [0.5, 0.4]})
        >>> ax = plot_ce_plane(Outcomes.from_wide(c, e))
        >>> ax.get_xlabel()
        'Incremental effect (qaly)'
    """
    ax = ax or plt.subplots()[1]
    data = _ce_plane_data(outcomes, comparator=comparator, effect=effect)
    colors = intervention_colors(outcomes.interventions)
    for s, grp in data.groupby("intervention", sort=False):
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
    ref = comparator or outcomes.comparator or outcomes.interventions[0]
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
        ceac_df: Output of `heormodel.cea.ceac`.
        ceaf_df: Optional output of `heormodel.cea.ceaf`; drawn as a bold
            trace over the optimal intervention's curve.
        ax: Existing axes to draw on.

    Example:
        >>> import pandas as pd
        >>> from heormodel.cea import ceac
        >>> from heormodel.models import Outcomes
        >>> from heormodel.report import plot_ceac
        >>> c = pd.DataFrame({"A": [0.0, 0.0], "B": [5.0, 5.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, -1.0]})
        >>> curve = ceac(Outcomes.from_wide(c, e), wtp=[0.0, 10.0, 20.0])
        >>> ax = plot_ceac(curve)
        >>> ax.get_ylabel()
        'Probability cost-effective'
    """
    ax = ax or plt.subplots()[1]
    _plot_columns(ceac_df, ax)
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


def plot_expected_loss(
    loss_df: pd.DataFrame,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Expected loss curves, one per intervention over the willingness-to-pay grid.

    The lower envelope of the curves is the expected value of perfect
    information, so the plot shows the optimal intervention (lowest curve) and the
    cost of decision uncertainty at each threshold together.

    Args:
        loss_df: Output of `heormodel.cea.expected_loss`.
        ax: Existing axes to draw on.

    Example:
        >>> import pandas as pd
        >>> from heormodel.cea import expected_loss
        >>> from heormodel.models import Outcomes
        >>> from heormodel.report import plot_expected_loss
        >>> c = pd.DataFrame({"A": [0.0, 0.0], "B": [5.0, 5.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, -1.0]})
        >>> curves = expected_loss(Outcomes.from_wide(c, e), wtp=[0.0, 10.0, 20.0])
        >>> ax = plot_expected_loss(curves)
        >>> ax.get_ylabel()
        'Expected loss'
    """
    ax = ax or plt.subplots()[1]
    _plot_columns(loss_df, ax)
    ax.set_xlabel("Willingness to pay")
    ax.set_ylabel("Expected loss")
    ax.set_title("Expected loss curves")
    ax.legend(frameon=False)
    _style(ax)
    return ax


def plot_frontier(
    source: Outcomes | pd.DataFrame,
    *,
    effect: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Mean cost vs mean effect per intervention with the efficiency frontier.

    Frontier interventions are connected; dominated (D) and extendedly
    dominated (ED) interventions are shown hollow and annotated.

    Example:
        >>> import pandas as pd
        >>> from heormodel.report import plot_frontier
        >>> means = pd.DataFrame(
        ...     {"cost": [0.0, 100.0], "effect": [0.0, 1.0]}, index=["A", "B"])
        >>> ax = plot_frontier(means)
        >>> ax.get_ylabel()
        'Mean cost'
    """
    ax = ax or plt.subplots()[1]
    effect_name = effect or (source.effect if isinstance(source, Outcomes) else "effect")
    table = icer_table(source, effect=effect)
    colors = intervention_colors([str(s) for s in table.index])
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


def _target_nmb(
    outcomes: Outcomes,
    wtp: float,
    *,
    intervention: str | None,
    comparator: str | None,
    effect: str | None,
) -> pd.Series:
    """NMB of the target intervention (incremental if a comparator is given)."""
    nb = nmb(outcomes, wtp, effect=effect)
    target = intervention or outcomes.interventions[-1]
    y = nb[target]
    if comparator is not None:
        y = y - nb[comparator]
    return y


def _tornado_from_dsa(
    outcomes: Outcomes,
    descriptor: pd.DataFrame,
    wtp: float,
    *,
    intervention: str | None,
    comparator: str | None,
    effect: str | None,
) -> pd.DataFrame:
    """Tornado table from a one-way or one-at-a-time DSA descriptor."""
    if PARAMETER_COL not in descriptor.columns or VALUE_COL not in descriptor.columns:
        raise ValueError(
            "DSA descriptor must carry 'parameter' and 'value' columns; pass a one-way "
            "or one-at-a-time design, not a grid."
        )
    require_shared_index(descriptor.index, outcomes.iterations, "descriptor")
    y = _target_nmb(outcomes, wtp, intervention=intervention, comparator=comparator, effect=effect)
    d = descriptor.copy()
    d["_nmb"] = y.reindex(d.index).to_numpy(dtype=np.float64)
    swept = d[d[PARAMETER_COL] != BASE_LABEL]
    rows = {}
    for p, grp in swept.groupby(PARAMETER_COL, sort=False):
        ordered = grp.sort_values(VALUE_COL)
        rows[str(p)] = (float(ordered["_nmb"].iloc[0]), float(ordered["_nmb"].iloc[-1]))
    return _tornado_table(rows)


def tornado_data(
    outcomes: Outcomes,
    draws: pd.DataFrame | Design,
    wtp: float,
    *,
    intervention: str | None = None,
    comparator: str | None = None,
    effect: str | None = None,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    """One-way sensitivity of net monetary benefit, probabilistic or deterministic.

    With a parameter draw matrix (the probabilistic path), fits a univariate linear
    regression of the intervention's NMB (or incremental NMB versus
    ``comparator``) on each parameter and evaluates it at the parameter's
    outer ``quantiles``. This estimates a one-way analysis from the probabilistic draws.

    With a `heormodel.dsa` ``(design, descriptor)`` pair from `one_way` or
    `one_at_a_time` (the DSA path), reads the NMB at each parameter's lowest
    and highest swept value directly, ignoring ``quantiles``.

    Returns:
        DataFrame indexed by parameter with columns ``low``, ``high`` and
        ``span``, sorted by descending span.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.report import tornado_data
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=500)
        >>> draws = pd.DataFrame({"x": x}, index=pd.RangeIndex(500, name="iteration"))
        >>> out = Outcomes.from_wide(
        ...     pd.DataFrame({"A": -x}), pd.DataFrame({"A": np.zeros(500)}))
        >>> td = tornado_data(out, draws, wtp=1.0, intervention="A")
        >>> td.index[0]
        'x'
    """
    if isinstance(draws, tuple):
        _, descriptor = draws
        return _tornado_from_dsa(
            outcomes, descriptor, wtp,
            intervention=intervention, comparator=comparator, effect=effect,
        )
    require_shared_index(draws.index, outcomes.iterations, "draws")
    yv = _target_nmb(
        outcomes, wtp, intervention=intervention, comparator=comparator, effect=effect
    ).to_numpy(dtype=np.float64)
    rows = {}
    for p in draws.columns:
        x = draws[p].to_numpy(dtype=np.float64)
        if np.ptp(x) == 0.0:
            rows[p] = (float(yv.mean()), float(yv.mean()))
            continue
        slope, intercept = np.polyfit(x, yv, 1)
        lo_x, hi_x = np.quantile(x, quantiles)
        rows[p] = (intercept + slope * lo_x, intercept + slope * hi_x)
    return _tornado_table(rows)


def plot_tornado(td: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Tornado diagram from `tornado_data`.

    Example:
        >>> import pandas as pd
        >>> from heormodel.report import plot_tornado
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


def heatmap_data(
    values: pd.Series,
    descriptor: pd.DataFrame,
    *,
    x: str,
    y: str,
) -> pd.DataFrame:
    """Reshape a two-parameter grid result into a matrix for a heatmap.

    Joins a per-scenario value (one number per iteration, e.g. an ICER or
    incremental NMB) to a `heormodel.dsa.grid` descriptor and pivots it into a
    matrix over two gridded parameters. The base-case scenario is dropped.

    Args:
        values: One value per scenario, indexed by the shared iteration index.
        descriptor: The grid descriptor, carrying a column per gridded
            parameter.
        x: Parameter to lay along the columns.
        y: Parameter to lay along the index.

    Returns:
        DataFrame with ``y`` values as the index and ``x`` values as the
        columns.

    Example:
        >>> import pandas as pd
        >>> from heormodel.dsa import grid
        >>> from heormodel.report import heatmap_data
        >>> base = pd.Series({"a": 1.0, "b": 2.0})
        >>> design, descriptor = grid(base, {"a": [0.0, 1.0], "b": [10.0, 20.0]})
        >>> values = design["a"] + design["b"]
        >>> hm = heatmap_data(values, descriptor, x="a", y="b")
        >>> hm.shape
        (2, 2)
        >>> float(hm.loc[20.0, 1.0])
        21.0
    """
    for col in (x, y):
        if col not in descriptor.columns:
            raise ValueError(f"Grid parameter {col!r} not in the descriptor.")
    scenarios = descriptor[descriptor[SCENARIO_COL] != BASE_LABEL]
    table = pd.DataFrame(
        {
            x: scenarios[x].to_numpy(),
            y: scenarios[y].to_numpy(),
            "value": values.reindex(scenarios.index).to_numpy(dtype=np.float64),
        }
    )
    return table.pivot(index=y, columns=x, values="value")
