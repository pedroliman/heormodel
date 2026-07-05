# 8. Deterministic sensitivity analysis

Probabilistic analysis answers how uncertain the decision is; deterministic sensitivity analysis (DSA) answers which parameters move the result and by how much. Both read the same model. This item adds a `heval.dsa` module that builds scenario designs, runs them through the existing loop, and returns results tied back to the scenario that produced them. It reuses the run loop and the `Outcomes` schema unchanged: a DSA design is a draw matrix whose rows are scenarios instead of random draws, plus a descriptor table naming what each scenario varied.

The design covers the three standard forms: one-way, one-at-a-time across a set of parameters, and a full-factorial grid.

## Designs

Each builder takes a base case (a `pandas.Series` of point values, from `ParameterSet.means()` or `single_draw`) and returns a `(design, descriptor)` pair. The `design` is a draw matrix ready for `run_psa`; the `descriptor` is a tidy table with one row per scenario recording the varied parameter names and their values, so results stay interpretable and plots can label axes.

```python
def one_way(base, parameter, values) -> Design:
    """Vary one parameter across ``values``, holding the rest at base."""

def one_at_a_time(base, ranges) -> Design:
    """Vary each parameter in ``ranges`` in turn (its low and high, or a
    sequence), holding the rest at base. The union of one-way sweeps."""

def grid(base, grids) -> Design:
    """Full-factorial: every combination of the listed parameters at their
    listed values, holding unlisted parameters at base."""
```

`ranges` and `grids` map a parameter name to a sequence of values (or a `(low, high)` pair). The base case is always included as one scenario so incremental effects read against it.

## Running and reading

The design is a draw matrix, so `run_psa(model, design)` evaluates it, in parallel by the default from item 9. Deterministic engines (`MarkovModel`) need no seed; stochastic engines (`MicrosimModel`, `DESModel`) run each scenario at a fixed seed so differences reflect the parameter, not Monte Carlo noise. Join the returned `Outcomes` to the descriptor on the iteration index to attribute every result to its scenario.

The one-way and one-at-a-time outputs feed the existing `heval.report.tornado_data` and `plot_tornado`, generalized to accept a DSA result rather than only a PSA. The grid output supports a two-way heatmap of an outcome (for example, the ICER) across two parameters.

## Deliverables

- `heval.dsa` with `one_way`, `one_at_a_time`, `grid`, and the `Design`/`descriptor` contract, added to `__all__` and the quartodoc reference.
- `tornado_data` accepts a DSA one-way result; a `heatmap_data` helper reshapes a two-parameter grid.
- `examples/dsa.py` running a one-way sweep, a one-at-a-time tornado, and a two-way grid on the Sick-Sicker model, printing the tornado table and saving a tornado and a heatmap.
- A website tutorial narrating the example alongside the PSA it complements.

## Acceptance

- A test asserts each design has the expected row count (one-way: `len(values)`; grid: the product of the value-set sizes, plus the base case) and that unlisted parameters equal the base case in every scenario.
- A test on a linear closed-form model asserts the one-way outcome changes match the analytic sensitivity.
- The example and tutorial run under `uv run python`.
