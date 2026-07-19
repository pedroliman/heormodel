# Architecture

This page explains the two guarantees that hold everywhere in `heormodel`: one standard outcome structure as the integration point, and the shared iteration index. It assumes the [quickstart](https://pedroliman.github.io/heormodel/); [engines](engines.md) covers the model side of the contract.

## The outcome structure is the integration point

`Outcomes` is a table indexed by `(intervention, iteration)` with a `cost` column, a primary effect column (`qaly` by default), and optionally more numeric columns carried as disaggregated components. Every engine returns this structure, every analysis consumes it, and none reaches into engine internals.

The constructor validates the contract so analyses do not have to: a balanced panel (every intervention evaluated on every iteration), unique `(intervention, iteration)` pairs, numeric columns, and no NaN or infinite values.

```python
import pandas as pd
from heormodel.models import Outcomes

tidy = pd.DataFrame({
    "intervention": ["A", "A", "B", "B"],
    "iteration": [0, 1, 0, 1],
    "cost": [100.0, 110.0, 200.0, 190.0],
    "qaly": [1.0, 1.1, 1.4, 1.3],
})
out = Outcomes.from_tidy(tidy)
out.summary()
```

Three constructors cover the common shapes: `from_tidy` for long tables (the bring-your-own-outputs entry point, also reachable as `as_outcomes`), `from_wide` for paired iterations-by-interventions cost and effect tables, and `Outcomes` itself for a table already in this form.

Because the analysis layer sees only this structure, cost-effectiveness and value-of-information analysis do not depend on the engine: `icer_table`, `ceac`, and `evpi` work identically on outcomes from a spreadsheet export, a decision tree function, or a future microsimulation engine.

## The shared iteration index

Parameter draw matrices and outcomes share one iteration index. `ParameterSet.sample` returns a draw matrix whose row index is the canonical iteration index, and `run_psa` guarantees the outcomes in the `RunResult` it returns carry exactly that index, in the same order. `run_psa` is the single execution point: it owns seeding (the `seed` argument builds the per-iteration streams a stochastic engine uses) and the optional event or individual log (the `collect` argument), so engines stay seed-free descriptions of a model.

The point is traceability: the expected value of partial perfect information and of sample information regress outcome quantities on parameter draws, which is only valid when row `i` of the draws produced iteration `i` of the outcomes. Analyses that need both objects, such as `evppi` and `tornado_data`, rely on the index to align them.

```python
from heormodel.params import Normal, ParameterSet
from heormodel.run import run_psa

def model(d: pd.DataFrame) -> Outcomes:
    costs = pd.DataFrame({"A": d["c"], "B": d["c"] + 10})
    effects = pd.DataFrame({"A": d["c"] * 0, "B": d["c"] * 0 + 0.1})
    return Outcomes.from_wide(costs, effects)

draws = ParameterSet({"c": Normal(100, 5)}).sample(500, seed=1)
outcomes = run_psa(model, draws, sequential=True).outcomes
outcomes.iterations.equals(draws.index)
```

`run_psa` rejects duplicated draw indices and re-checks the index on the way out, including after parallel batches are reassembled. If you bypass `run_psa` and pair a draw matrix with external outcomes, keeping the indices aligned is your responsibility; `evppi` will refuse mismatched indices.

## Interventions, and the value objects not yet built

Every engine names its arms the same way: `interventions` is a sequence of names or `Intervention(name, decision_levers)` objects, and every model function receives the intervention name so an arm can branch on it. `Intervention` also carries an `is_comparator` flag marking the PICOTS comparator (the reference arm), which each engine reads at construction and carries onto the `Outcomes` it returns as `Outcomes.comparator`; `heormodel.cea.ce_plane` and the tornado plots fall back to it when their own `comparator` argument is omitted. `Intervention` is the one shared value object the engines carry today. Two others were considered and deferred: `Timeline` (a cycle grid or a continuous horizon with its correction) and `Population` (size, attribute sampler, initial state). Their structural benefit, a single home for vocabulary shared across engines, pays off mainly when a fifth engine arrives; until then the flat keyword constructors read better in the two-state docstring examples. Add them when the engine roster grows, reusing them across engines rather than re-spelling the concepts.

Next: [engines](engines.md) describes what `ModelEngine` requires and what a contract on outputs means in practice.
