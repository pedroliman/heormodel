# Engines

This page explains the engine side of the contract: what `ModelEngine` requires, why engines share outputs rather than an implementation, and where the built-in engines stand. [Architecture](architecture.md) covers the two guarantees behind the outcome structure, and the [quickstart](https://pedroliman.github.io/heormodel/) introduces the `Outcomes` structure those engines return.

## A contract on outputs, not internals

Model engines differ radically inside: cohort matrix algebra, individual-level simulation, discrete-event simulation. `heormodel` deliberately does not force them into a shared implementation. What they share is one obligation: given a parameter draw matrix, return `Outcomes` whose iteration index equals the draws' index.

Two guarantees follow. The returned iteration index must match `draws.index` in values and order, preserving the parameter/outcome linkage that value-of-information estimation needs. And every intervention must be evaluated on every iteration, a balanced panel, which the `Outcomes` constructor enforces.

## An engine is anything with an `evaluate` method

`ModelEngine` names the required shape: one method, `evaluate`, taking a draw matrix and returning `Outcomes`. Any object with such a method qualifies, no inheritance required:

```python
import pandas as pd
from heormodel.models import ModelEngine, Outcomes

class TwoInterventionModel:
    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        costs = pd.DataFrame({"A": draws["c"], "B": draws["c"] * 2})
        effects = pd.DataFrame({"A": draws["e"], "B": draws["e"] * 1.5})
        return Outcomes.from_wide(costs, effects)

isinstance(TwoInterventionModel(), ModelEngine)
```

A plain function `draws -> Outcomes` works too; the name `ModelFn` refers to that form, and `run_psa` accepts either. The [full pipeline tutorial](https://pedroliman.github.io/heormodel/tutorials/full-pipeline.html) runs a decision tree written as a function.

`run_psa` is where the contract is enforced rather than trusted. It rejects empty or duplicate-indexed draws, checks the returned index, and because it runs in parallel batches by default, it verifies interventions are consistent across batches and reassembles the panel before checking again. An engine that violates the contract fails loudly at the run boundary, not silently inside an analysis.

## The built-in engines

Four built-in engines cover the model types used in cost-effectiveness analysis, and all four configure once and evaluate on draws. `MarkovModel` sweeps a cohort trace across iterations with constant or per-cycle (age-varying) transition arrays, per-state and per-transition rewards, and a choice of within-cycle correction. `MicrosimModel` advances an individual-level population on a cycle grid (`MicrosimModel.discrete`) or races competing time-to-event samplers between events (`MicrosimModel.continuous`), which lets it represent history and heterogeneity that a cohort trace averages away. `DESModel` builds on SimPy for discrete-event models where entities queue for scarce resources; the SimPy environment, processes, and resources stay the user's code, and the engine adds discounted accrual, seeding, and an event log. `ODEModel` integrates a system of ordinary differential equations over population compartments, for transmission models where a force of infection couples the compartments non-linearly; it accrues discounted costs and effects on compartment occupancy and on the flows between compartments. The first three and `ODEModel` share the accrual and discounting helpers in `heormodel.models._accrual`, so a change to how discounting works reaches every engine at once. The [cohort](https://pedroliman.github.io/heormodel/tutorials/mdm-cohort.html), [microsimulation](https://pedroliman.github.io/heormodel/tutorials/microsim.html), [discrete-event](https://pedroliman.github.io/heormodel/tutorials/des.html), and [compartmental](https://pedroliman.github.io/heormodel/tutorials/seir-vaccination.html) tutorials run them end to end, and the [replication gallery](https://pedroliman.github.io/heormodel/tutorials/replication-gallery.html) reproduces published results with them.

`MarkovModel` and `ODEModel` are deterministic given a parameter set, so they satisfy `ModelEngine` with an `evaluate` method alone. `MicrosimModel` and `DESModel` are stochastic: they additionally take per-iteration random streams from the run loop through the wider `StochasticEngine` contract, so a run's numbers do not depend on how it is split across workers.

## The model function each engine expects

Engines share the output contract but not the shape of the function you write to describe the model. A deterministic engine takes one function that returns a specification object bundling the dynamics and the rewards. A microsimulation takes two functions, because it evaluates transitions and rewards for every individual at every step and keeps those two jobs separate. The discrete-event engine takes one imperative process that accrues as it runs. Every one of them receives the parameter row and the intervention name, so an arm can branch on either.

| Engine | Model function(s) | Signature |
|--------|-------------------|-----------|
| `MarkovModel` | `transitions_and_rewards` | `(params, intervention) -> CohortSpec` |
| `ODEModel` | `dynamics_and_rewards` | `(params, intervention) -> ODESpec` |
| `MicrosimModel.discrete` | `transition_probabilities`, `state_rewards` | `(params, intervention, state, attrs, rng)`, `(params, intervention, state, attrs)` |
| `MicrosimModel.continuous` | `event_times`, `state_reward_rates` | `(params, intervention, state, attrs, rng)`, `(params, intervention, state, attrs)` |
| `DESModel` | `process` | `(env, entity, params, intervention, toolkit)` |

`DESModel` leads with `env` and `entity` rather than the parameter row because its process is a SimPy generator and SimPy passes the environment first; the parameter row and intervention name follow.

Because analyses consume only the outcome structure, everything downstream of an engine already works, so each engine ships against a stable analysis layer and a documented contract.
