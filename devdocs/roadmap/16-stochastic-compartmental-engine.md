# 16. Stochastic compartmental engine

Implement `StochasticODEModel` in `heormodel/models/stochastic_ode.py`, the stochastic counterpart to `ODEModel`. It simulates the same compartmental structure as a continuous-time Markov jump process rather than a deterministic flow, so that chance fade-out of an outbreak and the variability in its final size enter the cost-effectiveness analysis instead of being averaged into a single trajectory. It emits the standard `Outcomes` schema and reuses `heormodel.models._accrual`.

## Why a stochastic version

The deterministic engine integrates the mean-field limit: an infinite, perfectly mixed population where the infectious fraction is a smooth number. That limit misses two things a decision maker cares about. First, near the epidemic threshold a real outbreak either takes off or dies out by chance, and the deterministic model cannot represent the fade-out probability. Second, even above threshold the final size varies from one realization to the next, and that variance is part of the decision uncertainty a probabilistic sensitivity analysis should carry. A vaccination program that looks marginal on the mean trajectory may be clearly worthwhile once the chance of a large outbreak is priced in.

## Coherence with the engine architecture

The engine keeps the three commitments the other engines share, and takes the stochastic-engine shape of the microsimulation and discrete-event engines:

1. Configure once, evaluate on draws. `StochasticODEModel(...)` takes the compartments, interventions, and a `reactions_and_rewards` callback; evaluation returns `Outcomes` indexed by `draws.index`.
2. Randomness comes from the run loop, not the engine. It implements `StochasticEngine.evaluate_streamed`, drawing a per-iteration stream keyed by the iteration index, so a run is reproducible in isolation and invariant to how it is split across workers, exactly as `MicrosimModel` and `DESModel` are.
3. Accrual reuse. Discounting and the reduction to `Outcomes` rows come from `heormodel.models._accrual`; the continuous-time accrual (`integrate_flow`) already fits the piecewise-constant trajectory between events.

## The reaction description

A jump process is defined by its reactions, not its derivatives. Each reaction has a propensity (a rate that depends on the current compartment counts) and a state-change vector (which compartments it increments and decrements). For a susceptible-exposed-infectious-recovered model with vaccination the reactions are infection (`S -> E`, propensity `beta * S * I / N`), progression (`E -> I`, propensity `sigma * E`), recovery (`I -> R`, propensity `gamma * I`), and vaccination (`S -> V`, propensity `nu * S`). A `ReactionSpec` returned by `reactions_and_rewards(params, intervention)` carries the propensity function `fn(t, y) -> rates`, the integer state-change matrix, the initial integer compartment counts, and the same reward channels as `ODESpec`: per-compartment cost and effect rates accrued on occupancy, and per-reaction one-time costs (a dose, a treated infection) accrued on the count of firings.

Reusing `ODESpec`'s reward vocabulary lets a model author move a deterministic model to its stochastic version by describing the same transitions as reactions, without relearning how rewards attach.

## Simulation

Two regimes, chosen by population size:

- Exact (Gillespie direct method). Draw the time to the next reaction from an exponential with rate equal to the total propensity, choose which reaction fires with probability proportional to its propensity, update the counts, accrue the discounted occupancy and reaction rewards over the elapsed interval with `integrate_flow`, and repeat until the horizon. Exact but slow when propensities are large, since it steps one event at a time.
- Tau-leaping. Over a small fixed step, fire each reaction a Poisson-distributed number of times with mean its propensity times the step, holding propensities constant across the leap. Orders of magnitude faster for large populations, at the cost of a controlled approximation; guard against negative counts by capping or shrinking the step.

Select automatically from the initial population and expose an override. Both paths accrue rewards the same way, so the choice affects speed and approximation, not the reward bookkeeping.

## Validation (acceptance)

- Mean-field convergence: averaged over many realizations at a large population, the stochastic engine's discounted costs and effects converge to the deterministic `ODEModel` on the same parameters (the cross-validation), and the two diverge in the small-population regime where fade-out matters (the reason the engine exists). A test asserts both.
- A single-reaction closed form: a pure-death process (`A -> B` at constant per-capita rate) has a known mean discounted occupancy, which the engine reproduces, mirroring the exponential-decay check that anchors `ODEModel`.
- Contract and reproducibility tests identical in shape to the microsimulation ones: the returned iteration index matches the draws, and results are invariant to worker count and batch size under a fixed seed.
- An example and a website tutorial extend the SEIR vaccination case to show the outbreak-size distribution and the fade-out probability the deterministic model cannot express.
