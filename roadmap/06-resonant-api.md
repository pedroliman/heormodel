# 6. Resonant engine names and clearer parameters

Three API refinements that make the engine layer read the way health economists think. They are cosmetic but breaking, so they ship together in one release before the API stabilizes for PyPI. Nothing about the output contract, the draw matrix, or the analysis layer changes.

## Rename the engines

The current class names carry an implementation word (`Engine`) that users do not reach for. Rename to the model type:

| Now | New |
| --- | --- |
| `MarkovCohortEngine` | `MarkovModel` |
| `DiscreteTimeMicrosimEngine` | `MicrosimModel` |
| `ContinuousTimeMicrosimEngine` | `MicrosimModel(clock="continuous")` |
| `DESEngine` | `DESModel` |

Fold the two microsimulation classes into one `MicrosimModel` with a `clock` argument (`"discrete"` by default, `"continuous"` for the competing-hazards path). One name covers both, and the constructor's other arguments already differ by clock (`transition` versus `hazards`), so the split into two public classes bought nothing. The `ModelEngine` protocol name stays: it describes a role, not a model type, and users implement against it rather than reading it.

Keep the old names as thin subclasses that emit a `DeprecationWarning` pointing at the new name, for one release. The internal `_accrual` module and the protocol are untouched.

## Rename the `build` parameter

`MarkovModel` takes `build=fn`, where `fn(params, strategy)` returns the `CohortSpec` (transition matrix and reward arrays) for one parameter row and one strategy. The word `build` reads as a lifecycle hook, not as "the function that produces the model". Rename it to `model_fn`: the argument holds the function that maps parameters to the model's structure, and the `_fn` suffix matches the existing `ModelFn` type in `heval.models`. The callback's signature and return type are unchanged. Apply the same name if any other engine grows an equivalent structure callback; the microsimulation `transition`/`payoffs` and DES `process` callbacks already name what they return and stay as they are.

## One discount rate

Costs and effects are always discounted at the same rate in practice, so the two knobs invite an error without buying flexibility. Replace `discount_cost` and `discount_effect` with a single `discount_rate`, default `0.03`, on every engine (`MarkovModel`, `MicrosimModel`, `DESModel`) and anywhere `_accrual` exposes a rate.

`discount_rate` is an annual rate on an annual clock. `cycle_length` scales the clock: a model with `cycle_length=0.5` discounts each cycle by half a year. Document this assumption on every engine that has a `cycle_length`. Passing the removed `discount_cost` or `discount_effect` raises a `TypeError` whose message names `discount_rate`, so old code fails loudly rather than silently discounting at the default.

## Deliverables

- Renamed classes and the folded `MicrosimModel`, with deprecation shims for the old names for one release.
- `model_fn` in place of `build` on `MarkovModel`.
- `discount_rate` in place of `discount_cost` and `discount_effect` across all engines and `_accrual`.
- Updated `__all__`, the quartodoc reference section, every example and tutorial, the changelog, and the README.

## Acceptance

- All examples and tutorials run under `uv run python` and `uv run quartodoc build` with the new names.
- A test asserts the deprecated class names still construct and warn, and that `discount_cost`/`discount_effect` raise with an actionable message.
- Numerical results are identical to the pre-rename engines on the existing validation suite.
