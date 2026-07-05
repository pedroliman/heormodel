# 9. Parallel runs by default, with a time-remaining display

A PSA of a microsimulation or discrete-event model is embarrassingly parallel across iterations, and the seeding architecture already guarantees results do not depend on `n_jobs` (item 3, commitment 2). So the safe default is parallel, and a run long enough to parallelize is long enough to want a progress readout. This item flips the `run_psa` default and adds a completed-count and time-remaining display driven by finished experiments.

## Parallel by default

`run_psa` currently defaults to `n_jobs=1`. Change the default to all cores (`n_jobs=-1`), and add `sequential=False` as the readable off switch (equivalent to `n_jobs=1`) for debugging, doctests, and reproducibility checks. Keep `n_jobs` for explicit worker counts. Because iteration streams are seeded by position, the numbers are identical whether the run is parallel or sequential; a test already asserts this and stays the guardrail.

Two practical guards: fall back to sequential when there is one iteration or one core, and keep the doctests and the docs-site tutorials sequential (their runtimes are small and a spawned pool would dominate them).

## Time-remaining display

Show progress as experiments finish, not as a spinner. An experiment is one unit of work the loop dispatches (a batch, or an iteration when unbatched). After each finishes, print the completed count, the total, the elapsed time, and an estimate of time remaining from the mean throughput of finished experiments:

```
running_psa: 320/1000 experiments, 0:00:12 elapsed, ~0:00:26 remaining
```

Requirements:

- Works in both modes. In parallel, update as `joblib` returns each batch; use a callback or `joblib`'s batch-completion hook rather than waiting for the whole pool.
- On by default, off with `progress=False`, and auto-off when output is not a TTY (so CI logs and the docs build stay quiet) unless `progress=True` is explicit.
- The remaining-time estimate uses only finished experiments, so it sharpens as the run proceeds. State plainly that early estimates are noisy.
- No new required dependency. A minimal internal reporter writing to `stderr` is enough; if a progress-bar library is already in the environment it may be used, but the feature must not depend on one.

## Deliverables

- `run_psa` defaulting to parallel, with `sequential` and `progress` arguments documented, and the batching interaction with the display explained.
- An internal `heval.run._progress` reporter with the elapsed and remaining formatting.
- Updated docstring example, examples, and any tutorial that shows a long run.

## Acceptance

- A test asserts identical `Outcomes` for `sequential=True`, `n_jobs=1`, and `n_jobs=2` on a fixed seed.
- A test asserts the reporter's remaining-time estimate is finite and non-increasing in expectation as experiments complete, and that `progress=False` and a non-TTY produce no output.
- The default parallel run of an existing example completes and matches its committed reference numbers.
