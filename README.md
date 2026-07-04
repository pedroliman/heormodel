# heval

[![CI](https://github.com/pedroliman/heval/actions/workflows/ci.yml/badge.svg)](https://github.com/pedroliman/heval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pedroliman/heval/branch/main/graph/badge.svg)](https://codecov.io/gh/pedroliman/heval)
[![PyPI](https://img.shields.io/pypi/v/heval.svg)](https://pypi.org/project/heval/)
[![Python versions](https://img.shields.io/pypi/pyversions/heval.svg)](https://pypi.org/project/heval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Health economic evaluation in Python: parameter specification and probabilistic sampling, simulation across multiple model types, cost-effectiveness analysis (CEA), and value-of-information (VoI) analysis for model-based HEOR/HTA work.

Documentation: [pedroliman.github.io/heval](https://pedroliman.github.io/heval/)

## The core idea

One parameter object flows through swappable model engines into a shared analysis layer. Engines do not share an implementation API; they share a contract on their outputs. Given a matrix of parameter draws, every engine returns costs and effects per strategy per iteration in one standardized structure, `Outcomes`, a tidy frame indexed by `(strategy, iteration)`. Standardized outputs make CEA and VoI engine-agnostic.

Two invariants hold everywhere. First, the outcome schema is the integration point: every engine targets it, every analysis consumes it, and none reaches into engine internals. Second, parameter and outcome matrices share the iteration index, so EVPPI and EVSI can trace which draw produced which outcome; `run_psa` enforces it.

## Bring your own outputs

You do not need an engine to use `heval`. A costs/effects PSA table from any source enters the pipeline through one call:

```python
import numpy as np
from heval.run import as_outcomes
from heval.cea import icer_table, ceac
from heval.voi import evpi

outcomes = as_outcomes("my_psa.csv")   # columns: strategy, iteration, cost, qaly
print(icer_table(outcomes))            # ICERs, dominance, extended dominance
curves = ceac(outcomes, np.linspace(0, 150_000, 61))
print(evpi(outcomes, wtp=50_000))
```

[`examples/byoo_example.py`](examples/byoo_example.py) is a runnable walkthrough: an external CSV through CEA, VoI, plots, and a model card, plus the full pipeline (`ParameterSet` sampling, `SeedManager`, `run_psa`).

## Package layout

| Subpackage | Status | Contents |
|---|---|---|
| `heval.params` | done | Distribution specs with mean/SE constructors; correlated sampling |
| `heval.models` | contract done | `Outcomes` schema, `ModelEngine` protocol; engines stubbed |
| `heval.run` | done | `SeedManager`, `run_psa`, `as_outcomes`, running-mean diagnostics |
| `heval.cea` | done | ICERs, dominance, extended dominance, frontier, NMB/NHB, CEAC/CEAF |
| `heval.voi` | done | EVPI; EVPPI (spline/GP metamodels); EVSI (regression; others stubbed) |
| `heval.calibrate` | done, optional | ABC-SMC via `pyabc`; posterior as an iteration-indexed draw matrix |
| `heval.report` | done | CE plane, CEAC/CEAF, frontier, tornado plots; provenance, model card |

Next steps are prioritized in [`roadmap/`](roadmap/README.md). Prose follows [`guidance/writing_style.md`](guidance/writing_style.md). Changes are tracked in [CHANGELOG.md](CHANGELOG.md); the release process is in [RELEASING.md](RELEASING.md).

## Installation

```bash
pip install heval
```

## Development

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e ".[dev]"   # extras: calibration (pyabc), dev
uv run pytest                            # suite incl. validation checks
uv run pytest --doctest-modules src      # every docstring example runs
uv run ruff check . && uv run mypy
```

Two validation checks anchor the test suite: a hand-verified five-strategy dominance and ICER example (`tests/test_cea.py`), and analytic Gaussian EVPI/EVPPI/EVSI recovered within Monte Carlo error at 80,000 iterations (`tests/test_voi.py`).

The documentation site lives in `docs/` and builds with [Quarto](https://quarto.org) and [quartodoc](https://machow.github.io/quartodoc/); tutorials execute at render time, so the build doubles as a test. With Quarto installed and the `docs` extra synced (`uv sync --extra docs`):

```bash
uv run quartodoc build --config docs/_quarto.yml   # generate the API reference
quarto preview docs                                 # or: quarto render docs
```

CI (`.github/workflows/docs.yml`) rebuilds and publishes the site to GitHub Pages on every push to `main`.

## Design notes and deviations

- VoI metamodeling uses scikit-learn (`pygam` does not install against numpy 2.4+); `method=` leaves room for other backends.
- `heval.calibrate` is a seventh subpackage beyond the original six; calibrated draws re-enter the pipeline as a standard draw matrix.
- Mean/SE constructors turn published estimates into sampling distributions; direct parameterisation remains available.
- Dirichlet vectors sample as normalised independent Gammas; leave their correlation targets at zero.
- Tornado diagrams are PSA-based (linear fits at outer percentiles); phase 1 has no deterministic engine to re-run.
