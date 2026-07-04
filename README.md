# heval

[![CI](https://github.com/pedroliman/heval/actions/workflows/ci.yml/badge.svg)](https://github.com/pedroliman/heval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pedroliman/heval/branch/main/graph/badge.svg)](https://codecov.io/gh/pedroliman/heval)
[![PyPI](https://img.shields.io/pypi/v/heval.svg)](https://pypi.org/project/heval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Health economic evaluation in Python: parameter specification and probabilistic sampling, simulation across model types, cost-effectiveness analysis (CEA), and value-of-information (VoI) analysis for model-based HEOR and HTA work.

Every model engine returns costs and effects in one standardized structure, so CEA and VoI run the same on outputs from any source. You do not need a built-in engine to start: a PSA table from a spreadsheet or a legacy simulator enters the pipeline through one call.

Documentation: [pedroliman.github.io/heval](https://pedroliman.github.io/heval/)

## Install

```bash
pip install heval
```

The `calibration` extra adds ABC-SMC calibration: `pip install "heval[calibration]"`. Development uses [`uv`](https://docs.astral.sh/uv/); see [Development](#development).

## Quickstart

Load a costs/effects PSA table with `as_outcomes`, then run incremental analysis and value of information from the same object:

```python
import numpy as np
import pandas as pd
from heval.run import as_outcomes
from heval.cea import icer_table
from heval.voi import evpi

rng = np.random.default_rng(7)
n = 2_000
df = pd.concat(
    pd.DataFrame({
        "strategy": name, "iteration": range(n),
        "cost": rng.normal(cost, 2_000, n), "qaly": rng.normal(q, 0.4, n),
    })
    for name, cost, q in [("Standard care", 40_000, 8.0), ("New drug", 52_000, 8.6)]
)

outcomes = as_outcomes(df)   # accepts a DataFrame or a CSV path
icer_table(outcomes).round(1)
#                   cost  effect  inc_cost  inc_effect     icer status
# strategy
# Standard care  39920.1     8.0       NaN         NaN      NaN     ND
# New drug       51985.1     8.6   12065.0         0.6  20606.0     ND

round(evpi(outcomes, wtp=30_000), 1)
# 4339.3
```

[`examples/byoo_example.py`](examples/byoo_example.py) runs the same wedge end to end: an external table through CEA, VoI, plots, and a model card, plus the full pipeline (`ParameterSet` sampling, `SeedManager`, `run_psa`).

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

Full docs are at [pedroliman.github.io/heval](https://pedroliman.github.io/heval/). Next steps are prioritized in [`roadmap/`](roadmap/README.md); shipped changes are in [CHANGELOG.md](CHANGELOG.md); the release process is in [RELEASING.md](RELEASING.md). Prose follows [`guidance/writing_style.md`](guidance/writing_style.md).

## Development

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e ".[dev]"   # extras: calibration (pyabc), dev
uv run pytest                            # suite incl. validation checks
uv run pytest --doctest-modules src      # every docstring example runs
uv run ruff check . && uv run mypy
```

Two validation checks anchor the suite: a hand-verified five-strategy dominance and ICER example (`tests/test_cea.py`), and analytic Gaussian EVPI/EVPPI/EVSI recovered within Monte Carlo error at 80,000 iterations (`tests/test_voi.py`).

The documentation site lives in `docs/` and builds with [Quarto](https://quarto.org) and [quartodoc](https://machow.github.io/quartodoc/); tutorials execute at render time, so the build doubles as a test. With Quarto installed and the `docs` extra synced (`uv sync --extra docs`):

```bash
uv run quartodoc build --config docs/_quarto.yml   # generate the API reference
quarto preview docs                                 # or: quarto render docs
```

CI (`.github/workflows/docs.yml`) rebuilds and publishes the site to GitHub Pages on every push to `main`.

## Design notes

- VoI metamodeling uses scikit-learn; `method=` leaves room for other backends.
- `heval.calibrate` is a seventh subpackage beyond the original six; calibrated draws re-enter the pipeline as a standard draw matrix.
- Mean/SE constructors turn published estimates into sampling distributions; direct parameterisation remains available.
- Dirichlet vectors sample as normalised independent Gammas; leave their correlation targets at zero.
- Tornado diagrams are PSA-based (linear fits at outer percentiles); phase 1 has no deterministic engine to re-run.
