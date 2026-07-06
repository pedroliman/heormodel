# heormodel

[![CI](https://github.com/pedroliman/heormodel/actions/workflows/ci.yml/badge.svg)](https://github.com/pedroliman/heormodel/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pedroliman/heormodel/branch/main/graph/badge.svg)](https://codecov.io/gh/pedroliman/heormodel)
[![PyPI](https://img.shields.io/pypi/v/heormodel.svg)](https://pypi.org/project/heormodel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Decision-analytic modeling for health economic evaluation and health technology assessment in Python: probabilistic parameter specification, cohort state-transition, microsimulation, and discrete-event engines, cost-effectiveness analysis, and value-of-information analysis. Every engine returns costs and effects in one standardized structure, so the analysis layer also accepts a PSA table from any external source through `as_outcomes`.

Documentation: [pedroliman.github.io/heormodel](https://pedroliman.github.io/heormodel/)

## Install

```bash
pip install heormodel
```

Extras: `calibration` (ABC-SMC via `pyabc`), `des` (SimPy).

## Quickstart

A three-state cohort state-transition model comparing treatment with standard care, evaluated by probabilistic sensitivity analysis. `icer_table` reports the incremental cost-effectiveness ratio; `evpi` reports the expected value of perfect information at a willingness-to-pay threshold of 50,000 per QALY.

```python
import numpy as np
import pandas as pd
from heormodel.models import CohortSpec, MarkovModel
from heormodel.params import Beta, Gamma, ParameterSet
from heormodel.run import SeedManager, run_psa
from heormodel.cea import icer_table
from heormodel.voi import evpi

def model(p, strategy):
    p_progress = p["p_progress"] * (p["rr_treat"] if strategy == "Treatment" else 1.0)
    P = np.array([
        [1 - p_progress - p["p_die"], p_progress, p["p_die"]],
        [0.0, 1 - p["p_die_sick"], p["p_die_sick"]],
        [0.0, 0.0, 1.0],
    ])
    cost = np.array([0.0, p["c_sick"], 0.0])
    if strategy == "Treatment":
        cost[:2] += p["c_treat"]
    return CohortSpec(P, cost, np.array([1.0, p["u_sick"], 0.0]))

engine = MarkovModel(states=("Healthy", "Sick", "Dead"),
                     strategies=("Standard care", "Treatment"),
                     model_fn=model, n_cycles=40)

params = ParameterSet({
    "p_progress": Beta(20, 180), "rr_treat": Beta(60, 40),
    "p_die": Beta(5, 995), "p_die_sick": Beta(50, 450),
    "c_sick": Gamma(100, 250.0), "c_treat": Gamma(100, 80.0),
    "u_sick": Beta(150, 50),
})
draws = params.sample(1000, seed=SeedManager(1).generator())
outcomes = run_psa(engine, draws)

icer_table(outcomes).round(1)
#                    cost  effect  inc_cost  inc_effect     icer status
# strategy
# Standard care  142910.9    11.2       NaN         NaN      NaN     ND
# Treatment      233676.2    13.4   90765.3         2.2  41130.9     ND

round(evpi(outcomes, wtp=50_000), 1)
# 2738.7
```

Treatment is cost-effective at the threshold, and the positive EVPI quantifies the expected gain from resolving the remaining parameter uncertainty.

## Examples

Each script in [`examples/`](examples/) runs with `uv run python examples/<name>.py`.

| Script | Shows |
|---|---|
| `byoo_example.py` | An external PSA table through CEA, VoI, plots, and a run report |
| `mdm_cohort.py`, `mdm_cohort_timedep.py`, `mdm_microsim.py` | Replications of three published Sick-Sicker tutorials; see the [replication gallery](https://pedroliman.github.io/heormodel/tutorials/replication-gallery.html) |
| `microsim.py` | An individual-level model with frailty and duration-dependent mortality |
| `markov_vs_microsim.py` | One model as a cohort trace and a microsimulation; a cross-engine validation |
| `des.py` | A resource-constrained clinic where added capacity buys QALYs (`des` extra) |
| `calibration_workflow.py` | Calibrated rates mixed with literature parameters in one PSA (`calibration` extra) |
| `parameter_inputs.py` | Base-case runs, draw matrices from CSV, and posterior resampling |
| `voi_tutorial.py` | EVPI, EVPPI, and EVSI recovered against closed-form Gaussian results |

## Package layout

| Subpackage | Contents |
|---|---|
| `heormodel.params` | Distribution specs with mean/SE constructors; correlated sampling; draw matrices |
| `heormodel.models` | `Outcomes` schema; cohort state-transition, microsimulation, and discrete-event engines |
| `heormodel.run` | `SeedManager`, `run_psa`, `as_outcomes`, running-mean diagnostics |
| `heormodel.cea` | ICERs, dominance, frontier, NMB/NHB, CEAC/CEAF |
| `heormodel.dsa` | One-way, one-at-a-time, and grid deterministic sensitivity designs |
| `heormodel.voi` | EVPI; EVPPI (spline/GP metamodels); EVSI (regression) |
| `heormodel.calibrate` | ABC-SMC calibration; posterior as a draw matrix (optional) |
| `heormodel.report` | CE plane, CEAC/CEAF, frontier, tornado plots; run report |

Next steps: [`roadmap/`](roadmap/README.md). Shipped changes: [CHANGELOG.md](CHANGELOG.md). Release process: [RELEASING.md](RELEASING.md). Prose style: [`guidance/writing_style.md`](guidance/writing_style.md).

## Development

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run pytest --doctest-modules src
uv run ruff check . && uv run mypy
```

The suite includes a hand-verified five-strategy dominance example (`tests/test_cea.py`) and analytic Gaussian EVPI/EVPPI/EVSI recovered within Monte Carlo error (`tests/test_voi.py`).

The site in `docs/` builds with [Quarto](https://quarto.org) and [quartodoc](https://machow.github.io/quartodoc/); tutorials execute at render time. With Quarto installed and `uv sync --extra docs`:

```bash
uv run quartodoc build --config docs/_quarto.yml
quarto preview docs
```
