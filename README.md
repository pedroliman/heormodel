# heormodel

[![CI](https://github.com/pedroliman/heormodel/actions/workflows/ci.yml/badge.svg)](https://github.com/pedroliman/heormodel/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pedroliman/heormodel/branch/main/graph/badge.svg)](https://codecov.io/gh/pedroliman/heormodel)
[![PyPI](https://img.shields.io/pypi/v/heormodel.svg)](https://pypi.org/project/heormodel/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21314037.svg)](https://doi.org/10.5281/zenodo.21314037)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`heormodel` is a Python decision-analytic modeling framework for health economic evaluation and health technology assessment.

`heormodel` covers the full cost-effectiveness analysis workflow in one package. It supports probabilistic parameter specification for a range of models: Markov cohort state-transition models, microsimulation models, and discrete-event simulation models. It builds the incremental cost-effectiveness ratio (ICER) table and runs value-of-information analysis. If you are not ready to port your model to Python, you can also bring your existing model results directly into the package.

Read more in the documentation: [pedroliman.github.io/heormodel](https://pedroliman.github.io/heormodel/)

## Citing

Cite `heormodel` using the metadata in [`CITATION.cff`](CITATION.cff), or
click "Cite this repository" in the sidebar of the GitHub repository. Every
release also archives to [Zenodo](https://zenodo.org) with a DOI; the DOI
badge above always resolves to the archive of the latest release. See
[RELEASING.md](RELEASING.md#zenodo-archive) for how the archiving works.

## Install

If you are new to Python, I recommend installing it with [uv](https://docs.astral.sh/uv/guides/install-python/). Once you have a working Python installation, run this from your terminal within your project's folder:

```bash
pip install heormodel
# or using uv, which I prefer:
# (run uv init once)
uv init
uv add heormodel
```

## Quickstart

Here is a quick example to get you started: a three-state Markov cohort state-transition model comparing treatment with standard care, evaluated by probabilistic sensitivity analysis. This code builds the model, runs it, and reports the ICER table and the expected value of perfect information.

```python
import numpy as np
import pandas as pd
from heormodel.models import CohortSpec, MarkovModel
from heormodel.params import Beta, Gamma, ParameterSet
from heormodel.run import SeedManager, run_psa
from heormodel.cea import icer_table
from heormodel.voi import evpi

# define your model.
def model(p, intervention):
    p_progress = p["p_progress"] * (p["rr_treat"] if intervention == "Treatment" else 1.0)
    # Transition matrix. Rows: Current state. Columns: Next state.
    P = np.array([
        [1 - p_progress - p["p_die"], p_progress, p["p_die"]],
        [0.0, 1 - p["p_die_sick"], p["p_die_sick"]],
        [0.0, 0.0, 1.0],
    ])
    cost = np.array([0.0, p["c_sick"], 0.0])
    if intervention == "Treatment":
        cost[:2] += p["c_treat"]
    return CohortSpec(P, cost, np.array([1.0, p["u_sick"], 0.0]))

# create the MarkovModel engine.
engine = MarkovModel(states=("Healthy", "Sick", "Dead"),
                     interventions=("Standard care", "Treatment"),
                     transitions_and_rewards=model, n_cycles=40)

# Define your parameters:
params = ParameterSet({
    "p_progress": Beta(20, 180), "rr_treat": Beta(60, 40),
    "p_die": Beta(5, 995), "p_die_sick": Beta(50, 450),
    "c_sick": Gamma(100, 250.0), "c_treat": Gamma(100, 80.0),
    "u_sick": Beta(150, 50),
})

# sample your parameters:
draws = params.sample(1000, seed=SeedManager(1).generator())

# run your model over your parameters.
outcomes = run_psa(engine, draws).outcomes

# Get the ICER table.
icer_table(outcomes).round(1)
#                    cost  effect  inc_cost  inc_effect     icer status
# intervention
# Standard care  142910.9    11.2       NaN         NaN      NaN     ND
# Treatment      233676.2    13.4   90765.3         2.2  41130.9     ND

# And from here your EVPI
round(evpi(outcomes, wtp=50_000), 1)
# 2738.7
```

Once you master this workflow, you can do a lot more with the package, like defining microsimulations, discrete-event simulation models, and compartmental transmission models written as ordinary differential equations (the `ODEModel` engine, with a susceptible-exposed-infectious-recovered vaccination example). The package also has a calibration function you can use to calibrate some parameters, take others from the literature, then run a full probabilistic sensitivity analysis.

## Development

Developer documentation lives in [`devdocs/`](devdocs/README.md). See the [CHANGELOG.md](CHANGELOG.md) for recent changes and follow the release process: [RELEASING.md](RELEASING.md).

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run pytest --doctest-modules src
uv run ruff check . && uv run mypy
```

The site in `docs/` builds with [Quarto](https://quarto.org) and [quartodoc](https://machow.github.io/quartodoc/); tutorials execute at render time. With Quarto installed and `uv sync --extra docs`:

```bash
uv run quartodoc build --config docs/_quarto.yml
quarto preview docs
```

Each tutorial also carries an "Open in Colab" badge backed by a runnable notebook under `docs/_notebooks/`. Regenerate the badges and notebooks after editing a tutorial; continuous integration checks they stay in sync:

```bash
uv run python docs/build_colab_notebooks.py
```
