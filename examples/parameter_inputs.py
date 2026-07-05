"""Three ways parameters enter the run loop, each fed to ``run_psa``.

Not every analysis starts from ``ParameterSet`` distributions. This script
shows the three entry points that turn other sources into the standard
parameter draw matrix:

    - ``single_draw`` / ``ParameterSet.at_means`` for a base-case run;
    - ``read_draws`` for a draw matrix exported from another tool (a CSV here);
    - ``resample_posterior`` for a weighted posterior resampled with replacement.

Each produces a draw matrix that ``run_psa`` accepts unchanged.

Run it with::

    uv run python examples/parameter_inputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heval.models import Outcomes
from heval.params import (
    Beta,
    Gamma,
    ParameterSet,
    read_draws,
    resample_posterior,
    single_draw,
)
from heval.run import run_psa

HERE = Path(__file__).parent
OUT = HERE / "output"
WTP = 30_000.0


def model(draws: pd.DataFrame) -> Outcomes:
    """A two-strategy model: standard care versus a new drug.

    Cost and QALYs per iteration are simple functions of three parameters, so
    the outputs are traceable to the inputs.
    """
    base_qaly = 8.0
    effect_drug = base_qaly + draws["u_gain"] * draws["p_response"] * 10
    costs = pd.DataFrame(
        {"Standard care": 40_000.0, "New drug": 40_000.0 + draws["c_drug"]},
        index=draws.index,
    )
    effects = pd.DataFrame(
        {"Standard care": base_qaly, "New drug": effect_drug},
        index=draws.index,
    )
    return Outcomes.from_wide(costs, effects)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    params = ParameterSet(
        {
            "p_response": Beta.from_mean_se(0.35, 0.05),
            "c_drug": Gamma.from_mean_se(12_000, 1_500),
            "u_gain": Beta.from_mean_se(0.12, 0.03),
        },
        correlation={("p_response", "u_gain"): 0.4},
    )

    # --- 1. base case: the deterministic run at point values ---------------
    # ParameterSet.at_means is single_draw over the analytic means; single_draw
    # takes any hand-specified point values just as directly.
    base = params.at_means()
    single_draw({"p_response": 0.35, "c_drug": 12_000.0, "u_gain": 0.12})
    base_outcomes = run_psa(model, base)
    print("Base case (ParameterSet.at_means):")
    print(base_outcomes.summary().round(2).to_string())

    # --- 2. a draw matrix arrives from another tool as a CSV ---------------
    external = params.sample(2_000, seed=1)
    csv_path = OUT / "external_draws.csv"
    external.to_csv(csv_path)
    csv_draws = read_draws(csv_path, iteration="iteration")
    csv_outcomes = run_psa(model, csv_draws)
    print(f"\nCSV draw matrix (read_draws): {csv_draws.shape[0]} iterations")
    print(csv_outcomes.summary().round(2).to_string())

    # --- 3. a weighted posterior, resampled with replacement ----------------
    rng = np.random.default_rng(7)
    grid = params.sample(500, seed=2)
    # A calibration reweights the grid toward higher response probabilities.
    grid["weight"] = np.exp(4.0 * grid["p_response"])
    posterior = resample_posterior(grid, n=2_000, seed=rng)
    post_outcomes = run_psa(model, posterior)
    print("\nWeighted posterior (resample_posterior):")
    print(
        f"  grid mean p_response {grid['p_response'].mean():.3f}, "
        f"weighted resample mean {posterior['p_response'].mean():.3f}"
    )
    print(post_outcomes.summary().round(2).to_string())

    print(f"\nAll three matrices fed run_psa unchanged. WTP {WTP:,.0f}.")


if __name__ == "__main__":
    main()
