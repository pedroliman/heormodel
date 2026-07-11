"""Run orchestration (`heormodel.run`).

Seed management, the probabilistic run loop (serial or ``joblib``-parallel),
bring-your-own-outputs ingestion point, and convergence diagnostics.
"""

from heormodel.run.diagnostics import running_means
from heormodel.run.runner import RunResult, as_outcomes, run_psa
from heormodel.run.seeds import SeedManager

__all__ = ["RunResult", "SeedManager", "as_outcomes", "run_psa", "running_means"]
