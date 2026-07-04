"""Run orchestration (`heval.run`).

Seed management, the PSA run loop (serial or ``joblib``-parallel), the
bring-your-own-outputs ingestion point, and convergence diagnostics.
"""

from heval.run.diagnostics import running_means
from heval.run.runner import as_outcomes, run_psa
from heval.run.seeds import SeedManager

__all__ = ["SeedManager", "as_outcomes", "run_psa", "running_means"]
