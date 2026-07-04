"""Model engines behind an output contract (`heval.models`).

The durable pieces here are `Outcomes`, the standardized
(strategy, iteration) outcome schema, and `ModelEngine`, the
protocol every engine satisfies. Engine implementations (cohort
state-transition, microsimulation, discrete-event) arrive in later phases;
their stubs document the planned designs.
"""

from heval.models.des import DESEngine, queue_waits
from heval.models.markov import MarkovCohortEngine
from heval.models.microsim import ContinuousTimeMicrosimEngine, DiscreteTimeMicrosimEngine
from heval.models.outcomes import Outcomes
from heval.models.protocol import ModelEngine, ModelFn

__all__ = [
    "ContinuousTimeMicrosimEngine",
    "DESEngine",
    "DiscreteTimeMicrosimEngine",
    "MarkovCohortEngine",
    "ModelEngine",
    "ModelFn",
    "Outcomes",
    "queue_waits",
]
