"""Model engines behind an output contract (`heval.models`).

The durable pieces here are `Outcomes`, the standardized
(strategy, iteration) outcome schema, and `ModelEngine`, the
protocol every engine satisfies. The engines are `MarkovModel`
(cohort state-transition), `MicrosimModel` (individual-level, discrete- or
continuous-time via ``clock``), and `DESModel` (discrete-event, wrapping
SimPy).
"""

from heval.models.des import DESModel, queue_waits
from heval.models.markov import CohortSpec, MarkovModel
from heval.models.microsim import MicrosimModel
from heval.models.outcomes import Outcomes
from heval.models.protocol import ModelEngine, ModelFn

__all__ = [
    "CohortSpec",
    "DESModel",
    "MarkovModel",
    "MicrosimModel",
    "ModelEngine",
    "ModelFn",
    "Outcomes",
    "queue_waits",
]
