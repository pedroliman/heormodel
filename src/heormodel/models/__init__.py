"""Model engines behind an output contract (`heval.models`).

The durable pieces here are `Outcomes`, the standardized
(strategy, iteration) outcome structure, and `ModelEngine`, the
contract every engine satisfies. The engines are `MarkovModel`
(cohort state-transition), `MicrosimModel` (individual-level, discrete- or
continuous-time via ``clock``), and `DESModel` (discrete-event, wrapping
SimPy).
"""

from heormodel.models.des import DESModel, queue_waits
from heormodel.models.lifetable import LifeTable
from heormodel.models.markov import CohortSpec, MarkovModel
from heormodel.models.microsim import MicrosimModel
from heormodel.models.outcomes import Outcomes
from heormodel.models.protocol import ModelEngine, ModelFn

__all__ = [
    "CohortSpec",
    "DESModel",
    "LifeTable",
    "MarkovModel",
    "MicrosimModel",
    "ModelEngine",
    "ModelFn",
    "Outcomes",
    "queue_waits",
]
