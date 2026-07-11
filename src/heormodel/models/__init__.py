"""Model engines behind an output contract (`heormodel.models`).

The durable pieces here are `Outcomes`, the standardized
(strategy, iteration) outcome structure, and `ModelEngine`, the
contract every engine satisfies. The engines are `MarkovModel`
(cohort state-transition), `MicrosimModel` (individual-level, built through
`MicrosimModel.discrete` or `MicrosimModel.continuous`), and `DESModel`
(discrete-event, wrapping SimPy). `state_occupancy` turns an individual-level
event history into the proportion of the population in each state over time.
"""

from heormodel.models.des import DESModel, queue_waits
from heormodel.models.lifetable import LifeTable
from heormodel.models.markov import CohortSpec, MarkovModel
from heormodel.models.microsim import MicrosimModel
from heormodel.models.occupancy import state_occupancy
from heormodel.models.outcomes import Outcomes
from heormodel.models.protocol import EngineResult, ModelEngine, ModelFn, StochasticEngine

__all__ = [
    "CohortSpec",
    "DESModel",
    "EngineResult",
    "LifeTable",
    "MarkovModel",
    "MicrosimModel",
    "ModelEngine",
    "ModelFn",
    "Outcomes",
    "StochasticEngine",
    "queue_waits",
    "state_occupancy",
]
