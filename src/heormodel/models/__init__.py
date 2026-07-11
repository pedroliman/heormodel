"""Model engines behind an output contract (`heormodel.models`).

The durable pieces here are `Outcomes`, the standardized
(intervention, iteration) outcome structure, and `ModelEngine`, the
contract every engine satisfies. The engines are `MarkovModel`
(cohort state-transition), `MicrosimModel` (individual-level, built through
`MicrosimModel.discrete` or `MicrosimModel.continuous`), `DESModel`
(discrete-event, wrapping SimPy), and `ODEModel` (compartmental, integrating a
system of ordinary differential equations). `state_occupancy` turns an
individual-level event history into the proportion of the population in each
state over time.
"""

from heormodel.models._interventions import Intervention
from heormodel.models.des import DESModel, queue_waits
from heormodel.models.lifetable import LifeTable
from heormodel.models.markov import CohortSpec, MarkovModel
from heormodel.models.microsim import MicrosimModel
from heormodel.models.occupancy import state_occupancy
from heormodel.models.ode import ODEModel, ODESpec
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
    "ODEModel",
    "ODESpec",
    "Outcomes",
    "StochasticEngine",
    "Intervention",
    "queue_waits",
    "state_occupancy",
]
