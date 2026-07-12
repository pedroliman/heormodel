"""Ordinary differential equation (compartmental) engine.

`ODEModel` evaluates a system of ordinary differential equations across
population compartments and returns the standard `Outcomes` structure. It suits
infectious-disease models, where a force of infection couples the compartments
non-linearly and the natural description is a rate of change per compartment
rather than a per-cycle transition probability. A susceptible-exposed-infectious
-recovered (SEIR) model with vaccination is the worked example in
`examples/seir_vaccination.py`.

The engine follows the same shape as the other engines. The constructor takes
the model structure (compartments as states, interventions, a
``dynamics_and_rewards`` function, the time horizon, and discounting);
``evaluate`` takes only the parameter draw matrix and returns `Outcomes` indexed
by ``draws.index``. A system of ordinary differential equations is deterministic
given a parameter set, so no random streams are involved and the engine
satisfies `heormodel.models.ModelEngine`.

Rewards follow the same two channels as the cohort engine, adapted to
continuous time. State rewards are per-year cost and utility flows that accrue
on compartment occupancy: the integral over the horizon of ``occupancy dot
state_cost``, discounted continuously. Flow rewards attach a one-time amount to
a rate of movement, for costs that fall on an event rather than a state, such as
the cost of each vaccine dose administered (the vaccination flow) or the cost of
treating each new infection (the incidence flow). Discounting reuses
`heormodel.models._accrual`; the engine augments the system with two
accumulator equations so the same adaptive integrator that advances the
compartments also integrates the discounted reward flows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from heormodel.models._accrual import discount_factor
from heormodel.models._engine import DeterministicEngine
from heormodel.models._interventions import InterventionSpec

Derivatives = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
EventRates = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class ODESpec:
    """One intervention's dynamics and rewards for a single parameter set.

    Returned by the engine's ``dynamics_and_rewards`` function. Arrays are plain
    ``numpy`` arrays over the engine's compartment (state) order.

    Args:
        derivatives: The right-hand side of the system, ``fn(t, y) -> dy/dt``,
            where ``t`` is time in years and ``y`` is the compartment vector,
            shape ``(n_states,)``. Returns the instantaneous rate of change of
            each compartment, same shape.
        initial: Initial compartment sizes at ``t = 0``, shape ``(n_states,)``.
            Their sum is the population scale; costs and effects come out in the
            same units, so an incremental cost-effectiveness ratio is unaffected
            by it.
        state_cost: Per-year cost of each unit of occupancy in each compartment,
            shape ``(n_states,)``. The occupancy integral of this vector is the
            discounted cost that accrues while individuals sit in a compartment.
        state_effect: Per-year effect (quality-adjusted life-years) of each unit
            of occupancy in each compartment, shape ``(n_states,)``.
        event_rates: Optional ``fn(t, y) -> rates`` returning the per-year rate
            of each tracked flow event, shape ``(n_events,)``. A flow event is a
            movement between compartments carrying a one-time amount, such as a
            vaccination or a new infection. ``None`` when the model has no
            flow-based rewards.
        event_cost: One-time cost charged per unit of each flow event, shape
            ``(n_events,)``. Required when ``event_rates`` is given.
        event_effect: One-time effect charged per unit of each flow event, shape
            ``(n_events,)``. Required when ``event_rates`` is given; pass zeros
            when the events carry cost only.
    """

    derivatives: Derivatives
    initial: NDArray[np.float64]
    state_cost: NDArray[np.float64]
    state_effect: NDArray[np.float64]
    event_rates: EventRates | None = None
    event_cost: NDArray[np.float64] | None = None
    event_effect: NDArray[np.float64] | None = None


class ODEModel(DeterministicEngine):
    """Ordinary differential equation (compartmental) model engine.

    ``discount_rate`` is an annual rate discounted continuously, ``exp(-rate *
    t)``, the convention for continuous-time accrual. The horizon is in the same
    time units as the derivatives (years, by convention).

    Args:
        states: Compartment labels; their order fixes every array's axis order.
        interventions: A sequence of intervention names or
            `heormodel.models.Intervention` objects, in the order they appear in
            `Outcomes`. An `Intervention` may carry parameter decision levers
            merged into ``params`` for that intervention.
        dynamics_and_rewards: ``fn(params, intervention) -> ODESpec`` returning
            the system and reward arrays for one intervention under one parameter
            set. ``params`` is a draw-matrix row (a ``pandas.Series``);
            ``intervention`` is the intervention name.
        horizon: Length of the analytic time horizon, in years.
        discount_rate: Annual discount rate for costs and effects (0.03 by
            default), applied continuously.
        method: Integration method passed to ``scipy.integrate.solve_ivp``
            (``"RK45"`` by default; ``"LSODA"`` switches automatically to a stiff
            solver when the dynamics stiffen).
        rtol: Relative error tolerance for the integrator.
        atol: Absolute error tolerance for the integrator.
        max_step: Largest step the integrator may take, in years; ``None`` (the
            default) lets the solver choose. Set it to force the solver through a
            short-lived feature, such as a narrow epidemic peak, it might
            otherwise step over.
        effect: Name of the primary effect column (quality-adjusted life-years by
            default).

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heormodel.models.ode import ODEModel, ODESpec
        >>> def dynamics_and_rewards(params, intervention):
        ...     k = params["decay"]
        ...     return ODESpec(
        ...         derivatives=lambda t, y: np.array([-k * y[0], k * y[0]]),
        ...         initial=np.array([1.0, 0.0]),
        ...         state_cost=np.array([params["cost"], 0.0]),
        ...         state_effect=np.array([1.0, 0.0]))
        >>> engine = ODEModel(
        ...     states=("alive", "dead"), interventions=("care",),
        ...     dynamics_and_rewards=dynamics_and_rewards, horizon=10.0)
        >>> draws = pd.DataFrame({"decay": [0.1], "cost": [1000.0]},
        ...                      index=pd.RangeIndex(1, name="iteration"))
        >>> engine.evaluate(draws).interventions
        ['care']
    """

    def __init__(
        self,
        *,
        states: Sequence[str],
        interventions: InterventionSpec,
        dynamics_and_rewards: Callable[[pd.Series, str], ODESpec],
        horizon: float,
        discount_rate: float = 0.03,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_step: float | None = None,
        effect: str = "qaly",
    ) -> None:
        if len(states) < 2:
            raise ValueError("Provide at least two states.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self._configure(interventions, discount_rate, effect)
        self._states = tuple(states)
        self._n_states = len(self._states)
        self._dynamics_and_rewards = dynamics_and_rewards
        self._horizon = float(horizon)
        self._method = method
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._max_step = np.inf if max_step is None else float(max_step)

    def _check_vector(self, name: str, value: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (self._n_states,):
            raise ValueError(
                f"{name} must have shape {(self._n_states,)}, got {arr.shape}."
            )
        return arr

    def _rewards_of_events(
        self, spec: ODESpec
    ) -> tuple[EventRates | None, NDArray[np.float64], NDArray[np.float64]]:
        """Validate and return the flow-event channel, or empty arrays when unused."""
        if spec.event_rates is None:
            return None, np.zeros(0), np.zeros(0)
        if spec.event_cost is None or spec.event_effect is None:
            raise ValueError(
                "event_cost and event_effect are required when event_rates is given."
            )
        cost = np.asarray(spec.event_cost, dtype=np.float64)
        effect = np.asarray(spec.event_effect, dtype=np.float64)
        if cost.shape != effect.shape or cost.ndim != 1:
            raise ValueError("event_cost and event_effect must be 1-D arrays of equal length.")
        return spec.event_rates, cost, effect

    def _payoff(self, params: pd.Series, intervention: str) -> tuple[float, float]:
        return self._integrate(self._dynamics_and_rewards(params, intervention))

    def _integrate(self, spec: ODESpec) -> tuple[float, float]:
        """Integrate one intervention's system and return discounted (cost, effect).

        The compartments are advanced together with two accumulator equations
        whose derivatives are the discounted reward flows, so both are integrated
        under the same adaptive error control.
        """
        derivatives = spec.derivatives
        initial = self._check_vector("initial", spec.initial)
        state_cost = self._check_vector("state_cost", spec.state_cost)
        state_effect = self._check_vector("state_effect", spec.state_effect)
        event_rates, event_cost, event_effect = self._rewards_of_events(spec)
        rate = self._discount_rate
        n = self._n_states

        def augmented(t: float, y_aug: NDArray[np.float64]) -> NDArray[np.float64]:
            y = y_aug[:n]
            dy = np.asarray(derivatives(t, y), dtype=np.float64)
            if dy.shape != (n,):
                raise ValueError(
                    f"derivatives must return shape {(n,)}, got {dy.shape}."
                )
            cost_flow = float(y @ state_cost)
            effect_flow = float(y @ state_effect)
            if event_rates is not None:
                er = np.asarray(event_rates(t, y), dtype=np.float64)
                if er.shape != event_cost.shape:
                    raise ValueError(
                        f"event_rates must return shape {event_cost.shape}, got {er.shape}."
                    )
                cost_flow += float(er @ event_cost)
                effect_flow += float(er @ event_effect)
            disc = float(discount_factor(t, rate, continuous=True))
            out = np.empty(n + 2, dtype=np.float64)
            out[:n] = dy
            out[n] = disc * cost_flow
            out[n + 1] = disc * effect_flow
            return out

        y0 = np.concatenate([initial, [0.0, 0.0]])
        sol = solve_ivp(
            augmented,
            (0.0, self._horizon),
            y0,
            method=self._method,
            rtol=self._rtol,
            atol=self._atol,
            max_step=self._max_step,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return float(sol.y[n, -1]), float(sol.y[n + 1, -1])

    def trajectory(
        self, params: pd.Series, intervention: str, *, n_points: int = 200
    ) -> pd.DataFrame:
        """Compartment occupancy over the horizon for one parameter set.

        A convenience for inspection and plotting (an epidemic curve, a
        vaccination-coverage path); it is not part of the engine contract and
        does not accrue costs or effects. `evaluate` is what produces `Outcomes`.

        Args:
            params: One draw-matrix row (a ``pandas.Series``) of parameter values.
            intervention: The intervention name passed to
                ``dynamics_and_rewards``.
            n_points: Number of evenly spaced time points returned over
                ``[0, horizon]``.

        Returns:
            A ``DataFrame`` with a ``time`` column and one column per compartment.

        Example:
            >>> import numpy as np, pandas as pd
            >>> from heormodel.models.ode import ODEModel, ODESpec
            >>> def dynamics_and_rewards(params, intervention):
            ...     return ODESpec(
            ...         derivatives=lambda t, y: np.array([-0.1 * y[0], 0.1 * y[0]]),
            ...         initial=np.array([1.0, 0.0]),
            ...         state_cost=np.zeros(2), state_effect=np.array([1.0, 0.0]))
            >>> engine = ODEModel(states=("a", "b"), interventions=("s",),
            ...     dynamics_and_rewards=dynamics_and_rewards, horizon=10.0)
            >>> traj = engine.trajectory(pd.Series(dtype=float), "s", n_points=3)
            >>> list(traj.columns)
            ['time', 'a', 'b']
        """
        spec = self._dynamics_and_rewards(params, intervention)
        initial = self._check_vector("initial", spec.initial)
        times = np.linspace(0.0, self._horizon, n_points)
        sol = solve_ivp(
            lambda t, y: np.asarray(spec.derivatives(t, y), dtype=np.float64),
            (0.0, self._horizon),
            initial,
            method=self._method,
            rtol=self._rtol,
            atol=self._atol,
            max_step=self._max_step,
            t_eval=times,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        frame = pd.DataFrame({"time": sol.t})
        for i, name in enumerate(self._states):
            frame[name] = sol.y[i]
        return frame

