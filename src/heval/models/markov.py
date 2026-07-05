"""Cohort state-transition (Markov) engine.

`MarkovModel` evaluates a cohort state-transition model across PSA
iterations and emits the standard `Outcomes` schema. The cohort trace is a
matrix-power sweep: the state-occupancy vector is multiplied by a transition
matrix each cycle. Transitions may be constant or vary by cycle, which is how
age-dependent mortality enters.

The engine configures once and evaluates on draws. The constructor takes the
model structure (states, strategies, a ``model_fn`` callback, cycle count,
discounting, within-cycle correction); ``evaluate`` takes only the parameter
draw matrix and returns `Outcomes` indexed by ``draws.index``. Cohort models
are deterministic given a parameter set, so no random streams are involved.

Rewards follow the transition-dynamics convention. State rewards accrue on the
occupancy trace. Optional transition rewards accrue on the flow between states,
so a one-time cost of dying or a disutility of onset attaches to the transition
rather than to a state. Discounting reuses `heval.models._accrual`, and the
within-cycle correction offers Simpson's 1/3 rule, the half-cycle weights, or
none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from heval.models._accrual import discount_factor
from heval.models.outcomes import ITERATION_LEVEL, STRATEGY_LEVEL, Outcomes

_PROB_TOL = 1e-8


@dataclass
class CohortSpec:
    """One strategy's matrices for a single parameter set.

    Returned by the engine's ``model_fn`` callback. Arrays are plain ``numpy``
    arrays over the engine's state order.

    Args:
        transition: Transition-probability matrix, shape ``(n_states,
            n_states)`` for a time-independent model or ``(n_cycles, n_states,
            n_states)`` when transitions vary by cycle. Each row sums to 1.
        state_cost: Per-cycle cost of occupying each state, shape
            ``(n_states,)``, or ``(n_cycles + 1, n_states)`` when it varies.
        state_effect: Per-cycle effect (QALYs) of each state, same shape rule
            as ``state_cost``.
        transition_cost: Optional extra cost attached to a transition, added on
            the flow between states, shape ``(n_states, n_states)`` or
            ``(n_cycles, n_states, n_states)``. Entry ``[i, j]`` is the cost of
            moving from state ``i`` to state ``j``.
        transition_effect: Optional effect attached to a transition, same shape
            rule as ``transition_cost``.
    """

    transition: NDArray[np.float64]
    state_cost: NDArray[np.float64]
    state_effect: NDArray[np.float64]
    transition_cost: NDArray[np.float64] | None = None
    transition_effect: NDArray[np.float64] | None = None


def gen_wcc(n_cycles: int, method: str = "simpson") -> NDArray[np.float64]:
    """Within-cycle correction weights over the ``n_cycles + 1`` cycle points.

    Args:
        n_cycles: Number of cycles (transitions) in the model horizon.
        method: ``"simpson"`` for Simpson's 1/3 rule, ``"half-cycle"`` for
            half weights on the first and last point, or ``"none"`` for unit
            weights.

    Returns:
        Weight vector of length ``n_cycles + 1``.

    Example:
        >>> from heval.models.markov import gen_wcc
        >>> gen_wcc(4, "half-cycle").tolist()
        [0.5, 1.0, 1.0, 1.0, 0.5]
        >>> [round(float(w), 3) for w in gen_wcc(4, "simpson")]
        [0.333, 0.667, 1.333, 0.667, 0.333]
    """
    if n_cycles <= 0:
        raise ValueError("n_cycles must be positive.")
    n_points = n_cycles + 1
    if method == "simpson":
        positions = np.arange(n_points)
        wcc = np.where(positions % 2 == 1, 2.0 / 3.0, 4.0 / 3.0)
        wcc[0] = wcc[-1] = 1.0 / 3.0
    elif method == "half-cycle":
        wcc = np.ones(n_points)
        wcc[0] = wcc[-1] = 0.5
    elif method == "none":
        wcc = np.ones(n_points)
    else:
        raise ValueError(f"Unknown within-cycle correction method {method!r}.")
    return wcc


class MarkovModel:
    """Cohort state-transition model engine.

    ``discount_rate`` is an annual rate on an annual clock. ``cycle_length``
    scales the clock: with ``cycle_length=0.5`` each cycle discounts by half a
    year.

    Args:
        states: State labels; their order fixes every array's axis order.
        strategies: Strategy names, in the order they appear in `Outcomes`.
        model_fn: ``fn(params, strategy) -> CohortSpec`` returning the
            transition matrix and reward arrays for one strategy under one
            parameter set. ``params`` is a draw-matrix row (a ``pandas.Series``);
            ``strategy`` is one of ``strategies``.
        n_cycles: Number of cycles in the time horizon.
        start: Initial state distribution: a state label (all mass there), a
            mapping of state label to probability, or a length-``n_states``
            array. Defaults to all mass in the first state.
        cycle_length: Years per cycle; scales the discount clock.
        discount_rate: Annual discount rate for costs and effects (0.03 by
            default).
        half_cycle_correction: ``"simpson"`` (default), ``"half-cycle"``, or
            ``"none"``; see `gen_wcc`.
        effect: Name of the primary effect column (QALYs by default).

    Example:
        >>> import numpy as np, pandas as pd
        >>> from heval.models.markov import CohortSpec, MarkovModel
        >>> def model(params, strategy):
        ...     p = params["p_die"]
        ...     P = np.array([[1 - p, p], [0.0, 1.0]])
        ...     return CohortSpec(P, np.array([params["cost"], 0.0]),
        ...                       np.array([1.0, 0.0]))
        >>> engine = MarkovModel(
        ...     states=("alive", "dead"), strategies=("care",), model_fn=model,
        ...     n_cycles=10, half_cycle_correction="none")
        >>> draws = pd.DataFrame({"p_die": [0.1], "cost": [1000.0]},
        ...                      index=pd.RangeIndex(1, name="iteration"))
        >>> engine.evaluate(draws).strategies
        ['care']
    """

    def __init__(
        self,
        *,
        states: Sequence[str],
        strategies: Sequence[str],
        model_fn: Callable[[pd.Series, str], CohortSpec],
        n_cycles: int,
        start: str | Mapping[str, float] | Sequence[float] | None = None,
        cycle_length: float = 1.0,
        discount_rate: float = 0.03,
        half_cycle_correction: str = "simpson",
        effect: str = "qaly",
    ) -> None:
        if len(states) < 2:
            raise ValueError("Provide at least two states.")
        if not strategies:
            raise ValueError("Provide at least one strategy.")
        if n_cycles < 1:
            raise ValueError("n_cycles must be at least one.")
        self._states = tuple(states)
        self._n_states = len(self._states)
        self._strategies = tuple(strategies)
        self._model_fn = model_fn
        self._n_cycles = int(n_cycles)
        self._cycle_length = float(cycle_length)
        self._discount_rate = float(discount_rate)
        self._effect = effect
        self._start = self._resolve_start(start)
        times = np.arange(self._n_cycles + 1, dtype=np.float64) * self._cycle_length
        self._disc = discount_factor(times, self._discount_rate)
        self._wcc = gen_wcc(self._n_cycles, half_cycle_correction)

    def _resolve_start(
        self, start: str | Mapping[str, float] | Sequence[float] | None
    ) -> NDArray[np.float64]:
        vec = np.zeros(self._n_states, dtype=np.float64)
        if start is None:
            vec[0] = 1.0
        elif isinstance(start, str):
            if start not in self._states:
                raise ValueError(f"Unknown start state {start!r}; states are {self._states}.")
            vec[self._states.index(start)] = 1.0
        elif isinstance(start, Mapping):
            for name, prob in start.items():
                if name not in self._states:
                    raise ValueError(f"Unknown start state {name!r}; states are {self._states}.")
                vec[self._states.index(name)] = float(prob)
        else:
            vec = np.asarray(start, dtype=np.float64)
            if vec.shape != (self._n_states,):
                raise ValueError(f"start array must have length {self._n_states}.")
        if not np.isclose(vec.sum(), 1.0, atol=_PROB_TOL):
            raise ValueError("Initial state distribution must sum to 1.")
        return vec

    def _trace(self, transition: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cohort occupancy trace, shape ``(n_cycles + 1, n_states)``."""
        P = np.asarray(transition, dtype=np.float64)
        per_cycle = P.ndim == 3
        if per_cycle:
            if P.shape != (self._n_cycles, self._n_states, self._n_states):
                raise ValueError(
                    f"per-cycle transition must have shape "
                    f"{(self._n_cycles, self._n_states, self._n_states)}, got {P.shape}."
                )
        elif P.shape != (self._n_states, self._n_states):
            raise ValueError(
                f"transition must have shape {(self._n_states, self._n_states)} or "
                f"{(self._n_cycles, self._n_states, self._n_states)}, got {P.shape}."
            )
        if P.min() < -_PROB_TOL or P.max() > 1.0 + _PROB_TOL:
            raise ValueError("transition probabilities must lie in [0, 1].")
        if not np.allclose(P.sum(axis=-1), 1.0, atol=_PROB_TOL):
            raise ValueError("transition rows must each sum to 1.")
        trace = np.empty((self._n_cycles + 1, self._n_states), dtype=np.float64)
        trace[0] = self._start
        for t in range(self._n_cycles):
            trace[t + 1] = trace[t] @ (P[t] if per_cycle else P)
        return trace

    def _state_reward(
        self, trace: NDArray[np.float64], reward: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r = np.asarray(reward, dtype=np.float64)
        if r.shape == (self._n_states,):
            return trace @ r
        if r.shape == (self._n_cycles + 1, self._n_states):
            return np.einsum("ts,ts->t", trace, r)
        raise ValueError(
            f"state reward must have shape {(self._n_states,)} or "
            f"{(self._n_cycles + 1, self._n_states)}, got {r.shape}."
        )

    def _transition_reward(
        self,
        trace: NDArray[np.float64],
        transition: NDArray[np.float64],
        reward: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        P = np.asarray(transition, dtype=np.float64)
        R = np.asarray(reward, dtype=np.float64)
        out = np.zeros(self._n_cycles + 1, dtype=np.float64)
        for t in range(self._n_cycles):
            Pt = P[t] if P.ndim == 3 else P
            Rt = R[t] if R.ndim == 3 else R
            flow = trace[t][:, None] * Pt  # diag(trace[t]) @ Pt
            out[t + 1] = float(np.sum(flow * Rt))
        return out

    def _accrue(self, spec: CohortSpec) -> tuple[float, float]:
        trace = self._trace(spec.transition)
        cost_cycle = self._state_reward(trace, spec.state_cost)
        eff_cycle = self._state_reward(trace, spec.state_effect)
        if spec.transition_cost is not None:
            cost_cycle = cost_cycle + self._transition_reward(
                trace, spec.transition, spec.transition_cost
            )
        if spec.transition_effect is not None:
            eff_cycle = eff_cycle + self._transition_reward(
                trace, spec.transition, spec.transition_effect
            )
        total_cost = float(cost_cycle @ (self._disc * self._wcc))
        total_effect = float(eff_cycle @ (self._disc * self._wcc))
        return total_cost, total_effect

    def evaluate(self, draws: pd.DataFrame) -> Outcomes:
        """Evaluate every strategy on every draw and return `Outcomes`.

        Args:
            draws: Parameter draw matrix (rows = iterations). Its index becomes
                the outcome iteration index.

        Returns:
            `Outcomes` indexed by ``(strategy, draws.index)``.
        """
        if draws.empty:
            raise ValueError("draws is empty.")
        costs: list[float] = []
        effects: list[float] = []
        keys: list[tuple[str, object]] = []
        for label, (_, params) in zip(draws.index, draws.iterrows(), strict=True):
            for name in self._strategies:
                spec = self._model_fn(params, name)
                cost, effect = self._accrue(spec)
                costs.append(cost)
                effects.append(effect)
                keys.append((name, label))
        index = pd.MultiIndex.from_tuples(keys, names=[STRATEGY_LEVEL, ITERATION_LEVEL])
        data = pd.DataFrame({"cost": costs, self._effect: effects}, index=index)
        full_index = pd.MultiIndex.from_product(
            [self._strategies, draws.index], names=[STRATEGY_LEVEL, ITERATION_LEVEL]
        )
        return Outcomes(data.reindex(full_index), effect=self._effect)
