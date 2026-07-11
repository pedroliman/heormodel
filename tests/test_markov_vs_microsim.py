"""Cross-validate the Markov cohort and microsimulation engines.

Two checks anchor the file. The homogeneous microsimulation mean converges to
the cohort trace at a large population and a fixed seed, so the two engines are
the same model (the cross-validation). Adding a mean-1 frailty then moves the
microsimulation mean off the cohort trace by far more than Monte Carlo noise, in
the direction frailty selection predicts (higher survival, so higher QALYs), so
the divergence is a property of the model and not a bug.

The model matches ``examples/markov_vs_microsim.py``: a progressive
Healthy-Sick-Sicker-Dead model over 40 annual cycles, with onset, progression,
and death as competing annual hazards. Frailty multiplies the progression and
mortality hazards.
"""

import numpy as np
import pandas as pd
import pytest

from heormodel.models import CohortSpec, MarkovModel, MicrosimModel
from heormodel.run import SeedManager, run_psa

STATES = ("H", "S1", "S2", "D")
N_CYCLES = 40
STRATEGY = "Standard of care"
BASE = dict(
    r_HS1=0.12, r_S1S2=0.10, r_HD=0.010, hr_S1=3.0, hr_S2=10.0,
    c_H=1_000.0, c_S1=4_000.0, c_S2=15_000.0, u_H=1.0, u_S1=0.75, u_S2=0.5,
)
COST = np.array([BASE["c_H"], BASE["c_S1"], BASE["c_S2"], 0.0])
EFF = np.array([BASE["u_H"], BASE["u_S1"], BASE["u_S2"], 0.0])


def _draws():
    return pd.DataFrame([BASE], index=pd.RangeIndex(1, name="iteration"))


def _hazards(p, state, z):
    haz = np.zeros((len(state), 4))
    h, s1, s2 = state == 0, state == 1, state == 2
    haz[h, 1] = p["r_HS1"]
    haz[h, 3] = p["r_HD"] * z[h]
    haz[s1, 2] = p["r_S1S2"] * z[s1]
    haz[s1, 3] = p["r_HD"] * p["hr_S1"] * z[s1]
    haz[s2, 3] = p["r_HD"] * p["hr_S2"] * z[s2]
    return haz


def _rows_from(haz):
    total = haz.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total[:, None] > 0, haz / total[:, None], 0.0)
    return (1.0 - np.exp(-total))[:, None] * share


def _cohort_model(p, strategy):
    P = _rows_from(_hazards(p, np.arange(4), np.ones(4)))
    P[np.arange(4), np.arange(4)] += 1.0 - P.sum(axis=1)
    P[3] = [0.0, 0.0, 0.0, 1.0]
    return CohortSpec(P, COST, EFF)


def _make_pop(var):
    def population(rng, n):
        z = np.ones(n) if var == 0 else rng.gamma(1 / var, var, n)
        return pd.DataFrame({"z": z})

    return population


def _transition(p, state, attrs, rng):
    probs = _rows_from(_hazards(p, state, attrs["z"].to_numpy()))
    probs[np.arange(len(state)), state] += 1.0 - probs.sum(axis=1)
    probs[state == 3] = [0.0, 0.0, 0.0, 1.0]
    return probs


def _payoffs(p, state, attrs):
    return COST[state], EFF[state]


def _microsim(n, var, seed=1):
    return MicrosimModel(
        states=STATES,
        transition_probabilities=_transition,
        state_costs_and_utilities=_payoffs,
        population=_make_pop(var),
        n_individuals=n, strategies={STRATEGY: {}}, horizon=N_CYCLES,
        discount_rate=0.03, half_cycle_correction=True, seed_manager=SeedManager(seed),
    )


def _cohort():
    return MarkovModel(
        states=STATES, strategies=(STRATEGY,), model_fn=_cohort_model,
        n_cycles=N_CYCLES, start="H", discount_rate=0.03,
        half_cycle_correction="half-cycle",
    )


def _summary(model):
    return run_psa(model, _draws(), sequential=True).summary().loc[STRATEGY]


def test_homogeneous_microsim_converges_to_cohort():
    """At 40,000 individuals the homogeneous microsim matches the cohort trace."""
    cohort = _cohort().evaluate(_draws()).summary().loc[STRATEGY]
    micro = _summary(_microsim(40_000, 0.0))
    # Measured gap at this seed is under half a percent; 1% leaves clear margin.
    assert micro["cost"] == pytest.approx(cohort["cost"], rel=0.01)
    assert micro["qaly"] == pytest.approx(cohort["qaly"], rel=0.01)


def test_heterogeneity_raises_qalys_beyond_noise():
    """A mean-1 frailty raises microsim QALYs well above the cohort trace."""
    cohort = _cohort().evaluate(_draws()).summary().loc[STRATEGY]
    homo = _summary(_microsim(40_000, 0.0))
    het = _summary(_microsim(40_000, 0.5))

    homo_gap = abs(homo["qaly"] - cohort["qaly"]) / cohort["qaly"]
    het_gap = (het["qaly"] - cohort["qaly"]) / cohort["qaly"]
    # Frailty selection lifts survival, so QALYs rise, and the shift dwarfs the
    # homogeneous Monte Carlo gap at the same population and seed.
    assert het_gap > 0.03
    assert het_gap > 10 * homo_gap
