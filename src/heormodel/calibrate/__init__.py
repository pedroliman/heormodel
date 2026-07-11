"""Model calibration (`heormodel.calibrate`).

Approximate Bayesian computation via ``pyabc`` (optional dependency):
translate ``heormodel`` priors, calibrate a simulator to observed targets, and
get back a posterior parameter draw matrix that flows into the standard
probabilistic analysis through the shared iteration index.
"""

from heormodel.calibrate.abc import (
    CalibrationResult,
    TargetSimulator,
    abc_calibrate,
    to_pyabc_prior,
)

__all__ = ["CalibrationResult", "TargetSimulator", "abc_calibrate", "to_pyabc_prior"]
