"""Model calibration (`heval.calibrate`).

Approximate Bayesian computation via ``pyabc`` (optional dependency):
translate ``heval`` priors, calibrate a simulator to observed targets, and
get back a posterior parameter draw matrix that flows into the standard
PSA pipeline through the shared iteration index.
"""

from heval.calibrate.abc import (
    CalibrationResult,
    TargetSimulator,
    abc_calibrate,
    to_pyabc_prior,
)

__all__ = ["CalibrationResult", "TargetSimulator", "abc_calibrate", "to_pyabc_prior"]
