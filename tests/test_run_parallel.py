"""Tests for parallel-by-default `run_psa` and the progress reporter."""

import io

import pandas as pd
import pytest

from heval.models import Outcomes
from heval.params import Normal, ParameterSet, Uniform
from heval.run import run_psa
from heval.run._progress import ProgressReporter, resolve_enabled


def _model(draws: pd.DataFrame) -> Outcomes:
    costs = pd.DataFrame({"A": draws["c"], "B": draws["c"] * 2}, index=draws.index)
    effects = pd.DataFrame({"A": draws["e"], "B": draws["e"] + 0.5}, index=draws.index)
    return Outcomes.from_wide(costs, effects)


def _draws() -> pd.DataFrame:
    ps = ParameterSet({"c": Uniform(100, 200), "e": Normal(1.0, 0.1)})
    return ps.sample(40, seed=9)


class TestNumbersIndependentOfWorkers:
    def test_sequential_n_jobs_one_and_two_are_identical(self):
        draws = _draws()
        seq = run_psa(_model, draws, sequential=True, progress=False)
        one = run_psa(_model, draws, n_jobs=1, progress=False)
        two = run_psa(_model, draws, n_jobs=2, progress=False)
        pd.testing.assert_frame_equal(seq.data, one.data)
        pd.testing.assert_frame_equal(seq.data, two.data)

    def test_sequential_overrides_n_jobs(self):
        draws = _draws()
        forced = run_psa(_model, draws, sequential=True, n_jobs=4, progress=False)
        default = run_psa(_model, draws, progress=False)
        pd.testing.assert_frame_equal(forced.data, default.data)

    def test_single_iteration_falls_back(self):
        draws = _draws().iloc[[0]]
        out = run_psa(_model, draws, n_jobs=4, progress=False)
        assert out.n_iterations == 1

    def test_default_parallel_matches_reference_numbers(self):
        # The default run is parallel; its summary must match committed numbers.
        out = run_psa(_model, _draws(), progress=False)
        summary = out.summary()
        assert summary.loc["A", "cost"] == pytest.approx(139.582039, abs=1e-5)
        assert summary.loc["B", "cost"] == pytest.approx(279.164078, abs=1e-5)
        assert summary.loc["A", "qaly"] == pytest.approx(1.016360, abs=1e-5)
        assert summary.loc["B", "qaly"] == pytest.approx(1.516360, abs=1e-5)


class TestProgressReporter:
    def test_remaining_is_finite_and_non_increasing(self):
        # A constant per-experiment rate: elapsed advances 2s per step.
        times = iter([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        r = ProgressReporter(5, enabled=True, stream=io.StringIO(), clock=lambda: next(times))
        seen = []
        for done in range(1, 6):
            r.advance(done)
            seen.append(r.remaining_seconds)
        assert all(x != float("inf") for x in seen)
        assert seen == sorted(seen, reverse=True)
        assert seen[-1] == 0.0

    def test_disabled_writes_nothing(self):
        buf = io.StringIO()
        r = ProgressReporter(10, enabled=False, stream=buf)
        r.advance(3)
        r.close()
        assert buf.getvalue() == ""

    def test_non_tty_auto_off(self):
        assert resolve_enabled(None, io.StringIO()) is False
        assert resolve_enabled(True, io.StringIO()) is True
        assert resolve_enabled(False, io.StringIO()) is False

    def test_enabled_line_format(self):
        times = iter([0.0, 12.0])
        buf = io.StringIO()
        r = ProgressReporter(1000, enabled=True, stream=buf, clock=lambda: next(times))
        r.advance(320)
        assert (
            buf.getvalue().strip()
            == "running_psa: 320/1000 experiments, 0:00:12 elapsed, ~0:00:25 remaining"
        )


class TestProgressInRun:
    def test_default_run_is_quiet_without_tty(self, capsys):
        # pytest captures stderr, so it is not a TTY: no readout by default.
        run_psa(_model, _draws())
        assert capsys.readouterr().err == ""

    def test_progress_false_is_quiet_even_if_forced_parallel(self, capsys):
        run_psa(_model, _draws(), n_jobs=2, progress=False)
        assert capsys.readouterr().err == ""

    def test_progress_true_writes_to_stderr(self, capsys):
        run_psa(_model, _draws(), sequential=True, progress=True)
        err = capsys.readouterr().err
        assert "running_psa:" in err
        assert "experiments" in err
