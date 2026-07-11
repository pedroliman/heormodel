"""Tests for the LifeTable age-dependent event-time sampler."""

from __future__ import annotations

import numpy as np
import pytest

from heormodel.models import LifeTable


@pytest.fixture
def two_band() -> LifeTable:
    return LifeTable(ages=[0.0, 60.0], rates=[0.01, 0.1])


class TestConstruction:
    def test_rejects_non_increasing_ages(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            LifeTable(ages=[0.0, 0.0], rates=[0.01, 0.02])

    def test_rejects_nonpositive_rates(self):
        with pytest.raises(ValueError, match="positive"):
            LifeTable(ages=[0.0, 10.0], rates=[0.01, 0.0])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same nonzero length"):
            LifeTable(ages=[0.0, 10.0], rates=[0.01])

    def test_rejects_age_below_table(self, two_band: LifeTable):
        with pytest.raises(ValueError, match="below the first table age"):
            two_band.rate(-1.0)


class TestHazard:
    def test_rate_lookup(self, two_band: LifeTable):
        assert two_band.rate([0.0, 59.9, 60.0, 200.0]).tolist() == [0.01, 0.01, 0.1, 0.1]

    def test_cumulative_hazard_is_piecewise_linear(self, two_band: LifeTable):
        assert float(two_band.cumulative_hazard(30.0)) == pytest.approx(0.3)
        assert float(two_band.cumulative_hazard(60.0)) == pytest.approx(0.6)
        assert float(two_band.cumulative_hazard(70.0)) == pytest.approx(1.6)


class TestSampling:
    def test_constant_rate_matches_exponential(self):
        table = LifeTable(ages=[0.0], rates=[0.05])
        rng = np.random.default_rng(1)
        t = table.sample_time_to_death(rng, np.zeros(200_000))
        assert t.mean() == pytest.approx(20.0, rel=0.01)
        assert np.median(t) == pytest.approx(np.log(2) / 0.05, rel=0.02)

    def test_mean_matches_analytic_life_expectancy(self, two_band: LifeTable):
        rng = np.random.default_rng(2)
        for age in (0.0, 30.0, 65.0):
            t = table_mean = two_band.sample_time_to_death(rng, np.full(200_000, age))
            assert t.mean() == pytest.approx(two_band.life_expectancy(age), rel=0.01)
        assert table_mean is not None

    def test_hazard_ratio_scales_constant_rate(self):
        table = LifeTable(ages=[0.0], rates=[0.02])
        rng = np.random.default_rng(3)
        t = table.sample_time_to_death(rng, np.zeros(200_000), hazard_ratio=4.0)
        assert t.mean() == pytest.approx(1.0 / 0.08, rel=0.01)

    def test_per_individual_hazard_ratio_broadcasts(self):
        table = LifeTable(ages=[0.0], rates=[0.02])
        rng = np.random.default_rng(4)
        hr = np.concatenate([np.ones(100_000), np.full(100_000, 10.0)])
        t = table.sample_time_to_death(rng, np.zeros(200_000), hazard_ratio=hr)
        assert t[:100_000].mean() == pytest.approx(50.0, rel=0.02)
        assert t[100_000:].mean() == pytest.approx(5.0, rel=0.02)

    def test_conditioning_on_age_left_truncates(self, two_band: LifeTable):
        # Sampling at 65 must reflect only the 0.1 band, not the earlier 0.01 band.
        rng = np.random.default_rng(5)
        t = two_band.sample_time_to_death(rng, np.full(200_000, 65.0))
        assert t.mean() == pytest.approx(10.0, rel=0.01)
        assert np.all(t > 0)

    def test_rejects_nonpositive_hazard_ratio(self, two_band: LifeTable):
        with pytest.raises(ValueError, match="hazard_ratio must be positive"):
            two_band.sample_time_to_death(np.random.default_rng(0), [30.0], hazard_ratio=0.0)


class TestLifeExpectancy:
    def test_constant_rate_closed_form(self):
        assert LifeTable(ages=[0.0], rates=[0.02]).life_expectancy(30.0) == 50.0

    def test_hazard_ratio_divides_constant_rate_expectancy(self):
        table = LifeTable(ages=[0.0], rates=[0.02])
        assert table.life_expectancy(0.0, hazard_ratio=2.0) == pytest.approx(25.0)

    def test_mid_band_start(self, two_band: LifeTable):
        # From age 50: 10 years at rate 0.01, then rate 0.1 forever.
        expected = (1 - np.exp(-0.1)) / 0.01 + np.exp(-0.1) * 10.0
        assert two_band.life_expectancy(50.0) == pytest.approx(expected)
