"""Tests for the shared `Intervention` value object and intervention normalisation."""

import pandas as pd
import pytest

from heormodel.models import Intervention
from heormodel.models._interventions import (
    comparator_of,
    merge_decision_levers,
    normalize_interventions,
)


def test_intervention_defaults_to_no_decision_levers():
    assert Intervention("care").decision_levers == {}
    assert Intervention("care", {"n": 2}).decision_levers == {"n": 2}


def test_intervention_defaults_to_not_a_comparator():
    assert Intervention("care").is_comparator is False
    assert Intervention("care", is_comparator=True).is_comparator is True


def test_normalize_mixes_names_and_intervention_objects():
    got = normalize_interventions(["A", Intervention("B", {"scale": 2.0})])
    assert got == {"A": {}, "B": {"scale": 2.0}}


def test_normalize_rejects_a_non_intervention_item():
    with pytest.raises(TypeError, match="names or Intervention objects"):
        normalize_interventions([object()])


def test_normalize_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate intervention name"):
        normalize_interventions(["A", Intervention("A", {"x": 1})])


def test_normalize_rejects_empty_sequence():
    with pytest.raises(ValueError, match="at least one intervention"):
        normalize_interventions([])


def test_comparator_of_finds_the_flagged_intervention():
    assert comparator_of(["A", Intervention("B", is_comparator=True)]) == "B"


def test_comparator_of_returns_none_when_unset():
    assert comparator_of(["A", "B"]) is None


def test_comparator_of_rejects_more_than_one_flag():
    flagged = [Intervention("A", is_comparator=True), Intervention("B", is_comparator=True)]
    with pytest.raises(ValueError, match="At most one intervention"):
        comparator_of(flagged)


def test_merge_decision_levers_leaves_the_original_row_untouched():
    params = pd.Series({"a": 1.0, "b": 2.0})
    merged = merge_decision_levers(params, {"b": 5.0})
    assert merged["b"] == 5.0
    assert params["b"] == 2.0  # original unchanged


def test_merge_decision_levers_returns_the_same_row_when_empty():
    params = pd.Series({"a": 1.0})
    assert merge_decision_levers(params, {}) is params
