"""Automated tests for evaluation metrics and energy saving formulas."""

import pytest

from metrics.evaluator import calculate_energy_saving


def test_energy_saving_calculation_normal():
    """Verify formula: ((baseline - proposed) / baseline) * 100."""
    baseline = 1000.0
    proposed = 800.0
    # Expected: (1000 - 800) / 1000 * 100 = 20.0%
    saving = calculate_energy_saving(baseline, proposed)
    assert saving == 20.0


def test_energy_saving_calculation_zero_or_negative_baseline():
    """Verify safe handling of zero or negative baseline energy."""
    saving_zero = calculate_energy_saving(0.0, 100.0)
    assert saving_zero == 0.0

    saving_neg = calculate_energy_saving(-50.0, 100.0)
    assert saving_neg == 0.0


def test_energy_saving_when_proposed_consumes_more():
    """Verify negative energy saving when proposed uses more energy."""
    baseline = 500.0
    proposed = 600.0
    # (500 - 600) / 500 * 100 = -20.0%
    saving = calculate_energy_saving(baseline, proposed)
    assert saving == -20.0
