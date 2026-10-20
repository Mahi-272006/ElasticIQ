"""Tests for inventory planning calculations."""

import pytest

from src.inventory.planning import (
    SERVICE_LEVEL_Z,
    InventoryInputs,
    coverage_period_demand,
    recommend,
    reorder_point,
    safety_stock,
)


def test_safety_stock_known_z():
    # z(0.95)=1.6449, std=2/day, lead time 9 days -> 1.6449*2*3 = 9.8694
    value = safety_stock(daily_std=2.0, service_level=0.95, lead_time_days=9)
    assert value == pytest.approx(1.6449 * 2 * 3, rel=1e-3)


def test_safety_stock_higher_service_level_costs_more():
    low = safety_stock(2.0, 0.90, 7)
    high = safety_stock(2.0, 0.99, 7)
    assert high > low


def test_safety_stock_zero_variance():
    assert safety_stock(0.0, 0.95, 7) == 0.0


def test_reorder_point_components():
    inputs = InventoryInputs(service_level=0.95, lead_time_days=7)
    rop = reorder_point(daily_mean=10.0, daily_std=1.0, inputs=inputs)

    expected = 10.0 * 7 + 1.6449 * 1.0 * (7**0.5)
    assert rop == pytest.approx(expected, rel=1e-3)


def test_coverage_period_includes_review_time():
    inputs = InventoryInputs(lead_time_days=7, review_period_days=14)
    assert coverage_period_demand(4.0, inputs) == pytest.approx(4.0 * 21)


def test_recommend_stockout_risk_when_position_low():
    inputs = InventoryInputs(
        service_level=0.95, lead_time_days=7, on_hand_units=5, open_order_units=0
    )
    result = recommend(
        daily_mean=10.0,
        daily_std=2.0,
        horizon_days=28,
        forecast_demand_horizon=280.0,
        inputs=inputs,
    )

    assert result.status == "stockout_risk"
    assert result.suggested_replenishment > 0
    assert any("reorder point" in reason for reason in result.status_reasons)


def test_recommend_overstock_risk_when_position_high():
    inputs = InventoryInputs(
        service_level=0.95,
        lead_time_days=7,
        on_hand_units=3000,
        open_order_units=0,
        overstock_months_threshold=3.0,
    )
    result = recommend(
        daily_mean=10.0,
        daily_std=2.0,
        horizon_days=28,
        forecast_demand_horizon=280.0,
        inputs=inputs,
    )

    assert result.status == "overstock_risk"
    assert result.suggested_replenishment == 0.0
    assert result.expected_days_of_cover == pytest.approx(300.0)


def test_recommend_ok_when_position_covers_rop():
    # ROP ~ 70 + 8.7 = 78.7; position 200 covers it and is not overstocked.
    # Coverage period = lead(7) + review(14) = 21 days -> order-up-to ~ 218.7.
    inputs = InventoryInputs(
        service_level=0.95,
        lead_time_days=7,
        review_period_days=14,
        on_hand_units=200,
        open_order_units=0,
    )
    result = recommend(
        daily_mean=10.0,
        daily_std=2.0,
        horizon_days=28,
        forecast_demand_horizon=280.0,
        inputs=inputs,
    )

    assert result.status == "ok"
    expected_replenishment = (10.0 * 21 + 1.6449 * 2 * 7**0.5) - 200
    assert result.suggested_replenishment == pytest.approx(expected_replenishment)


def test_zero_demand_warns():
    inputs = InventoryInputs(on_hand_units=10)
    result = recommend(
        daily_mean=0.0,
        daily_std=0.0,
        horizon_days=28,
        forecast_demand_horizon=0.0,
        inputs=inputs,
    )

    assert any("zero" in warning for warning in result.warnings)
    assert result.suggested_replenishment >= 0


def test_non_standard_service_level_warns():
    inputs = InventoryInputs(service_level=0.93, on_hand_units=0)
    result = recommend(
        daily_mean=5.0,
        daily_std=1.0,
        horizon_days=7,
        forecast_demand_horizon=35.0,
        inputs=inputs,
    )

    assert any("not in the standard table" in warning for warning in result.warnings)
    assert result.safety_stock > 0


def test_inputs_validation_bounds():
    with pytest.raises(ValueError):
        InventoryInputs(service_level=0.2)  # below 0.5

    with pytest.raises(ValueError):
        InventoryInputs(lead_time_days=0)

    with pytest.raises(ValueError):
        InventoryInputs(on_hand_units=-1)


def test_standard_service_levels_cover_common_cases():
    assert set(SERVICE_LEVEL_Z) >= {0.90, 0.95, 0.99}
