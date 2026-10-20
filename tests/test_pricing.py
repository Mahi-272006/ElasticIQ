"""Tests for the pricing scenario engine, optimizer, and associations."""

import numpy as np
import pytest

from src.pricing.associations import (
    generate_simulated_experiment,
    pooled_association,
    sensitivity_range,
    simulate_ols_estimate,
    within_series_association,
)
from src.pricing.optimize import OptimizeRequest, optimize_price
from src.pricing.scenario import (
    PricingConstraints,
    PricingScenarioRequest,
    demand_sensitivity_check,
    price_is_permitted,
    projected_units,
    simulate_scenario,
)


def make_request(**overrides) -> PricingScenarioRequest:
    defaults = dict(
        current_price=4.0,
        candidate_price=3.6,
        unit_cost=2.0,
        baseline_units=100.0,
        demand_model="elasticity",
        elasticity=-1.5,
        horizon_days=28,
        constraints=PricingConstraints(min_price=2.5, max_price=6.0),
    )
    defaults.update(overrides)
    return PricingScenarioRequest(**defaults)


# ------------------------------------------------------------- scenarios ---


def test_constant_model_holds_demand_flat():
    request = make_request(demand_model="constant", candidate_price=1.0)
    units, warnings = projected_units(request)

    assert units == 100.0
    assert any("constant" in w.lower() for w in warnings)


def test_elasticity_model_math():
    # A 10% price cut with elasticity -2: demand ratio = (0.9)^-2 = 1.2346
    request = make_request(candidate_price=3.6, elasticity=-2.0)
    units, _ = projected_units(request)

    assert units == pytest.approx(100.0 * 0.9**-2)


def test_elasticity_requires_parameter():
    request = make_request(elasticity=None)

    with pytest.raises(ValueError, match="elasticity parameter required"):
        projected_units(request)


def test_linear_slope_negative_units_floored():
    # slope -150 units per currency unit, +1.0 price move -> -50 units -> floored
    request = make_request(
        demand_model="linear_slope", linear_slope=-150.0, candidate_price=5.0
    )
    units, warnings = projected_units(request)

    assert units == 0.0
    assert any("negative units" in w for w in warnings)


def test_scenario_arithmetic():
    request = make_request(demand_model="constant")
    outcome = simulate_scenario(request)

    assert outcome.current_revenue == pytest.approx(400.0)
    assert outcome.candidate_revenue == pytest.approx(360.0)
    assert outcome.current_gross_profit == pytest.approx(200.0)
    assert outcome.candidate_gross_profit == pytest.approx(160.0)
    assert outcome.profit_change_percent == pytest.approx(-20.0)


def test_inventory_cap_limits_units():
    request = make_request(
        demand_model="constant",
        constraints=PricingConstraints(
            min_price=2.5, max_price=6.0, inventory_cap_units=60
        ),
    )
    outcome = simulate_scenario(request)

    assert outcome.expected_units_candidate == 60.0
    assert outcome.candidate_revenue == pytest.approx(60 * 3.6)
    assert any("inventory cap" in w for w in outcome.warnings)


# ------------------------------------------------------------ constraints ---


def test_price_bounds_enforced():
    request = make_request()
    assert price_is_permitted(6.5, request)  # above max_price
    assert price_is_permitted(2.0, request)  # below min_price
    assert not price_is_permitted(4.2, request)


def test_max_change_percent_enforced():
    request = make_request(
        constraints=PricingConstraints(
            min_price=1.0, max_price=10.0, max_change_percent=5.0
        )
    )
    violations = price_is_permitted(4.5, request)  # +12.5%

    assert any("max_change_percent" in v for v in violations)


def test_min_margin_enforced():
    request = make_request(
        constraints=PricingConstraints(
            min_price=1.0, max_price=10.0, min_margin_percent=50.0
        )
    )
    violations = price_is_permitted(3.9, request)  # margin = (3.9-2)/3.9 ~ 48.7%

    assert any("min_margin_percent" in v for v in violations)


def test_constraints_validator_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="max_price must exceed min_price"):
        PricingConstraints(min_price=10.0, max_price=5.0)


# ------------------------------------------------------------- optimizer ---


def test_optimizer_respects_constraints_and_objective():
    request = OptimizeRequest(
        current_price=4.0,
        unit_cost=2.0,
        baseline_units=100.0,
        demand_model="elasticity",
        elasticity=-1.5,
        constraints={"min_price": 2.5, "max_price": 6.0, "inventory_cap_units": 150},
    )
    result = optimize_price(request)

    assert 2.5 <= result.optimal_price <= 6.0
    assert "gross profit" in result.objective
    assert result.optimal_gross_profit >= result.current_gross_profit
    assert any("not a causal claim" in a for a in result.assumptions)


def test_optimizer_infeasible_raises():
    request = OptimizeRequest(
        current_price=4.0,
        unit_cost=2.0,
        baseline_units=100.0,
        demand_model="constant",
        # min_margin 99% impossible with cost 2 and max price 3
        constraints={"min_price": 1.0, "max_price": 3.0, "min_margin_percent": 99.0},
    )

    with pytest.raises(ValueError, match="No feasible price"):
        optimize_price(request)


def test_optimizer_under_uncertainty_spans_conclusions():
    checks = demand_sensitivity_check(
        baseline_units=100.0,
        current_price=4.0,
        candidate_price=4.4,
        unit_cost=2.0,
        constraint=PricingConstraints(min_price=1.0, max_price=10.0),
    )

    assert len(checks) == 5
    # Inelastic vs elastic assumptions should disagree on profitability.
    profits = [check["profit_change_percent"] for check in checks]
    assert min(profits) < 0 < max(profits) or max(profits) < 0


# ---------------------------------------------------------- associations ---


def test_within_series_association_detects_negative_slope():
    rng = np.random.default_rng(1)
    prices = rng.uniform(2, 6, 200)
    # Large base demand keeps the log(units + 1) offset negligible.
    demand = 500 * prices**-1.5 * rng.lognormal(0, 0.05, 200)

    result = within_series_association(demand, prices)

    assert result.slope < -1.0
    assert result.slope == pytest.approx(-1.5, abs=0.1)
    assert any("ASSOCIATION ONLY" in d for d in result.diagnostics)


def test_within_series_requires_price_variation():
    prices = np.full(50, 3.0)
    units = np.random.default_rng(0).uniform(0, 5, 50)

    result = within_series_association(units, prices)

    assert np.isnan(result.slope)
    assert any("not identifiable" in d for d in result.diagnostics)


def test_pooled_association_uses_fixed_effects():
    data = generate_simulated_experiment(n_series=15, n_days=100, seed=3)
    panel = data.frame.rename(columns={"series_id": "id", "price": "sell_price"})
    result = pooled_association(panel)

    assert result.n_observations == 15 * 100
    assert any("fixed effects" in d for d in result.diagnostics)


def test_sensitivity_range_reports_instability():
    data = generate_simulated_experiment(n_series=12, n_days=90, seed=5)
    panel = data.frame.rename(columns={"series_id": "id", "price": "sell_price"})
    summary = sensitivity_range(panel)

    assert summary["series_with_identifiable_slope"] > 0
    assert "note" in summary


def test_simulated_randomized_recovers_true_elasticity():
    data = generate_simulated_experiment(
        n_series=25, n_days=200, true_elasticity=-1.3, seed=11, randomized=True
    )
    result = simulate_ols_estimate(data)

    assert result["design"] == "randomized"
    assert result["estimated_elasticity_median"] == pytest.approx(-1.3, abs=0.15)
    assert abs(result["bias"]) < 0.15
    assert "SIMULATED" in result["label"]


def test_simulated_confounded_bias_is_visible():
    data = generate_simulated_experiment(
        n_series=25, n_days=200, true_elasticity=-1.3, seed=11, randomized=False
    )
    result = simulate_ols_estimate(data)

    assert result["design"] == "confounded"
    # The hidden promotion lifts demand when price drops: the naive
    # estimate becomes MORE negative than the truth.
    assert result["estimated_elasticity_median"] < -1.3 - 0.1
