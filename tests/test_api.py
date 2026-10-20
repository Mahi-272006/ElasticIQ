"""End-to-end API tests against isolated fixtures (tmp Parquet + tmp SQLite)."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient


def test_health_reports_ok(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database"]["connected"] is True
    assert payload["data"]["data_ready"] is True
    assert payload["model"]["ready"] is True
    assert payload["model"]["version"] == "stub_v1"


def test_root_points_to_docs(api_client: TestClient) -> None:
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json()["api_base"] == "/api/v1"


def test_overview_shape(api_client: TestClient) -> None:
    payload = api_client.get("/api/v1/overview").json()
    assert payload["coverage"]["series_in_catalog"] == 120
    assert payload["coverage"]["days"] == 60
    assert payload["totals"]["total_units"] > 0
    assert {c["key"] for c in payload["categories"]} == {"FOODS", "HOBBIES"}
    assert payload["model"]["version"] == "stub_v1"
    assert payload["model"]["wape_percent_by_horizon"]["7"] == 45.0


def test_products_pagination_and_filters(api_client: TestClient) -> None:
    page = api_client.get("/api/v1/products?limit=5&offset=0").json()
    assert page["total"] == 120
    assert len(page["items"]) == 5
    assert page["items"][0]["series_id"] == "FOODS_3_090_CA_1"

    filtered = api_client.get("/api/v1/products?store_id=CA_2&limit=200").json()
    assert filtered["total"] == 117

    search = api_client.get("/api/v1/products?search=GEN_0001").json()
    assert search["total"] == 1
    assert search["items"][0]["series_id"] == "GEN_0001_CA_2"


def test_product_detail_and_404(api_client: TestClient) -> None:
    detail = api_client.get("/api/v1/products/FOODS_3_090_CA_1")
    assert detail.status_code == 200
    payload = detail.json()
    assert len(payload["history"]) == 60
    assert payload["price_stats"]["store_id"] == "CA_1"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", payload["history"][0]["date"])
    assert 0.0 <= payload["zero_demand_share"] <= 1.0

    missing = api_client.get("/api/v1/products/NOPE")
    assert missing.status_code == 404


def test_stores_list(api_client: TestClient) -> None:
    payload = api_client.get("/api/v1/stores").json()
    assert payload["total"] == 3
    keys = {item["store_id"] for item in payload["items"]}
    assert keys == {"CA_1", "CA_2", "TX_1"}
    shares = sum(item["share_percent"] for item in payload["items"])
    assert shares == 100.0 or abs(shares - 100.0) < 0.01


def test_trends_filters_and_limitation(api_client: TestClient) -> None:
    default = api_client.get("/api/v1/sales/trends?days=10").json()
    assert default["source_aggregate"] == "daily_total_units"
    assert len(default["points"]) == 10
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", default["points"][0]["date"])

    by_store = api_client.get("/api/v1/sales/trends?days=5&store_id=CA_1").json()
    assert by_store["source_aggregate"] == "daily_store_units"
    assert len(by_store["points"]) == 5

    by_cat = api_client.get("/api/v1/sales/trends?days=5&cat_id=FOODS").json()
    assert by_cat["source_aggregate"] == "daily_category_units"

    combined = api_client.get("/api/v1/sales/trends?store_id=CA_1&cat_id=FOODS")
    assert combined.status_code == 422
    assert "not supported" in combined.json()["detail"]

    unknown = api_client.get("/api/v1/sales/trends?store_id=ZZ_9")
    assert unknown.status_code == 404


def test_forecast_roundtrip(api_client: TestClient) -> None:
    created = api_client.post(
        "/api/v1/forecasts", json={"series_id": "FOODS_3_090_CA_1", "horizon_days": 7}
    )
    assert created.status_code == 201
    payload = created.json()
    assert payload["forecast_id"] is not None
    assert payload["model_version"] == "stub_v1"
    assert payload["origin_day"] == 60
    assert payload["origin_date"] == "2011-03-29"
    assert payload["beyond_evaluated_horizons"] is False
    assert len(payload["points"]) == 7
    # Stub model predicts exactly 3.0 units per day.
    assert all(point["predicted_units"] == 3.0 for point in payload["points"])
    assert any("price" in note.lower() for note in payload["assumptions"])

    readback = api_client.get(f"/api/v1/forecasts/{payload['forecast_id']}")
    assert readback.status_code == 200
    assert readback.json()["points"] == payload["points"]

    listing = api_client.get("/api/v1/forecasts").json()
    assert listing["total"] == 1
    assert listing["items"][0]["forecast_id"] == payload["forecast_id"]


def test_forecast_validation_errors(api_client: TestClient) -> None:
    unknown = api_client.post(
        "/api/v1/forecasts", json={"series_id": "NOPE", "horizon_days": 7}
    )
    assert unknown.status_code == 404

    zero = api_client.post(
        "/api/v1/forecasts", json={"series_id": "FOODS_3_090_CA_1", "horizon_days": 0}
    )
    assert zero.status_code == 422

    too_far = api_client.post(
        "/api/v1/forecasts", json={"series_id": "FOODS_3_090_CA_1", "horizon_days": 60}
    )
    assert too_far.status_code == 422

    missing_run = api_client.get("/api/v1/forecasts/deadbeef")
    assert missing_run.status_code == 404


def test_forecast_without_persistence(api_client: TestClient) -> None:
    created = api_client.post(
        "/api/v1/forecasts",
        json={"series_id": "FOODS_3_090_CA_1", "horizon_days": 3, "persist": False},
    )
    assert created.status_code == 201
    assert created.json()["forecast_id"] is None
    assert api_client.get("/api/v1/forecasts").json()["total"] == 0


def test_pricing_scenario_and_list(api_client: TestClient) -> None:
    body = {
        "series_id": "FOODS_3_090_CA_1",
        "current_price": 2.5,
        "candidate_price": 2.25,
        "unit_cost": 1.2,
        "baseline_units": 40,
        "demand_model": "elasticity",
        "elasticity": -1.5,
        "horizon_days": 28,
        "constraints": {"min_price": 1.0, "max_price": 5.0},
    }
    created = api_client.post("/api/v1/pricing/scenarios", json=body)
    assert created.status_code == 201
    payload = created.json()
    # units ratio = (2.25/2.5)^-1.5 = 1.171; revenue ratio = 1.171*0.9 ~ 1.054
    assert payload["outcome"]["revenue_change_percent"] == 5.4093
    assert payload["scenario_id"] is not None
    assert (
        "assumption" in payload["honesty_note"].lower()
        or "observational" in payload["honesty_note"].lower()
    )

    listing = api_client.get("/api/v1/pricing/scenarios").json()
    assert listing["total"] == 1
    assert listing["items"][0]["scenario_id"] == payload["scenario_id"]

    bad = api_client.post(
        "/api/v1/pricing/scenarios",
        json={**body, "candidate_price": -1.0},
    )
    assert bad.status_code == 422


def test_pricing_optimize(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/pricing/optimize",
        json={
            "current_price": 2.5,
            "unit_cost": 1.2,
            "baseline_units": 40,
            "demand_model": "elasticity",
            "elasticity": -1.5,
            "constraints": {"min_price": 1.0, "max_price": 5.0},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 1.0 <= payload["optimal_price"] <= 5.0
    assert payload["objective"].startswith("maximize expected gross profit")
    assert any("causal" in note for note in payload["assumptions"])


def test_pricing_optimize_infeasible(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/pricing/optimize",
        json={
            "current_price": 2.5,
            "unit_cost": 1.2,
            "baseline_units": 40,
            "demand_model": "elasticity",
            "elasticity": -1.5,
            "constraints": {
                "min_price": 1.0,
                "max_price": 5.0,
                "min_margin_percent": 99,
            },
        },
    )
    assert response.status_code == 422


def test_pricing_sensitivity_sweep(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/pricing/sensitivity",
        json={
            "baseline_units": 40,
            "current_price": 2.5,
            "candidate_price": 2.0,
            "unit_cost": 1.2,
            "constraints": {"min_price": 1.0, "max_price": 5.0},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["elasticity_sweep"]) == 5
    profits = [row["candidate_gross_profit"] for row in payload["elasticity_sweep"]]
    assert profits != sorted(profits, reverse=True) or len(set(profits)) > 1
    assert "assumed demand response" in payload["note"]


def test_inventory_explicit_and_series_derived(api_client: TestClient) -> None:
    explicit = api_client.post(
        "/api/v1/inventory/recommendations",
        json={
            "daily_demand_mean": 10.0,
            "daily_demand_std": 1.0,
            "horizon_days": 14,
            "inputs": {
                "service_level": 0.95,
                "lead_time_days": 7,
                "review_period_days": 7,
                "on_hand_units": 100,
            },
        },
    )
    assert explicit.status_code == 200
    payload = explicit.json()
    assert payload["demand_source"] == "explicit"
    assert payload["recommendation_id"] is not None
    # ROP = 70 + 1.6449*sqrt(7) ~ 74.33 < 100 position -> ok.
    assert payload["recommendation"]["status"] == "ok"
    assert any("supplied explicitly" in w for w in payload["warnings"])

    derived = api_client.post(
        "/api/v1/inventory/recommendations",
        json={"series_id": "FOODS_3_090_CA_1", "horizon_days": 7},
    )
    assert derived.status_code == 200
    assert derived.json()["demand_source"] == "series_forecast"
    # Stub forecast is a flat 3 units/day.
    assert derived.json()["recommendation"]["daily_demand_mean"] == 3.0


def test_inventory_validation_errors(api_client: TestClient) -> None:
    no_source = api_client.post("/api/v1/inventory/recommendations", json={})
    assert no_source.status_code == 422

    bad_inputs = api_client.post(
        "/api/v1/inventory/recommendations",
        json={
            "daily_demand_mean": 5,
            "daily_demand_std": 1,
            "inputs": {"service_level": 1.2},
        },
    )
    assert bad_inputs.status_code == 422

    unknown = api_client.post(
        "/api/v1/inventory/recommendations", json={"series_id": "NOPE"}
    )
    assert unknown.status_code == 404


def test_models_endpoints(api_client: TestClient) -> None:
    listing = api_client.get("/api/v1/models").json()
    assert listing["total"] == 1
    assert listing["items"][0]["version"] == "stub_v1"

    detail = api_client.get("/api/v1/models/stub_v1")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["metadata"]["algorithm"] == "stub"
    assert payload["evaluation"]["model"]["overall"]["wape_percent"] == 50.0
    assert "bundle.joblib" in payload["artifacts"]

    missing = api_client.get("/api/v1/models/nope")
    assert missing.status_code == 404


def test_monitoring_drift_and_evaluation(api_client: TestClient) -> None:
    drift = api_client.get("/api/v1/monitoring/drift")
    assert drift.status_code == 200
    payload = drift.json()
    assert payload["demand"]["series_compared"] == 120
    assert payload["demand"]["classification"] in {"stable", "moderate", "significant"}
    assert payload["price"]["series_compared"] == 120
    assert payload["staleness"]["artifact"]["version"] == "stub_v1"

    replay = api_client.get("/api/v1/monitoring/evaluation/stub_v1")
    assert replay.status_code == 200
    assert "replay_note" in replay.json()

    missing = api_client.get("/api/v1/monitoring/evaluation/nope")
    assert missing.status_code == 404


def test_monitoring_score_endpoint(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/monitoring/score",
        json={
            "rows": [
                {"actual_units": 5, "predicted_units": 6, "horizon": 1},
                {
                    "actual_units": 3,
                    "predicted_units": 2,
                    "horizon": 2,
                    "cat_id": "FOODS",
                },
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    # |5-6| + |3-2| = 2 over |5| + |3| = 8 -> WAPE 25%.
    assert payload["overall"]["wape_percent"] == 25.0
    assert payload["by_horizon"]["1"]["mae_units"] == 1.0
    assert payload["by_category"]["FOODS"]["rows"] == 1

    empty = api_client.post("/api/v1/monitoring/score", json={"rows": []})
    assert empty.status_code == 422
