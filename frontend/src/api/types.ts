/** Typed contracts mirroring the FastAPI v1 schemas (src/api/schemas.py). */

export interface CoverageBlock {
  first_date: string;
  last_date: string;
  days: number;
  series_in_catalog: number;
  stores: number;
  categories: number;
}

export interface MetricShare {
  key: string;
  units: number;
  share_percent: number;
}

export interface ModelSummary {
  version: string;
  evaluation_horizons: number[];
  wape_percent_by_horizon: Record<string, number>;
  baseline_comparison: {
    model_overall_wape_percent?: number;
    baselines_overall_wape_percent?: Record<string, number>;
  };
  note: string;
}

export interface OverviewResponse {
  coverage: CoverageBlock;
  totals: Record<string, number>;
  categories: MetricShare[];
  stores: MetricShare[];
  model: ModelSummary | null;
  data_notes: string[];
}

export interface Product {
  series_id: string;
  item_id: string;
  dept_id: string;
  cat_id: string;
  store_id: string;
  state_id: string;
}

export interface ProductsPage {
  total: number;
  limit: number;
  offset: number;
  items: Product[];
}

export interface HistoryPoint {
  date: string;
  d: string;
  units: number;
}

export interface ProductDetail {
  product: Product;
  price_stats: Record<string, string | number | null> | null;
  history: HistoryPoint[];
  zero_demand_share: number | null;
}

export interface StoreOut {
  store_id: string;
  state_id: string | null;
  total_units: number;
  share_percent: number;
  first_date: string;
  last_date: string;
}

export interface StoresResponse {
  total: number;
  items: StoreOut[];
}

export interface TrendsPoint {
  date: string;
  d: string;
  units: number;
}

export interface TrendsResponse {
  source_aggregate: string;
  days: number;
  store_id: string | null;
  cat_id: string | null;
  total_units: number;
  points: TrendsPoint[];
}

export interface ForecastPoint {
  day_idx: number;
  d: string;
  date: string;
  predicted_units: number;
}

export interface ForecastResponse {
  forecast_id: string | null;
  series_id: string;
  horizon_days: number;
  model_version: string;
  origin_day: number;
  origin_date: string;
  beyond_evaluated_horizons: boolean;
  assumptions: string[];
  points: ForecastPoint[];
  created_at: string | null;
}

export interface ForecastRunSummary {
  forecast_id: string;
  series_id: string;
  horizon_days: number;
  model_version: string;
  origin_day: number;
  origin_date: string;
  created_at: string;
}

export interface ForecastRunsPage {
  total: number;
  limit: number;
  offset: number;
  items: ForecastRunSummary[];
}

export interface ScenarioOutcome {
  current_revenue: number;
  candidate_revenue: number;
  current_gross_profit: number;
  candidate_gross_profit: number;
  expected_units_current: number;
  expected_units_candidate: number;
  revenue_change_percent: number;
  profit_change_percent: number;
  capped_units: number | null;
  warnings: string[];
}

export interface PricingScenarioResponse {
  scenario_id: number | null;
  series_id: string | null;
  request: Record<string, unknown>;
  outcome: ScenarioOutcome;
  honesty_note: string;
}

export interface CurvePoint {
  price: number;
  units: number;
  revenue: number;
  gross_profit: number;
}

export interface OptimizeResult {
  series_id?: string | null;
  optimal_price: number;
  optimal_units: number;
  optimal_revenue: number;
  optimal_gross_profit: number;
  current_gross_profit: number;
  improvement_percent: number;
  evaluated_prices: number;
  feasible_prices: number;
  objective: string;
  assumptions: string[];
  all_prices: CurvePoint[];
}

export interface SensitivityRow {
  elasticity: number;
  expected_units: number;
  candidate_revenue: number;
  candidate_gross_profit: number;
  profit_change_percent: number;
}

export interface InventoryRecommendation {
  horizon_days: number;
  forecast_demand_horizon: number;
  daily_demand_mean: number;
  daily_demand_std: number;
  lead_time_demand: number;
  safety_stock: number;
  reorder_point: number;
  order_up_to_level: number;
  suggested_replenishment: number;
  expected_days_of_cover: number | null;
  status: "ok" | "stockout_risk" | "overstock_risk";
  status_reasons: string[];
  assumptions: string[];
  warnings: string[];
}

export interface InventoryResponse {
  recommendation_id: number | null;
  series_id: string | null;
  demand_source: "explicit" | "series_forecast";
  recommendation: InventoryRecommendation;
  warnings: string[];
}

export interface ModelDetail {
  version: string;
  metadata: Record<string, unknown> | null;
  evaluation: Record<string, any> | null;
  artifacts: string[];
}

export interface DriftWindow {
  metric: string;
  psi: number;
  classification: "stable" | "moderate" | "significant";
  [key: string]: unknown;
}

export interface DriftResponse {
  demand: DriftWindow;
  price: DriftWindow;
  psi_thresholds: Record<string, unknown>;
  staleness: Record<string, any>;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  service: string;
  api_version: string;
  database: { connected: boolean; error?: string };
  data: { data_ready: boolean; missing_datasets?: string[]; [key: string]: unknown };
  model: { version: string | null; ready: boolean; error?: string };
}

export interface ScoreResult {
  rows: number;
  overall: {
    wape_percent: number;
    mae_units: number;
    mean_bias_units: number;
  };
  by_horizon?: Record<
    string,
    { wape_percent: number; mae_units: number; mean_bias_units: number; rows: number }
  >;
  by_category?: Record<string, { wape_percent: number; mae_units: number; rows: number }>;
  note: string;
}
