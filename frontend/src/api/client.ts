/** Minimal typed fetch client for the ElasticIQ API (proxied by Vite). */

import type {
  DriftResponse,
  ForecastResponse,
  ForecastRunsPage,
  HealthResponse,
  InventoryResponse,
  ModelDetail,
  OptimizeResult,
  OverviewResponse,
  PricingScenarioResponse,
  ProductDetail,
  ProductsPage,
  ScoreResult,
  StoresResponse,
  TrendsResponse,
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body?.detail)) {
        // Pydantic validation errors: summarize the first issue.
        detail = body.detail
          .slice(0, 3)
          .map((issue: any) => `${issue.loc?.join(".")}: ${issue.msg}`)
          .join("; ");
      }
    } catch {
      /* keep status text */
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

function qs(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  overview: () => request<OverviewResponse>("/api/v1/overview"),

  products: (params: {
    limit?: number;
    offset?: number;
    cat_id?: string;
    store_id?: string;
    search?: string;
  }) => request<ProductsPage>(`/api/v1/products${qs(params)}`),
  product: (seriesId: string) =>
    request<ProductDetail>(`/api/v1/products/${encodeURIComponent(seriesId)}`),
  stores: () => request<StoresResponse>("/api/v1/stores"),

  trends: (params: { days?: number; store_id?: string; cat_id?: string }) =>
    request<TrendsResponse>(`/api/v1/sales/trends${qs(params)}`),

  createForecast: (body: { series_id: string; horizon_days: number }) =>
    request<ForecastResponse>("/api/v1/forecasts", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  forecast: (id: string) => request<ForecastResponse>(`/api/v1/forecasts/${id}`),
  forecasts: (params: { limit?: number; offset?: number; series_id?: string }) =>
    request<ForecastRunsPage>(`/api/v1/forecasts${qs(params)}`),

  pricingScenario: (body: Record<string, unknown>) =>
    request<PricingScenarioResponse>("/api/v1/pricing/scenarios", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  pricingOptimize: (body: Record<string, unknown>) =>
    request<OptimizeResult>("/api/v1/pricing/optimize", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  pricingSensitivity: (body: Record<string, unknown>) =>
    request<{ elasticity_sweep: any[]; note: string }>("/api/v1/pricing/sensitivity", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  inventory: (body: Record<string, unknown>) =>
    request<InventoryResponse>("/api/v1/inventory/recommendations", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  models: () => request<{ total: number; items: any[] }>("/api/v1/models"),
  model: (version: string) => request<ModelDetail>(`/api/v1/models/${version}`),

  drift: () => request<DriftResponse>("/api/v1/monitoring/drift"),
  evaluation: (version: string) =>
    request<Record<string, any>>(`/api/v1/monitoring/evaluation/${version}`),
  score: (rows: Array<Record<string, unknown>>) =>
    request<ScoreResult>("/api/v1/monitoring/score", {
      method: "POST",
      body: JSON.stringify({ rows }),
    }),
};
