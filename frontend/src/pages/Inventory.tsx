import { useState } from "react";
import { api } from "../api/client";
import type { InventoryResponse } from "../api/types";
import {
  EmptyState,
  ErrorBox,
  Metric,
  Note,
  PageHead,
  Panel,
  Spinner,
  StatusBadge,
  num,
} from "../components/ui";

export default function Inventory() {
  const [seriesId, setSeriesId] = useState("FOODS_3_090_CA_1_validation");
  const [useSeries, setUseSeries] = useState(true);
  const [mean, setMean] = useState("10");
  const [std, setStd] = useState("2");
  const [horizon, setHorizon] = useState(28);
  const [serviceLevel, setServiceLevel] = useState(0.95);
  const [leadTime, setLeadTime] = useState(7);
  const [reviewPeriod, setReviewPeriod] = useState(7);
  const [onHand, setOnHand] = useState("0");
  const [openOrders, setOpenOrders] = useState("0");
  const [result, setResult] = useState<InventoryResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, unknown> = {
        horizon_days: horizon,
        inputs: {
          service_level: serviceLevel,
          lead_time_days: leadTime,
          review_period_days: reviewPeriod,
          on_hand_units: Number(onHand),
          open_order_units: Number(openOrders),
        },
      };
      if (useSeries) {
        body.series_id = seriesId;
      } else {
        body.daily_demand_mean = Number(mean);
        body.daily_demand_std = Number(std);
      }
      setResult(await api.inventory(body));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const rec = result?.recommendation;

  return (
    <>
      <PageHead
        title="Inventory"
        subtitle="Reorder points, safety stock, and order-up-to levels from forecast demand. The M5 dataset has no stock or lead-time data — every inventory parameter below is yours to supply."
      />

      <div className="grid cols-2">
        <Panel title="Inputs" tag="your inputs">
          <div className="form-legend">Demand</div>
          <div className="form-grid">
            <label className="field">
              Source
              <select
                value={useSeries ? "series" : "explicit"}
                onChange={(e) => setUseSeries(e.target.value === "series")}
              >
                <option value="series">series forecast</option>
                <option value="explicit">explicit statistics</option>
              </select>
            </label>
            {useSeries ? (
              <label className="field" style={{ gridColumn: "span 2" }}>
                Series id
                <input value={seriesId} onChange={(e) => setSeriesId(e.target.value)} />
              </label>
            ) : (
              <>
                <label className="field">
                  Daily mean
                  <input value={mean} onChange={(e) => setMean(e.target.value)} />
                </label>
                <label className="field">
                  Daily std
                  <input value={std} onChange={(e) => setStd(e.target.value)} />
                </label>
              </>
            )}
            <label className="field">
              Horizon (days)
              <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
                {[7, 14, 28].map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="form-legend">Service &amp; lead time</div>
          <div className="form-grid">
            <label className="field">
              Service level
              <select
                value={serviceLevel}
                onChange={(e) => setServiceLevel(Number(e.target.value))}
              >
                {[0.9, 0.95, 0.975, 0.99].map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Lead time (days)
              <input
                type="number"
                min={0}
                value={leadTime}
                onChange={(e) => setLeadTime(Number(e.target.value))}
              />
            </label>
            <label className="field">
              Review period (days)
              <input
                type="number"
                min={0}
                value={reviewPeriod}
                onChange={(e) => setReviewPeriod(Number(e.target.value))}
              />
            </label>
          </div>

          <div className="form-legend">Inventory position</div>
          <div className="form-grid">
            <label className="field">
              On hand
              <input value={onHand} onChange={(e) => setOnHand(e.target.value)} />
            </label>
            <label className="field">
              Open orders
              <input value={openOrders} onChange={(e) => setOpenOrders(e.target.value)} />
            </label>
          </div>

          <div className="mt-3">
            <button onClick={run} disabled={busy}>
              {busy ? "Computing…" : "Get recommendation"}
            </button>
          </div>
          {busy ? <Spinner /> : null}
          {error ? <ErrorBox message={error} /> : null}
        </Panel>

        <Panel title="Recommendation" tag="modeled">
          {rec && result ? (
            <>
              <div className="inline" style={{ marginBottom: 12 }}>
                <StatusBadge status={rec.status} />
                <span className="small muted">
                  demand from{" "}
                  {result.demand_source === "explicit"
                    ? "explicit stats"
                    : "series forecast"}
                  {result.recommendation_id
                    ? ` · saved as #${result.recommendation_id}`
                    : ""}
                </span>
              </div>
              <div className="metrics">
                <Metric
                  label="Reorder point"
                  value={num(rec.reorder_point, 1)}
                  sub={`${num(rec.lead_time_demand, 1)} lead demand + ${num(rec.safety_stock, 1)} safety`}
                />
                <Metric
                  label="Order-up-to level"
                  value={num(rec.order_up_to_level, 1)}
                  sub={`coverage ${leadTime + reviewPeriod} days`}
                />
                <Metric
                  label="Suggested replenishment"
                  value={num(rec.suggested_replenishment, 1)}
                  sub="order-up-to minus position"
                />
                <Metric
                  label="Days of cover"
                  value={
                    rec.expected_days_of_cover !== null
                      ? num(rec.expected_days_of_cover, 1)
                      : "—"
                  }
                  sub={`forecast ${num(rec.forecast_demand_horizon, 0)} units / ${horizon}d`}
                />
              </div>
              {rec.status_reasons.length > 0 ? (
                <ul className="assumptions">
                  {rec.status_reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              ) : null}
              {result.warnings.length > 0 ? (
                <ul className="assumptions">
                  {result.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              ) : null}
              <ul className="assumptions">
                {rec.assumptions.slice(0, 2).map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </>
          ) : (
            <EmptyState>
              Configure inputs and request a recommendation. Nothing here reflects a
              real warehouse — M5 contains no inventory data, so stock levels are
              always inputs you provide.
            </EmptyState>
          )}
        </Panel>
      </div>

      <Note honesty>
        Safety stock uses the normal approximation (z × σ × √lead time). Intermittent
        demand series may violate the normality assumption — treat results as a
        starting point for planner judgment, not an automatic order.
      </Note>
    </>
  );
}
