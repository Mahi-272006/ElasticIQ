import { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { ForecastResponse } from "../api/types";
import { ErrorBox, Note, PageHead, Spinner, num, useAsync } from "../components/ui";

interface ChartRow {
  date: string;
  history: number | null;
  forecast: number | null;
}

export default function Forecasting() {
  const [query, setQuery] = useState("FOODS_3_090_CA_1_validation");
  const [seriesId, setSeriesId] = useState("FOODS_3_090_CA_1_validation");
  const [horizon, setHorizon] = useState(28);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const search = useAsync(
    () => api.products({ search: query || undefined, limit: 8 }),
    [query],
  );
  const detail = useAsync(() => api.product(seriesId), [seriesId]);
  const runs = useAsync(() => api.forecasts({ limit: 10 }), []);

  async function run() {
    setRunning(true);
    setRunError(null);
    try {
      setResult(await api.createForecast({ series_id: seriesId, horizon_days: horizon }));
      runs.refresh();
    } catch (err) {
      setRunError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  const chart: ChartRow[] = [];
  if (detail.data) {
    const recent = detail.data.history.slice(-28);
    for (const point of recent) {
      chart.push({ date: point.date, history: point.units, forecast: null });
    }
  }
  if (result) {
    const originIndex = chart.length - 1;
    for (const point of result.points) {
      if (originIndex >= 0 && chart[originIndex]) {
        chart[originIndex].forecast = chart[originIndex].history; // bridge day
      }
      chart.push({ date: point.date, history: null, forecast: point.predicted_units });
    }
  }

  return (
    <>
      <PageHead
        title="Demand forecasting"
        subtitle="Recursive multi-horizon forecasts from the trained HGB Poisson model. Predictions feed back into the feature history; future prices are assumed to equal the last observed price."
      />

      <div className="controls">
        <label className="field" style={{ minWidth: 260 }}>
          Search series (id or item)
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="FOODS_3_090_CA_1"
          />
        </label>
        <label className="field">
          Series
          <select value={seriesId} onChange={(e) => setSeriesId(e.target.value)}>
            {!search.data?.items.some((p) => p.series_id === seriesId) ? (
              <option value={seriesId}>{seriesId}</option>
            ) : null}
            {search.data?.items.map((p) => (
              <option key={p.series_id} value={p.series_id}>
                {p.series_id}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Horizon
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
            {[7, 14, 28, 56].map((h) => (
              <option key={h} value={h}>
                {h} days
              </option>
            ))}
          </select>
        </label>
        <button onClick={run} disabled={running}>
          {running ? "Forecasting…" : "Run forecast"}
        </button>
      </div>

      {runError ? <ErrorBox message={runError} /> : null}
      {detail.loading ? <Spinner label="Loading history…" /> : null}

      {chart.length > 0 ? (
        <div className="card">
          <h3>
            {seriesId} — last 28 observed days{result ? ` + ${horizon}-day forecast` : ""}
          </h3>
          <div className="chart-box">
            <ResponsiveContainer>
              <LineChart data={chart}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                <XAxis
                  dataKey="date"
                  stroke="#8b98ad"
                  fontSize={11}
                  minTickGap={42}
                  tickFormatter={(v: string) => v.slice(5)}
                />
                <YAxis stroke="#8b98ad" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                  formatter={(value: any, name: any) => [num(value, 2), name]}
                />
                <Legend />
                {result ? (
                  <ReferenceLine
                    x={result.origin_date}
                    stroke="#d9a441"
                    label={{ value: "forecast origin", fill: "#d9a441", fontSize: 11 }}
                  />
                ) : null}
                <Line
                  type="monotone"
                  dataKey="history"
                  stroke="#8b98ad"
                  dot={false}
                  strokeWidth={1.5}
                  name="Observed units"
                  connectNulls={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#4f8cff"
                  dot={false}
                  strokeWidth={2}
                  name="Predicted units"
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : null}

      {result ? (
        <div className="grid cols-2" style={{ marginTop: 14 }}>
          <div className="card">
            <h3>Run details</h3>
            <table>
              <tbody>
                <tr>
                  <td>Forecast id</td>
                  <td>{result.forecast_id ?? "(not persisted)"}</td>
                </tr>
                <tr>
                  <td>Model version</td>
                  <td>{result.model_version}</td>
                </tr>
                <tr>
                  <td>Origin</td>
                  <td>
                    {result.origin_date} (d_{result.origin_day})
                  </td>
                </tr>
                <tr>
                  <td>Predicted total</td>
                  <td>
                    {num(
                      result.points.reduce((sum, p) => sum + p.predicted_units, 0),
                      1,
                    )}{" "}
                    units
                  </td>
                </tr>
                <tr>
                  <td>Evaluated horizons</td>
                  <td>
                    {result.beyond_evaluated_horizons ? (
                      <span className="badge warn">beyond evaluated 7/14/28</span>
                    ) : (
                      <span className="badge good">evaluated horizon</span>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="card">
            <h3>Assumptions (always shown)</h3>
            <ul className="assumptions">
              {result.assumptions.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Recent stored runs</h3>
        {runs.loading ? <Spinner /> : null}
        {runs.error ? <ErrorBox message={runs.error} /> : null}
        {runs.data && runs.data.items.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Run id</th>
                <th>Series</th>
                <th>Horizon</th>
                <th>Model</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.items.map((run) => (
                <tr key={run.forecast_id}>
                  <td>{run.forecast_id.slice(0, 10)}…</td>
                  <td>{run.series_id}</td>
                  <td>{run.horizon_days}d</td>
                  <td>{run.model_version}</td>
                  <td>{run.created_at.replace("T", " ").slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="muted small">No runs yet — run a forecast above.</div>
        )}
      </div>

      <Note honesty>
        Forecast accuracy figures on the Models page are holdout measurements, not a
        guarantee for these future days. The M5 holdout WAPE is high (~89%) because most
        series are intermittent — plan accordingly.
      </Note>
    </>
  );
}
