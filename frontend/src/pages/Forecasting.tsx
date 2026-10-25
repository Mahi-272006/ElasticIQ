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
import { Icon } from "../components/Icon";
import {
  CHART_COLORS,
  fmtDateFull,
  fmtDateShort,
  gridProps,
  tooltipProps,
  xAxisProps,
  yAxisProps,
} from "../components/charts";
import {
  ErrorBox,
  Note,
  PageHead,
  Panel,
  Spinner,
  num,
  useAsync,
} from "../components/ui";

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
        title="Forecasting"
        subtitle="Recursive multi-horizon forecasts from the trained HGB Poisson model. Predictions feed back into the feature history; future prices are assumed to equal the last observed price."
      />

      <div className="toolbar">
        <label className="field" style={{ minWidth: 250 }}>
          Search series (id or item)
          <span className="input-with-icon">
            <span className="input-icon">
              <Icon name="search" />
            </span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="FOODS_3_090_CA_1"
            />
          </span>
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
        <Panel
          title={seriesId}
          sub={
            result
              ? `Last 28 observed days · ${horizon}-day forecast from ${result.origin_date}`
              : "Last 28 observed days · run a forecast to project demand"
          }
          tag={result ? "forecast" : "observed"}
        >
          <div
            className="chart-box"
            role="img"
            aria-label={`Observed and predicted daily units for ${seriesId}`}
          >
            <ResponsiveContainer>
              <LineChart data={chart} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis
                  dataKey="date"
                  {...xAxisProps}
                  minTickGap={42}
                  tickFormatter={(v: string) => fmtDateShort(v)}
                />
                <YAxis {...yAxisProps} />
                <Tooltip
                  {...tooltipProps}
                  labelFormatter={(label) => fmtDateFull(String(label))}
                  formatter={(value: unknown, name: unknown) => [
                    num(Number(value), 2),
                    String(name),
                  ]}
                />
                <Legend />
                {result ? (
                  <ReferenceLine
                    x={result.origin_date}
                    stroke={CHART_COLORS.amber}
                    strokeDasharray="4 3"
                    label={{
                      value: "forecast origin",
                      fill: CHART_COLORS.amber,
                      fontSize: 11,
                      position: "insideTopRight",
                    }}
                  />
                ) : null}
                <Line
                  type="monotone"
                  dataKey="history"
                  stroke={CHART_COLORS.slate}
                  dot={false}
                  strokeWidth={1.5}
                  name="Observed units"
                  connectNulls={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke={CHART_COLORS.primary}
                  dot={false}
                  strokeWidth={2}
                  name="Predicted units"
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      ) : null}

      {result ? (
        <div className="grid cols-2 mt-3">
          <Panel title="Run details">
            <table className="kv">
              <tbody>
                <tr>
                  <td>Forecast id</td>
                  <td className="mono wrap">{result.forecast_id ?? "(not persisted)"}</td>
                </tr>
                <tr>
                  <td>Model version</td>
                  <td className="mono">{result.model_version}</td>
                </tr>
                <tr>
                  <td>Origin</td>
                  <td>
                    {result.origin_date} (d_{result.origin_day})
                  </td>
                </tr>
                <tr>
                  <td>Predicted total</td>
                  <td className="tnum">
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
          </Panel>
          <Panel title="Assumptions" tag="always shown">
            <ul className="assumptions">
              {result.assumptions.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          </Panel>
        </div>
      ) : null}

      <Panel className="mt-3" title="Recent stored runs">
        {runs.loading ? <Spinner /> : null}
        {runs.error ? <ErrorBox message={runs.error} /> : null}
        {runs.data && runs.data.items.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run id</th>
                  <th>Series</th>
                  <th className="num">Horizon</th>
                  <th>Model</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.items.map((run) => (
                  <tr key={run.forecast_id}>
                    <td className="mono">{run.forecast_id.slice(0, 10)}…</td>
                    <td className="mono">{run.series_id}</td>
                    <td className="num">{run.horizon_days}d</td>
                    <td className="mono">{run.model_version}</td>
                    <td className="nowrap">{run.created_at.replace("T", " ").slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">No runs yet — run a forecast above.</div>
        )}
      </Panel>

      <Note honesty>
        Accuracy figures on the Models page are holdout measurements, not a guarantee
        for these future days. WAPE is high on this dataset because most series are
        intermittent — treat single-day predictions as a planning input, not a promise.
      </Note>
    </>
  );
}
