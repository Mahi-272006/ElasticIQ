import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { ErrorBox, Kpi, Note, PageHead, Spinner, num, useAsync } from "../components/ui";
import { effectiveVersion } from "../lib/version";

interface HorizonRow {
  horizon: string;
  model: number;
  mean_baseline: number;
  last_day_naive: number;
  weekly_seasonal_naive: number;
}

export default function Models() {
  const [selected, setSelected] = useState<string | null>(null);
  const versions = useAsync(() => api.models(), []);
  // Single source of truth for "which version is active": the detail fetch
  // and the <select> value must derive from the same value, otherwise the
  // dropdown can show a version whose detail was never requested and the
  // page falsely reports "No trained model found".
  const active = effectiveVersion(selected, versions.data?.items);
  const detail = useAsync(
    () => (active ? api.model(active) : Promise.resolve(null)),
    [active],
  );

  const evaluation = detail.data?.evaluation;
  const modelEval = evaluation?.model;
  const baselines = evaluation?.baselines ?? {};

  const horizonRows: HorizonRow[] = modelEval?.by_horizon
    ? Object.keys(modelEval.by_horizon)
        .sort((a, b) => Number(a) - Number(b))
        .map((h) => ({
          horizon: `h${h}`,
          model: modelEval.by_horizon[h].wape_percent,
          mean_baseline: baselines.mean_baseline?.by_horizon?.[h]?.wape_percent ?? NaN,
          last_day_naive: baselines.last_day_naive?.by_horizon?.[h]?.wape_percent ?? NaN,
          weekly_seasonal_naive:
            baselines.weekly_seasonal_naive?.by_horizon?.[h]?.wape_percent ?? NaN,
        }))
    : [];

  const categoryRows = modelEval?.by_category
    ? Object.entries(modelEval.by_category).map(([name, metrics]: [string, any]) => ({
        category: name,
        model: metrics.wape_percent,
      }))
    : [];

  return (
    <>
      <PageHead
        title="Model performance"
        subtitle="Holdout evaluation on the chronological validation window (d_1886–d_1913). Every model and baseline is scored on identical series, dates, and the recursive 28-day protocol."
      />

      <div className="controls">
        <label className="field">
          Model version
          <select
            value={active ?? ""}
            onChange={(e) => setSelected(e.target.value)}
          >
            {versions.data?.items.map((item: any) => (
              <option key={item.version} value={item.version}>
                {item.version}
              </option>
            ))}
          </select>
        </label>
      </div>

      {versions.loading ? <Spinner /> : null}
      {versions.error ? <ErrorBox message={versions.error} /> : null}
      {detail.loading ? <Spinner label="Loading evaluation…" /> : null}
      {detail.error ? <ErrorBox message={detail.error} /> : null}

      {modelEval ? (
        <>
          <div className="grid cols-4">
            <Kpi label="h7 WAPE" value={`${num(modelEval.by_horizon["7"]?.wape_percent, 2)}%`} sub="7-day horizon" />
            <Kpi label="h14 WAPE" value={`${num(modelEval.by_horizon["14"]?.wape_percent, 2)}%`} sub="14-day horizon" />
            <Kpi label="h28 WAPE" value={`${num(modelEval.by_horizon["28"]?.wape_percent, 2)}%`} sub="28-day horizon" />
            <Kpi
              label="Mean baseline h28"
              value={`${num(baselines.mean_baseline?.by_horizon?.["28"]?.wape_percent, 2)}%`}
              sub="the benchmark to beat"
            />
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <h3>WAPE by horizon — model vs baselines</h3>
            <div className="chart-box">
              <ResponsiveContainer>
                <BarChart data={horizonRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                  <XAxis dataKey="horizon" stroke="#8b98ad" fontSize={12} />
                  <YAxis stroke="#8b98ad" fontSize={11} unit="%" />
                  <Tooltip
                    contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                    formatter={(value: any, name: any) => [`${num(value, 2)}%`, name]}
                  />
                  <Legend />
                  <Bar dataKey="model" fill="#4f8cff" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="mean_baseline" fill="#d9a441" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="last_day_naive" fill="#7a5cff" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="weekly_seasonal_naive" fill="#e5534b" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <Note honesty>
              The gradient-boosting model beats both naive baselines at every horizon but
              still loses to the plain mean baseline. With intermittent retail demand, a
              constant mean is a hard benchmark — this is reported as measured, not
              dressed up.
            </Note>
          </div>

          {categoryRows.length > 0 ? (
            <div className="card" style={{ marginTop: 14 }}>
              <h3>Model WAPE by category (h28)</h3>
              <div className="chart-box short">
                <ResponsiveContainer>
                  <BarChart data={categoryRows}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                    <XAxis dataKey="category" stroke="#8b98ad" fontSize={12} />
                    <YAxis stroke="#8b98ad" fontSize={11} unit="%" />
                    <Tooltip
                      contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                      formatter={(value: any) => [`${num(value, 2)}%`, "WAPE"]}
                    />
                    <Bar dataKey="model" fill="#3fb97f" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : null}
        </>
      ) : !detail.loading && !versions.loading ? (
        <div className="card">
          <div className="muted">
            No trained model found. Run <code>python -m src.forecasting.train</code> to
            produce artifacts.
          </div>
        </div>
      ) : null}

      {detail.data ? (
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Version metadata — {detail.data.version}</h3>
          <table>
            <tbody>
              {Object.entries(detail.data.metadata ?? {}).map(([key, value]) => (
                <tr key={key}>
                  <td>{key.replaceAll("_", " ")}</td>
                  <td>
                    {typeof value === "object" ? JSON.stringify(value) : String(value)}
                  </td>
                </tr>
              ))}
              <tr>
                <td>artifacts</td>
                <td>{detail.data.artifacts.join(", ")}</td>
              </tr>
            </tbody>
          </table>
        </div>
      ) : null}
    </>
  );
}
