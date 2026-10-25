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
import {
  CHART_COLORS,
  gridProps,
  tooltipProps,
  xAxisProps,
  yAxisProps,
} from "../components/charts";
import {
  EmptyState,
  ErrorBox,
  Metric,
  Note,
  PageHead,
  Panel,
  Spinner,
  pct,
  useAsync,
} from "../components/ui";
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

  const modelH28 = modelEval?.by_horizon?.["28"]?.wape_percent;
  const meanH28 = baselines.mean_baseline?.by_horizon?.["28"]?.wape_percent;
  const beatsMean =
    typeof modelH28 === "number" && typeof meanH28 === "number" && modelH28 < meanH28;
  const comparable = horizonRows.filter(
    (r) =>
      Number.isFinite(r.model) &&
      Number.isFinite(r.last_day_naive) &&
      Number.isFinite(r.weekly_seasonal_naive),
  );
  const beatsNaive =
    comparable.length > 0 &&
    comparable.every(
      (r) => r.model < r.last_day_naive && r.model < r.weekly_seasonal_naive,
    );

  return (
    <>
      <PageHead
        title="Model performance"
        subtitle="Holdout evaluation on the chronological validation window (d_1886–d_1913). Every model and baseline is scored on identical series, dates, and the recursive 28-day protocol."
      />

      <div className="toolbar">
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
        {detail.data?.version ? (
          <span className="tag" style={{ marginBottom: 7 }}>
            evaluating {detail.data.version}
          </span>
        ) : null}
      </div>

      {versions.loading ? <Spinner /> : null}
      {versions.error ? <ErrorBox message={versions.error} /> : null}
      {detail.loading ? <Spinner label="Loading evaluation…" /> : null}
      {detail.error ? <ErrorBox message={detail.error} /> : null}

      {modelEval ? (
        <>
          <div className="metrics">
            <Metric label="h7 WAPE" value={pct(modelEval.by_horizon["7"]?.wape_percent)} sub="7-day horizon" />
            <Metric label="h14 WAPE" value={pct(modelEval.by_horizon["14"]?.wape_percent)} sub="14-day horizon" />
            <Metric label="h28 WAPE" value={pct(modelEval.by_horizon["28"]?.wape_percent)} sub="28-day horizon" />
            <Metric
              label="Mean baseline h28"
              value={pct(meanH28)}
              sub="the benchmark to beat"
            />
          </div>

          <Panel
            className="mt-3"
            title="WAPE by horizon — model vs baselines"
            sub="lower is better"
          >
            <div
              className="chart-box"
              role="img"
              aria-label="Grouped bar chart of WAPE by horizon for the model and baselines"
            >
              <ResponsiveContainer>
                <BarChart data={horizonRows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="horizon" {...xAxisProps} />
                  <YAxis {...yAxisProps} unit="%" />
                  <Tooltip
                    {...tooltipProps}
                    formatter={(value: unknown, name: unknown) => [
                      pct(Number(value)),
                      String(name).replaceAll("_", " "),
                    ]}
                  />
                  <Legend />
                  <Bar dataKey="model" name="model" fill={CHART_COLORS.primary} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  <Bar
                    dataKey="mean_baseline"
                    name="mean baseline"
                    fill={CHART_COLORS.slate}
                    radius={[3, 3, 0, 0]}
                    isAnimationActive={false}
                  />
                  <Bar
                    dataKey="last_day_naive"
                    name="last day naive"
                    fill={CHART_COLORS.amber}
                    radius={[3, 3, 0, 0]}
                    isAnimationActive={false}
                  />
                  <Bar
                    dataKey="weekly_seasonal_naive"
                    name="weekly seasonal naive"
                    fill={CHART_COLORS.grey}
                    radius={[3, 3, 0, 0]}
                    isAnimationActive={false}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="note honesty">
              Reported as measured: the gradient-boosting model{" "}
              <strong>{beatsNaive ? "beats" : "does not beat"}</strong> both naive
              baselines at every evaluated horizon, and{" "}
              <strong>{beatsMean ? "beats" : "does not beat"}</strong> the plain mean
              baseline
              {typeof modelH28 === "number" && typeof meanH28 === "number"
                ? ` (${pct(modelH28)} vs ${pct(meanH28)} at h28)`
                : ""}
              . With intermittent retail demand, a constant mean is a hard benchmark —
              all numbers come from the same chronological window.
            </div>
          </Panel>

          {categoryRows.length > 0 ? (
            <Panel className="mt-3" title="Model WAPE by category" sub="h28 horizon">
              <div
                className="chart-box short"
                role="img"
                aria-label="Bar chart of model WAPE by product category"
              >
                <ResponsiveContainer>
                  <BarChart data={categoryRows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="category" {...xAxisProps} />
                    <YAxis {...yAxisProps} unit="%" />
                    <Tooltip
                      {...tooltipProps}
                      formatter={(value: unknown) => [pct(Number(value)), "WAPE"]}
                    />
                    <Bar dataKey="model" name="model" fill={CHART_COLORS.primary} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          ) : null}
        </>
      ) : !detail.loading && !versions.loading ? (
        <Panel>
          <EmptyState>
            No trained model found. Run <code>python -m src.forecasting.train</code> to
            produce artifacts.
          </EmptyState>
        </Panel>
      ) : null}

      {detail.data ? (
        <Panel className="mt-3" title={`Version metadata — ${detail.data.version}`}>
          <div className="table-wrap">
            <table className="kv">
              <tbody>
                {Object.entries(detail.data.metadata ?? {}).map(([key, value]) => (
                  <tr key={key}>
                    <td>{key.replaceAll("_", " ")}</td>
                    <td className="mono wrap">
                      {typeof value === "object" ? JSON.stringify(value) : String(value)}
                    </td>
                  </tr>
                ))}
                <tr>
                  <td>artifacts</td>
                  <td className="mono wrap">{detail.data.artifacts.join(", ")}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      <Note>
        WAPE = Σ|actual − predicted| / Σ actual, in percent, over the holdout rows.
        Lower is better. Baselines use the same rows, dates, and recursive protocol
        as the model.
      </Note>
    </>
  );
}
