import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
  PageHead,
  Panel,
  Section,
  Spinner,
  num,
  numCompact,
  pct,
  useAsync,
} from "../components/ui";

function baselineColor(name: string): string {
  if (name.startsWith("model")) return CHART_COLORS.primary;
  if (name.includes("mean")) return CHART_COLORS.slate;
  if (name.includes("last")) return CHART_COLORS.amber;
  return CHART_COLORS.grey;
}

function StatusMetric({
  label,
  tone,
  value,
  sub,
}: {
  label: string;
  tone: "good" | "warn" | "bad" | "neutral";
  value: string;
  sub?: string;
}) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="status-value">
        <span className={`dot ${tone === "neutral" ? "" : tone}`} aria-hidden="true" />
        <span>{value}</span>
      </div>
      {sub ? <div className="metric-sub">{sub}</div> : null}
    </div>
  );
}

export default function Overview() {
  const { data, error, loading } = useAsync(() => api.overview(), []);
  const health = useAsync(() => api.health(), []);

  if (loading) return <Spinner label="Loading overview…" />;
  if (error) return <ErrorBox message={error} />;
  if (!data) return null;

  const { coverage, totals, categories, stores, model } = data;

  const baselineRows = model
    ? Object.entries(
        model.baseline_comparison.baselines_overall_wape_percent ?? {},
      ).map(([name, wape]) => ({ name: name.replaceAll("_", " "), wape }))
    : [];

  const modelWape = model?.baseline_comparison.model_overall_wape_percent;
  if (modelWape !== undefined) {
    baselineRows.unshift({ name: "model (HGB)", wape: modelWape });
  }
  const meanWape =
    model?.baseline_comparison.baselines_overall_wape_percent?.mean_baseline;
  const beatsMean =
    modelWape !== undefined && meanWape !== undefined && modelWape < meanWape;

  const status = health.data;
  const dbTone =
    status?.database.connected === undefined
      ? "neutral"
      : status.database.connected
        ? "good"
        : "bad";

  return (
    <>
      <PageHead
        title="Overview"
        subtitle="Historical demand coverage from the M5 catalog, holdout evaluation of the deployed forecasting model, and current service status."
      />

      <Section title="Catalog" hint={`${coverage.first_date} → ${coverage.last_date}`}>
        <div className="metrics">
          <Metric
            label="Total units"
            value={num(totals.total_units, 0)}
            sub={`${num(coverage.days, 0)} days of history`}
          />
          <Metric
            label="Mean daily units"
            value={num(totals.mean_daily_units, 0)}
            sub="across the catalog"
          />
          <Metric
            label="Series in catalog"
            value={num(coverage.series_in_catalog, 0)}
            sub="one item in one store"
          />
          <Metric
            label="Stores · categories"
            value={`${coverage.stores} · ${coverage.categories}`}
            sub="M5 dimensions"
          />
        </div>

        <div className="grid cols-2 mt-3">
          <Panel title="Units by category">
            <div className="chart-box short" role="img" aria-label="Bar chart of total units by category">
              <ResponsiveContainer>
                <BarChart data={categories} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="key" {...xAxisProps} />
                  <YAxis {...yAxisProps} tickFormatter={numCompact} />
                  <Tooltip
                    {...tooltipProps}
                    formatter={(value: unknown) => [num(Number(value), 0), "units"]}
                  />
                  <Bar dataKey="units" fill={CHART_COLORS.primary} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          <Panel title="Units by store">
            <div className="chart-box short" role="img" aria-label="Bar chart of total units by store">
              <ResponsiveContainer>
                <BarChart data={stores} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="key" {...xAxisProps} />
                  <YAxis {...yAxisProps} tickFormatter={numCompact} />
                  <Tooltip
                    {...tooltipProps}
                    formatter={(value: unknown) => [num(Number(value), 0), "units"]}
                  />
                  <Bar dataKey="units" fill={CHART_COLORS.primary} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>
      </Section>

      <Section title="Model evaluation" hint="chronological holdout window">
        <div className="metrics">
          <Metric
            label="Model h28 WAPE"
            value={model ? pct(model.wape_percent_by_horizon["28"]) : "—"}
            sub={model ? `version ${model.version}` : "no model trained yet"}
          />
          <Metric
            label="Mean baseline h28"
            value={meanWape !== undefined ? pct(meanWape) : "—"}
            sub="benchmark to beat"
          />
          <Metric
            label="Evaluation horizon"
            value={model ? `${model.evaluation_horizons.join(" / ")} d` : "—"}
            sub="recursive 28-day protocol"
          />
        </div>

        {model && baselineRows.length > 0 ? (
          <Panel className="mt-3" title="Holdout WAPE — model vs baselines" sub="lower is better · identical series, dates, and protocol">
            <div className="chart-box short" role="img" aria-label="Bar chart comparing holdout WAPE of the model against baselines">
              <ResponsiveContainer>
                <BarChart data={baselineRows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="name" {...xAxisProps} interval={0} tick={{ fill: "#67707d", fontSize: 11 }} />
                  <YAxis {...yAxisProps} unit="%" />
                  <Tooltip
                    {...tooltipProps}
                    formatter={(value: unknown) => [pct(Number(value)), "WAPE"]}
                  />
                  <Bar dataKey="wape" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                    {baselineRows.map((row) => (
                      <Cell key={row.name} fill={baselineColor(row.name)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            {modelWape !== undefined && meanWape !== undefined ? (
              <div className="note honesty">
                Reported as measured: on this holdout the gradient-boosting model{" "}
                <strong>{beatsMean ? "beats" : "does not beat"}</strong> the simple
                mean baseline ({pct(modelWape)} vs {pct(meanWape)} WAPE). All figures
                come from the same chronological validation window.
              </div>
            ) : null}
          </Panel>
        ) : (
          <Panel className="mt-3" title="Holdout WAPE — model vs baselines">
            <EmptyState>
              No model has been trained yet. Run{" "}
              <code>python -m src.forecasting.train</code> to produce artifacts.
            </EmptyState>
          </Panel>
        )}
      </Section>

      <Section title="Service status">
        {health.loading ? <Spinner label="Checking service status…" /> : null}
        {health.error ? (
          <div className="hint">Service status unavailable: {health.error}</div>
        ) : null}
        {status ? (
          <div className="metrics">
            <StatusMetric
              label="API"
              tone={status.status === "ok" ? "good" : "warn"}
              value={status.status === "ok" ? "Operational" : "Degraded"}
              sub={`service ${status.service} · v${status.api_version}`}
            />
            <StatusMetric
              label="Database"
              tone={dbTone}
              value={status.database.connected ? "Connected" : "Disconnected"}
              sub={status.database.error ?? "application store"}
            />
            <StatusMetric
              label="Data"
              tone={status.data.data_ready ? "good" : "warn"}
              value={status.data.data_ready ? "Ready" : "Not ready"}
              sub={
                status.data.missing_datasets?.length
                  ? `missing: ${status.data.missing_datasets.join(", ")}`
                  : "processed datasets loaded"
              }
            />
            <StatusMetric
              label="Forecast model"
              tone={status.model.ready ? "good" : "warn"}
              value={status.model.ready ? "Ready" : "Unavailable"}
              sub={status.model.version ?? status.model.error ?? "no artifact"}
            />
          </div>
        ) : null}
      </Section>

      <ul className="notes-list">
        {data.data_notes.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </>
  );
}
