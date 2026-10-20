import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { ErrorBox, Kpi, Note, PageHead, Spinner, num, useAsync } from "../components/ui";

export default function Overview() {
  const { data, error, loading } = useAsync(() => api.overview(), []);

  if (loading) return <Spinner label="Loading overview…" />;
  if (error) return <ErrorBox message={error} />;
  if (!data) return null;

  const { coverage, totals, categories, stores, model } = data;

  const baselineRows = model
    ? Object.entries(model.baseline_comparison.baselines_overall_wape_percent ?? {}).map(
        ([name, wape]) => ({ name: name.replaceAll("_", " "), wape }),
      )
    : [];

  if (model && model.baseline_comparison.model_overall_wape_percent !== undefined) {
    baselineRows.unshift({
      name: "model (HGB)",
      wape: model.baseline_comparison.model_overall_wape_percent,
    });
  }

  return (
    <>
      <PageHead
        title="Business overview"
        subtitle="Full-catalog daily aggregates from the M5 dataset, plus the holdout performance of the deployed forecasting model."
      />

      <div className="grid cols-4">
        <Kpi
          label="Total units"
          value={num(totals.total_units, 0)}
          sub={`${coverage.first_date} → ${coverage.last_date}`}
        />
        <Kpi label="Mean daily units" value={num(totals.mean_daily_units, 0)} sub="across the catalog" />
        <Kpi label="Series in catalog" value={num(coverage.series_in_catalog, 0)} sub={`${coverage.stores} stores · ${coverage.categories} categories`} />
        <Kpi
          label="Model h28 WAPE"
          value={model ? `${num(model.wape_percent_by_horizon["28"])}%` : "—"}
          sub={model ? `version ${model.version}` : "no model trained yet"}
        />
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Units by category</h3>
          <div className="chart-box short">
            <ResponsiveContainer>
              <BarChart data={categories}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                <XAxis dataKey="key" stroke="#8b98ad" fontSize={12} />
                <YAxis stroke="#8b98ad" fontSize={11} tickFormatter={(v) => num(v, 0)} />
                <Tooltip
                  contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                  formatter={(value: any) => [num(value, 0), "units"]}
                />
                <Bar dataKey="units" fill="#4f8cff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h3>Units by store</h3>
          <div className="chart-box short">
            <ResponsiveContainer>
              <BarChart data={stores}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                <XAxis dataKey="key" stroke="#8b98ad" fontSize={12} />
                <YAxis stroke="#8b98ad" fontSize={11} tickFormatter={(v) => num(v, 0)} />
                <Tooltip
                  contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                  formatter={(value: any) => [num(value, 0), "units"]}
                />
                <Bar dataKey="units" fill="#7a5cff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {model ? (
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Model vs baselines — holdout WAPE (lower is better)</h3>
          <div className="chart-box short">
            <ResponsiveContainer>
              <BarChart data={baselineRows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                <XAxis dataKey="name" stroke="#8b98ad" fontSize={12} />
                <YAxis stroke="#8b98ad" fontSize={11} unit="%" />
                <Tooltip
                  contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                  formatter={(value: any) => [`${num(value, 2)}%`, "WAPE"]}
                />
                <Bar dataKey="wape" fill="#d9a441" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Note honesty>
            Honest result: the gradient-boosting model beats both naive baselines but
            <strong> does not beat the simple mean baseline</strong> on this holdout
            (h28: {num(model.wape_percent_by_horizon["28"])}% vs{" "}
            {num(model.baseline_comparison.baselines_overall_wape_percent?.mean_baseline)}%).
            All numbers come from the same chronological validation window.
          </Note>
        </div>
      ) : null}

      <div className="note" style={{ marginTop: 14 }}>
        {data.data_notes.map((line) => (
          <div key={line}>• {line}</div>
        ))}
      </div>
    </>
  );
}
