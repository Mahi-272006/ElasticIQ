import { useState } from "react";
import { api } from "../api/client";
import type { ScoreResult } from "../api/types";
import { ErrorBox, Kpi, Note, PageHead, Spinner, StatusBadge, num, useAsync } from "../components/ui";

export default function Monitoring() {
  const drift = useAsync(() => api.drift(), []);
  const [scoreRows, setScoreRows] = useState([
    { actual_units: "5", predicted_units: "6", horizon: "1" },
    { actual_units: "3", predicted_units: "2", horizon: "2" },
  ]);
  const [score, setScore] = useState<ScoreResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runScore() {
    setBusy(true);
    setError(null);
    try {
      setScore(
        await api.score(
          scoreRows.map((row) => ({
            actual_units: Number(row.actual_units),
            predicted_units: Number(row.predicted_units),
            horizon: Number(row.horizon),
          })),
        ),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const staleness = drift.data?.staleness;

  return (
    <>
      <PageHead
        title="Monitoring"
        subtitle="Distribution drift (PSI), artifact staleness, offline evaluation replay, and live error scoring when actuals arrive."
      />

      {drift.loading ? <Spinner /> : null}
      {drift.error ? <ErrorBox message={drift.error} /> : null}

      {drift.data ? (
        <>
          <div className="grid cols-2">
            <div className="card">
              <h3>Demand drift — per-series mean units, last 28d vs prior 28d</h3>
              <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
                <div className="kpi-value">{num(drift.data.demand.psi, 4)}</div>
                <StatusBadge status={drift.data.demand.classification} />
              </div>
              <div className="kpi-sub">
                {num(drift.data.demand.series_compared, 0)} series compared · window{" "}
                {(drift.data.demand.current_window as string[] | undefined)?.[0]}–
                {(drift.data.demand.current_window as string[] | undefined)?.[1]}
              </div>
              <div className="kpi-sub" style={{ marginTop: 6 }}>
                Zero-demand share: {num((drift.data.demand.zero_share_reference as number) * 100, 1)}% →{" "}
                {num((drift.data.demand.zero_share_current as number) * 100, 1)}%
              </div>
              <Note>{String(drift.data.demand.note)}</Note>
            </div>

            <div className="card">
              <h3>Price drift — first vs last observed price per series</h3>
              <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
                <div className="kpi-value">{num(drift.data.price.psi, 4)}</div>
                <StatusBadge status={drift.data.price.classification} />
              </div>
              <div className="kpi-sub">
                {num(drift.data.price.series_compared, 0)} series ·{" "}
                {num((drift.data.price.share_of_series_with_any_price_change as number) * 100, 1)}%
                had any price change
              </div>
              <Note>{String(drift.data.price.note)}</Note>
            </div>
          </div>

          {staleness ? (
            <div className="card" style={{ marginTop: 14 }}>
              <h3>Staleness</h3>
              <div className="grid cols-3">
                <Kpi
                  label="Dataset last date"
                  value={String(staleness.dataset_last_date).slice(0, 10)}
                  sub={`${num(staleness.dataset_age_days_vs_wall_clock, 0)} days vs wall clock`}
                />
                <Kpi
                  label="Model version"
                  value={staleness.artifact?.available ? String(staleness.artifact.version) : "—"}
                  sub={
                    staleness.artifact?.available
                      ? `trained ${String(staleness.artifact.trained_at).slice(0, 10)}`
                      : "no artifact"
                  }
                />
                <Kpi
                  label="PSI thresholds"
                  value="< 0.10 stable"
                  sub={`moderate < ${String(drift.data.psi_thresholds.moderate_below)} · rule of thumb, not a test`}
                />
              </div>
              <Note honesty>{String(staleness.dataset_staleness_note)}</Note>
            </div>
          ) : null}
        </>
      ) : null}

      <div className="card" style={{ marginTop: 14 }}>
        <h3>Live error scoring — paste actuals when they arrive</h3>
        <p className="small muted" style={{ marginTop: 0 }}>
          Each row compares a forecast with its realized demand. This is the operational
          error-tracking entry point; below are editable example rows.
        </p>
        <table>
          <thead>
            <tr>
              <th>Actual units</th>
              <th>Predicted units</th>
              <th>Horizon</th>
            </tr>
          </thead>
          <tbody>
            {scoreRows.map((row, index) => (
              <tr key={index}>
                <td>
                  <input
                    style={{ width: 90 }}
                    value={row.actual_units}
                    onChange={(e) => {
                      const next = [...scoreRows];
                      next[index] = { ...row, actual_units: e.target.value };
                      setScoreRows(next);
                    }}
                  />
                </td>
                <td>
                  <input
                    style={{ width: 90 }}
                    value={row.predicted_units}
                    onChange={(e) => {
                      const next = [...scoreRows];
                      next[index] = { ...row, predicted_units: e.target.value };
                      setScoreRows(next);
                    }}
                  />
                </td>
                <td>
                  <input
                    style={{ width: 70 }}
                    value={row.horizon}
                    onChange={(e) => {
                      const next = [...scoreRows];
                      next[index] = { ...row, horizon: e.target.value };
                      setScoreRows(next);
                    }}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="controls" style={{ marginTop: 10 }}>
          <button
            className="ghost"
            onClick={() =>
              setScoreRows([...scoreRows, { actual_units: "", predicted_units: "", horizon: "1" }])
            }
          >
            Add row
          </button>
          <button onClick={runScore} disabled={busy}>
            {busy ? "Scoring…" : "Score"}
          </button>
        </div>
        {error ? <ErrorBox message={error} /> : null}
        {score ? (
          <div className="grid cols-3" style={{ marginTop: 8 }}>
            <Kpi label="WAPE" value={`${num(score.overall.wape_percent, 2)}%`} />
            <Kpi label="MAE" value={num(score.overall.mae_units, 2)} sub="units" />
            <Kpi
              label="Mean bias"
              value={num(score.overall.mean_bias_units, 2)}
              sub={score.overall.mean_bias_units > 0 ? "predictions run high" : "predictions run low"}
            />
          </div>
        ) : null}
      </div>

      <Note honesty>
        PSI compares distribution shapes and cannot explain causes. The M5 dataset ends in
        2016, so wall-clock staleness is expected by design and is reported for context,
        not as an alert.
      </Note>
    </>
  );
}
