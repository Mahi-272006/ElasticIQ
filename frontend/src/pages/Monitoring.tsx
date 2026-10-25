import { useState } from "react";
import { api } from "../api/client";
import type { ScoreResult } from "../api/types";
import {
  ErrorBox,
  Metric,
  Note,
  PageHead,
  Panel,
  Spinner,
  StatusBadge,
  num,
  useAsync,
} from "../components/ui";

function DriftPanel({
  title,
  sub,
  psi,
  classification,
  note,
  children,
}: {
  title: string;
  sub: string;
  psi: number;
  classification: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <Panel title={title} sub={sub}>
      <div className="inline spread">
        <span className="metric-label">PSI</span>
        <StatusBadge status={classification} />
      </div>
      <div className="metric-value">{num(psi, 4)}</div>
      <div className="metric-sub">{children}</div>
      <div className="note">{note}</div>
    </Panel>
  );
}

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
  const thresholds = drift.data?.psi_thresholds;

  return (
    <>
      <PageHead
        title="Monitoring"
        subtitle="Distribution drift (PSI), artifact freshness, offline evaluation replay, and live error scoring when actuals arrive."
      />

      {drift.loading ? <Spinner label="Loading drift report…" /> : null}
      {drift.error ? <ErrorBox message={drift.error} /> : null}

      {drift.data ? (
        <>
          <div className="grid cols-2">
            <DriftPanel
              title="Demand drift"
              sub="per-series mean units, last 28d vs prior 28d"
              psi={Number(drift.data.demand.psi)}
              classification={String(drift.data.demand.classification)}
              note={String(drift.data.demand.note)}
            >
              {num(drift.data.demand.series_compared, 0)} series compared · window{" "}
              {(drift.data.demand.current_window as string[] | undefined)?.[0]}–
              {(drift.data.demand.current_window as string[] | undefined)?.[1]} ·
              zero-demand share{" "}
              {num((drift.data.demand.zero_share_reference as number) * 100, 1)}% →{" "}
              {num((drift.data.demand.zero_share_current as number) * 100, 1)}%
            </DriftPanel>

            <DriftPanel
              title="Price drift"
              sub="first vs last observed price per series"
              psi={Number(drift.data.price.psi)}
              classification={String(drift.data.price.classification)}
              note={String(drift.data.price.note)}
            >
              {num(drift.data.price.series_compared, 0)} series ·{" "}
              {num(
                (drift.data.price.share_of_series_with_any_price_change as number) * 100,
                1,
              )}
              % had any price change
            </DriftPanel>
          </div>

          {thresholds ? (
            <Panel className="mt-3" title="Drift thresholds">
              <div className="metrics">
                <Metric
                  label="Stable"
                  value={`< ${String(thresholds.stable_below ?? "—")}`}
                  sub="PSI, no action indicated"
                />
                <Metric
                  label="Moderate"
                  value={`< ${String(thresholds.moderate_below ?? "—")}`}
                  sub="review the window"
                />
                <Metric
                  label="Significant"
                  value={`≥ ${String(thresholds.moderate_below ?? "—")}`}
                  sub="alert — investigate before trusting forecasts"
                />
                <Metric
                  label="Basis"
                  value="Rule of thumb"
                  sub="not a statistical test"
                />
              </div>
            </Panel>
          ) : null}

          {staleness ? (
            <Panel
              className="mt-3"
              title="Freshness"
              tag={staleness.artifact?.available ? "artifact available" : "no artifact"}
            >
              <div className="metrics">
                <Metric
                  label="Dataset last date"
                  value={String(staleness.dataset_last_date).slice(0, 10)}
                  sub={`${num(staleness.dataset_age_days_vs_wall_clock, 0)} days vs wall clock`}
                />
                <Metric
                  label="Model version"
                  value={
                    staleness.artifact?.available
                      ? String(staleness.artifact.version)
                      : "—"
                  }
                  sub={
                    staleness.artifact?.available
                      ? `trained ${String(staleness.artifact.trained_at).slice(0, 10)}`
                      : "no artifact"
                  }
                />
                <Metric
                  label="PSI thresholds"
                  value={`stable < ${String(thresholds?.stable_below ?? "—")}`}
                  sub={`moderate < ${String(thresholds?.moderate_below ?? "—")}`}
                />
              </div>
              <div className="note honesty">
                {String(staleness.dataset_staleness_note)}
              </div>
            </Panel>
          ) : null}
        </>
      ) : null}

      <Panel
        className={drift.data ? "mt-3" : undefined}
        title="Live error scoring"
        sub="paste actuals when they arrive — each row compares a forecast with its realized demand"
        tag="manual input"
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="num">Actual units</th>
                <th className="num">Predicted units</th>
                <th className="num">Horizon</th>
              </tr>
            </thead>
            <tbody>
              {scoreRows.map((row, index) => (
                <tr key={index}>
                  <td>
                    <input
                      className="input-sm"
                      aria-label={`Row ${index + 1} actual units`}
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
                      className="input-sm"
                      aria-label={`Row ${index + 1} predicted units`}
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
                      className="input-xs"
                      aria-label={`Row ${index + 1} horizon`}
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
        </div>
        <div className="controls mt-2">
          <button
            className="ghost"
            onClick={() =>
              setScoreRows([
                ...scoreRows,
                { actual_units: "", predicted_units: "", horizon: "1" },
              ])
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
          <div className="metrics mt-3">
            <Metric label="WAPE" value={`${num(score.overall.wape_percent, 2)}%`} />
            <Metric label="MAE" value={num(score.overall.mae_units, 2)} sub="units" />
            <Metric
              label="Mean bias"
              value={num(score.overall.mean_bias_units, 2)}
              sub={
                score.overall.mean_bias_units > 0
                  ? "predictions run high"
                  : score.overall.mean_bias_units < 0
                    ? "predictions run low"
                    : "unbiased on this sample"
              }
            />
            <Metric label="Rows scored" value={num(score.rows, 0)} sub={score.note} />
          </div>
        ) : (
          <div className="hint">
            Results (WAPE, MAE, mean bias) appear here after you score the rows above.
          </div>
        )}
      </Panel>

      <Note honesty>
        PSI compares distribution shapes and cannot explain causes; a flagged window
        is an <strong>alert to review</strong>, not a confirmed model failure. The M5
        dataset ends in 2016, so wall-clock staleness is expected by design and is
        reported for context, not as an alert.
      </Note>
    </>
  );
}
