import { useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { OptimizeResult, PricingScenarioResponse } from "../api/types";
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
  num,
} from "../components/ui";

interface FormState {
  current_price: string;
  candidate_price: string;
  unit_cost: string;
  baseline_units: string;
  demand_model: "constant" | "elasticity" | "linear_slope";
  elasticity: string;
  linear_slope: string;
  min_price: string;
  max_price: string;
  max_change_percent: string;
}

const INITIAL: FormState = {
  current_price: "2.50",
  candidate_price: "2.25",
  unit_cost: "1.20",
  baseline_units: "40",
  demand_model: "elasticity",
  elasticity: "-1.5",
  linear_slope: "-2.0",
  min_price: "1.00",
  max_price: "5.00",
  max_change_percent: "",
};

function buildConstraints(form: FormState): Record<string, unknown> {
  const constraints: Record<string, unknown> = {
    min_price: Number(form.min_price),
    max_price: Number(form.max_price),
  };
  if (form.max_change_percent.trim() !== "") {
    constraints.max_change_percent = Number(form.max_change_percent);
  }
  return constraints;
}

export default function Pricing() {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [scenario, setScenario] = useState<PricingScenarioResponse | null>(null);
  const [optimize, setOptimize] = useState<OptimizeResult | null>(null);
  const [sweep, setSweep] = useState<any[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = (key: keyof FormState) => (e: { target: { value: string } }) =>
    setForm({ ...form, [key]: e.target.value });

  function payload(): Record<string, unknown> {
    const base: Record<string, unknown> = {
      current_price: Number(form.current_price),
      unit_cost: Number(form.unit_cost),
      baseline_units: Number(form.baseline_units),
      demand_model: form.demand_model,
      horizon_days: 28,
      constraints: buildConstraints(form),
    };
    if (form.demand_model === "elasticity") base.elasticity = Number(form.elasticity);
    if (form.demand_model === "linear_slope") base.linear_slope = Number(form.linear_slope);
    return base;
  }

  async function runScenario() {
    setBusy(true);
    setError(null);
    try {
      setScenario(
        await api.pricingScenario({
          ...payload(),
          candidate_price: Number(form.candidate_price),
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function runOptimize() {
    setBusy(true);
    setError(null);
    try {
      setOptimize(await api.pricingOptimize(payload()));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function runSweep() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.pricingSensitivity({
        baseline_units: Number(form.baseline_units),
        current_price: Number(form.current_price),
        candidate_price: Number(form.candidate_price),
        unit_cost: Number(form.unit_cost),
        constraints: buildConstraints(form),
      });
      setSweep(result.elasticity_sweep);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const signed = (value: number, digits = 2) =>
    `${value > 0 ? "+" : ""}${num(value, digits)}%`;

  return (
    <>
      <PageHead
        title="Pricing what-if"
        subtitle="Simulate a price change or search for the profit-maximizing price under an explicit demand-response assumption. The M5 data is observational, so no causal price effect is claimed or implied."
      />

      <div className="grid cols-2">
        <Panel title="Scenario inputs" tag="your inputs">
          <div className="form-legend">Price inputs</div>
          <div className="form-grid">
            <label className="field">
              Current price
              <input value={form.current_price} onChange={set("current_price")} />
            </label>
            <label className="field">
              Candidate price
              <input value={form.candidate_price} onChange={set("candidate_price")} />
            </label>
            <label className="field">
              Unit cost
              <input value={form.unit_cost} onChange={set("unit_cost")} />
            </label>
            <label className="field">
              Baseline units (horizon)
              <input value={form.baseline_units} onChange={set("baseline_units")} />
            </label>
          </div>

          <div className="form-legend">Demand assumption</div>
          <div className="form-grid">
            <label className="field">
              Model
              <select value={form.demand_model} onChange={set("demand_model")}>
                <option value="constant">constant (no response)</option>
                <option value="elasticity">constant elasticity</option>
                <option value="linear_slope">linear slope</option>
              </select>
            </label>
            {form.demand_model === "elasticity" ? (
              <label className="field">
                Elasticity (negative)
                <input value={form.elasticity} onChange={set("elasticity")} />
              </label>
            ) : null}
            {form.demand_model === "linear_slope" ? (
              <label className="field">
                Units per +1 price
                <input value={form.linear_slope} onChange={set("linear_slope")} />
              </label>
            ) : null}
          </div>

          <div className="form-legend">Constraints</div>
          <div className="form-grid">
            <label className="field">
              Min price
              <input value={form.min_price} onChange={set("min_price")} />
            </label>
            <label className="field">
              Max price
              <input value={form.max_price} onChange={set("max_price")} />
            </label>
            <label className="field">
              Max change % (optional)
              <input
                value={form.max_change_percent}
                onChange={set("max_change_percent")}
                placeholder="e.g. 20"
              />
            </label>
          </div>

          <div className="controls mt-3">
            <button onClick={runScenario} disabled={busy}>
              Run scenario
            </button>
            <button className="ghost" onClick={runOptimize} disabled={busy}>
              Optimize price
            </button>
            <button className="ghost" onClick={runSweep} disabled={busy}>
              Sensitivity sweep
            </button>
          </div>
          {busy ? <Spinner label="Computing…" /> : null}
          {error ? <ErrorBox message={error} /> : null}
        </Panel>

        <Panel title="Outcome" tag="modeled under your assumption">
          {scenario ? (
            <>
              <div className="metrics">
                <Metric
                  label="Revenue change"
                  value={signed(scenario.outcome.revenue_change_percent)}
                  sub={`${num(scenario.outcome.current_revenue)} → ${num(scenario.outcome.candidate_revenue)}`}
                />
                <Metric
                  label="Profit change"
                  value={signed(scenario.outcome.profit_change_percent)}
                  sub={`${num(scenario.outcome.current_gross_profit)} → ${num(scenario.outcome.candidate_gross_profit)}`}
                />
                <Metric
                  label="Expected units"
                  value={`${num(scenario.outcome.expected_units_current, 1)} → ${num(scenario.outcome.expected_units_candidate, 1)}`}
                  sub="current → candidate"
                />
                <Metric
                  label="Assumed demand model"
                  value={String(scenario.request.demand_model)}
                  sub={
                    scenario.request.demand_model === "elasticity"
                      ? `ε = ${String(scenario.request.elasticity)}`
                      : scenario.request.demand_model === "linear_slope"
                        ? `slope = ${String(scenario.request.linear_slope)}`
                        : "demand held constant"
                  }
                />
              </div>
              {scenario.outcome.warnings.length > 0 ? (
                <ul className="assumptions">
                  {scenario.outcome.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              ) : null}
              <Note honesty>{scenario.honesty_note}</Note>
            </>
          ) : (
            <EmptyState>
              Run a scenario to see revenue and profit deltas under your stated
              assumption. Every figure here is modeled, not observed.
            </EmptyState>
          )}
        </Panel>
      </div>

      {optimize ? (
        <Panel className="mt-3" title={`Optimization — ${optimize.objective}`} tag="modeled">
          <div className="metrics">
            <Metric label="Optimal price" value={num(optimize.optimal_price, 2)} />
            <Metric label="Optimal gross profit" value={num(optimize.optimal_gross_profit, 2)} />
            <Metric label="Improvement vs current" value={`${num(optimize.improvement_percent, 2)}%`} />
            <Metric
              label="Feasible prices"
              value={`${optimize.feasible_prices} / ${optimize.evaluated_prices}`}
            />
          </div>
          <div
            className="chart-box mt-3"
            role="img"
            aria-label="Expected gross profit as a function of price"
          >
            <ResponsiveContainer>
              <LineChart data={optimize.all_prices} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis
                  dataKey="price"
                  {...xAxisProps}
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tickFormatter={(v: number) => num(v, 2)}
                />
                <YAxis {...yAxisProps} tickFormatter={(v: number) => num(v, 0)} />
                <Tooltip
                  {...tooltipProps}
                  formatter={(value: unknown, name: unknown) => [
                    num(Number(value), 2),
                    String(name),
                  ]}
                />
                <Line
                  type="monotone"
                  dataKey="gross_profit"
                  stroke={CHART_COLORS.primary}
                  dot={false}
                  strokeWidth={2}
                  name="Expected gross profit"
                  isAnimationActive={false}
                />
                <ReferenceDot
                  x={optimize.optimal_price}
                  y={optimize.optimal_gross_profit}
                  r={4}
                  fill={CHART_COLORS.amber}
                  stroke="none"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <ul className="assumptions">
            {optimize.assumptions.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </Panel>
      ) : null}

      {sweep ? (
        <Panel
          className="mt-3"
          title="Sensitivity sweep"
          sub="same price move, different assumed elasticities"
          tag="modeled"
        >
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="num">Assumed ε</th>
                  <th className="num">Expected units</th>
                  <th className="num">Revenue</th>
                  <th className="num">Gross profit</th>
                  <th className="num">Profit change</th>
                </tr>
              </thead>
              <tbody>
                {sweep.map((row) => (
                  <tr key={row.elasticity}>
                    <td className="num">{row.elasticity}</td>
                    <td className="num">{num(row.expected_units, 2)}</td>
                    <td className="num">{num(row.candidate_revenue, 2)}</td>
                    <td className="num">{num(row.candidate_gross_profit, 2)}</td>
                    <td className="num">{num(row.profit_change_percent, 2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Note honesty>
            The spread across rows is the honest answer: the data cannot identify
            which elasticity is true, so the decision depends on the assumption you
            bring.
          </Note>
        </Panel>
      ) : null}
    </>
  );
}
