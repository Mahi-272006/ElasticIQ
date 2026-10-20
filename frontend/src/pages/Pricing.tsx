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
import { ErrorBox, Kpi, Note, PageHead, Spinner, num } from "../components/ui";

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

  return (
    <>
      <PageHead
        title="Pricing what-if"
        subtitle="Simulate a price change or search for the profit-maximizing price under an explicit demand-response assumption. The M5 data is observational, so no causal price effect is claimed or implied."
      />

      <div className="grid cols-2">
        <div className="card">
          <h3>Scenario inputs</h3>
          <div className="controls">
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
          <div className="controls">
            <label className="field">
              Demand model
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
          <div className="controls">
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
          <div className="controls" style={{ marginBottom: 0 }}>
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
        </div>

        <div className="card">
          <h3>Outcome</h3>
          {scenario ? (
            <>
              <div className="grid cols-2">
                <Kpi
                  label="Revenue change"
                  value={`${scenario.outcome.revenue_change_percent > 0 ? "+" : ""}${num(scenario.outcome.revenue_change_percent, 2)}%`}
                  sub={`${num(scenario.outcome.current_revenue)} → ${num(scenario.outcome.candidate_revenue)}`}
                />
                <Kpi
                  label="Profit change"
                  value={`${scenario.outcome.profit_change_percent > 0 ? "+" : ""}${num(scenario.outcome.profit_change_percent, 2)}%`}
                  sub={`${num(scenario.outcome.current_gross_profit)} → ${num(scenario.outcome.candidate_gross_profit)}`}
                />
                <Kpi
                  label="Expected units"
                  value={`${num(scenario.outcome.expected_units_current, 1)} → ${num(scenario.outcome.expected_units_candidate, 1)}`}
                  sub="current → candidate"
                />
                <Kpi
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
            <div className="muted small">
              Run a scenario to see revenue/profit deltas under your stated assumption.
            </div>
          )}
        </div>
      </div>

      {optimize ? (
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Optimization — {optimize.objective}</h3>
          <div className="grid cols-4">
            <Kpi label="Optimal price" value={num(optimize.optimal_price, 2)} />
            <Kpi label="Optimal gross profit" value={num(optimize.optimal_gross_profit, 2)} />
            <Kpi
              label="Improvement vs current"
              value={`${num(optimize.improvement_percent, 2)}%`}
            />
            <Kpi
              label="Feasible prices"
              value={`${optimize.feasible_prices} / ${optimize.evaluated_prices}`}
            />
          </div>
          <div className="chart-box" style={{ marginTop: 12 }}>
            <ResponsiveContainer>
              <LineChart data={optimize.all_prices}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                <XAxis
                  dataKey="price"
                  stroke="#8b98ad"
                  fontSize={11}
                  type="number"
                  domain={["dataMin", "dataMax"]}
                />
                <YAxis stroke="#8b98ad" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                  formatter={(value: any, name: any) => [num(value, 2), name]}
                />
                <Line
                  type="monotone"
                  dataKey="gross_profit"
                  stroke="#3fb97f"
                  dot={false}
                  strokeWidth={2}
                  name="Expected gross profit"
                />
                <ReferenceDot
                  x={optimize.optimal_price}
                  y={optimize.optimal_gross_profit}
                  r={5}
                  fill="#d9a441"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <ul className="assumptions">
            {optimize.assumptions.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {sweep ? (
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Sensitivity sweep — same price move, different assumed elasticities</h3>
          <table>
            <thead>
              <tr>
                <th>Assumed ε</th>
                <th>Expected units</th>
                <th>Revenue</th>
                <th>Gross profit</th>
                <th>Profit change</th>
              </tr>
            </thead>
            <tbody>
              {sweep.map((row) => (
                <tr key={row.elasticity}>
                  <td>{row.elasticity}</td>
                  <td>{num(row.expected_units, 2)}</td>
                  <td>{num(row.candidate_revenue, 2)}</td>
                  <td>{num(row.candidate_gross_profit, 2)}</td>
                  <td>{num(row.profit_change_percent, 2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Note honesty>
            The spread across rows is the honest answer: the data cannot identify which
            elasticity is true, so the decision depends on the assumption you bring.
          </Note>
        </div>
      ) : null}
    </>
  );
}
