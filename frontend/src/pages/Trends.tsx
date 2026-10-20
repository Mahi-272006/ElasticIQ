import { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { ErrorBox, Note, PageHead, Spinner, num, useAsync } from "../components/ui";

const STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"];
const CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"];

export default function Trends() {
  const [days, setDays] = useState(90);
  const [storeId, setStoreId] = useState("");
  const [catId, setCatId] = useState("");

  const { data, error, loading } = useAsync(
    () => api.trends({ days, store_id: storeId || undefined, cat_id: catId || undefined }),
    [days, storeId, catId],
  );

  return (
    <>
      <PageHead
        title="Sales trends"
        subtitle="Daily unit demand from full-catalog aggregates. Filter by one dimension at a time — the processed aggregates do not include store × category slices."
      />

      <div className="controls">
        <label className="field">
          Window (days)
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {[30, 90, 180, 365, 730].map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Store
          <select
            value={storeId}
            onChange={(e) => {
              setStoreId(e.target.value);
              if (e.target.value) setCatId("");
            }}
          >
            <option value="">All stores</option>
            {STORES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Category
          <select
            value={catId}
            onChange={(e) => {
              setCatId(e.target.value);
              if (e.target.value) setStoreId("");
            }}
          >
            <option value="">All categories</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      </div>

      {loading ? <Spinner /> : null}
      {error ? <ErrorBox message={error} /> : null}

      {data ? (
        <div className="card">
          <h3>
            {data.source_aggregate} — {num(data.total_units, 0)} units over {data.days} days
          </h3>
          <div className="chart-box">
            <ResponsiveContainer>
              <LineChart data={data.points}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                <XAxis
                  dataKey="date"
                  stroke="#8b98ad"
                  fontSize={11}
                  minTickGap={40}
                  tickFormatter={(v: string) => v.slice(0, 7)}
                />
                <YAxis stroke="#8b98ad" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                  labelFormatter={(v: any) => `Date: ${v}`}
                  formatter={(value: any) => [num(value, 0), "units"]}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="units"
                  stroke="#4f8cff"
                  dot={false}
                  strokeWidth={1.6}
                  name="Units sold"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <Note>
            Source: M5 <code>sales_train_validation</code> aggregated by the offline
            pipeline (Parquet). No interpolation or smoothing is applied.
          </Note>
        </div>
      ) : null}
    </>
  );
}
