import { useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
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
  EmptyState,
  ErrorBox,
  PageHead,
  Panel,
  Spinner,
  num,
  numCompact,
  useAsync,
} from "../components/ui";

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

      <div className="toolbar">
        <label className="field">
          Window
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {[30, 90, 180, 365, 730].map((d) => (
              <option key={d} value={d}>
                {d} days
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

      {loading ? <Spinner label="Loading trends…" /> : null}
      {error ? <ErrorBox message={error} /> : null}

      {data ? (
        <Panel
          title={data.source_aggregate}
          sub={`${num(data.total_units, 0)} units over ${data.days} days · ${data.store_id ?? "all stores"} · ${data.cat_id ?? "all categories"}`}
        >
          {data.points.length === 0 ? (
            <EmptyState>No data matches this filter.</EmptyState>
          ) : (
            <div
              className="chart-box"
              role="img"
              aria-label={`Line chart of daily units for ${data.source_aggregate}`}
            >
              <ResponsiveContainer>
                <LineChart data={data.points} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis
                    dataKey="date"
                    {...xAxisProps}
                    minTickGap={48}
                    tickFormatter={(v: string) => fmtDateShort(v, days > 365)}
                  />
                  <YAxis
                    {...yAxisProps}
                    width={64}
                    tickFormatter={numCompact}
                    label={{
                      value: "Units / day",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#67707d",
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    {...tooltipProps}
                    labelFormatter={(label) => fmtDateFull(String(label))}
                    formatter={(value: unknown) => [num(Number(value), 0), "Units"]}
                  />
                  <Line
                    type="monotone"
                    dataKey="units"
                    stroke={CHART_COLORS.primary}
                    dot={false}
                    strokeWidth={1.6}
                    name="Units sold"
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>
      ) : null}

      <div className="note">
        Source: M5 <code>sales_train_validation</code> aggregated by the offline
        pipeline (Parquet). No interpolation or smoothing is applied.
      </div>
    </>
  );
}
