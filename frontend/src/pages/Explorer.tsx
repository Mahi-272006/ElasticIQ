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
import type { ProductDetail } from "../api/types";
import { Icon } from "../components/Icon";
import {
  CHART_COLORS,
  fmtDateShort,
  gridProps,
  tooltipProps,
  xAxisProps,
  yAxisProps,
} from "../components/charts";
import {
  ErrorBox,
  Note,
  PageHead,
  Panel,
  Spinner,
  num,
  useAsync,
} from "../components/ui";

const PAGE = 25;

export default function Explorer() {
  const [tab, setTab] = useState<"products" | "stores">("products");
  const [search, setSearch] = useState("");
  const [catId, setCatId] = useState("");
  const [storeId, setStoreId] = useState("");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);

  const products = useAsync(
    () =>
      api.products({
        limit: PAGE,
        offset,
        search: search || undefined,
        cat_id: catId || undefined,
        store_id: storeId || undefined,
      }),
    [offset, search, catId, storeId],
  );
  const stores = useAsync(() => api.stores(), []);
  const detail = useAsync<ProductDetail | null>(
    () => (selected ? api.product(selected) : Promise.resolve(null)),
    [selected],
  );

  return (
    <>
      <PageHead
        title="Explorer"
        subtitle="The full 30,490-series M5 catalog (one series = one item in one store), with observed history and price statistics."
      />

      <div className="toolbar">
        <div className="tabs" role="tablist" aria-label="Catalog view">
          <button
            role="tab"
            aria-selected={tab === "products"}
            className={tab === "products" ? "tab active" : "tab"}
            onClick={() => setTab("products")}
          >
            Products
          </button>
          <button
            role="tab"
            aria-selected={tab === "stores"}
            className={tab === "stores" ? "tab active" : "tab"}
            onClick={() => setTab("stores")}
          >
            Stores
          </button>
        </div>

        {tab === "products" ? (
          <>
            <label className="field" style={{ minWidth: 210 }}>
              Search
              <span className="input-with-icon">
                <span className="input-icon">
                  <Icon name="search" />
                </span>
                <input
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    setOffset(0);
                  }}
                  placeholder="FOODS_3_090"
                />
              </span>
            </label>
            <label className="field">
              Category
              <select
                value={catId}
                onChange={(e) => {
                  setCatId(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="">All</option>
                <option value="FOODS">FOODS</option>
                <option value="HOBBIES">HOBBIES</option>
                <option value="HOUSEHOLD">HOUSEHOLD</option>
              </select>
            </label>
            <label className="field">
              Store
              <select
                value={storeId}
                onChange={(e) => {
                  setStoreId(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="">All</option>
                {stores.data?.items.map((s) => (
                  <option key={s.store_id} value={s.store_id}>
                    {s.store_id}
                  </option>
                ))}
              </select>
            </label>
          </>
        ) : null}
      </div>

      {tab === "products" ? (
        <>
          {products.loading ? <Spinner label="Loading catalog…" /> : null}
          {products.error ? <ErrorBox message={products.error} /> : null}

          {products.data ? (
            <Panel>
              {products.data.items.length === 0 ? (
                <div className="empty">No series match these filters.</div>
              ) : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Series</th>
                        <th>Dept</th>
                        <th>Category</th>
                        <th>Store</th>
                        <th>State</th>
                      </tr>
                    </thead>
                    <tbody>
                      {products.data.items.map((p) => (
                        <tr
                          key={p.series_id}
                          className="row-click"
                          tabIndex={0}
                          onClick={() => setSelected(p.series_id)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") setSelected(p.series_id);
                          }}
                        >
                          <td className="mono">{p.series_id}</td>
                          <td>{p.dept_id}</td>
                          <td>{p.cat_id}</td>
                          <td>{p.store_id}</td>
                          <td>{p.state_id}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="pager">
                <button
                  className="icon-btn"
                  aria-label="Previous page"
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - PAGE))}
                >
                  <Icon name="chevron-left" />
                </button>
                <button
                  className="icon-btn"
                  aria-label="Next page"
                  disabled={offset + PAGE >= products.data.total}
                  onClick={() => setOffset(offset + PAGE)}
                >
                  <Icon name="chevron-right" />
                </button>
                <span className="tnum">
                  {products.data.total === 0
                    ? "0 results"
                    : `${offset + 1}–${Math.min(offset + PAGE, products.data.total)} of ${num(products.data.total, 0)}`}
                </span>
              </div>
            </Panel>
          ) : null}
        </>
      ) : (
        <Panel>
          {stores.loading ? <Spinner label="Loading stores…" /> : null}
          {stores.error ? <ErrorBox message={stores.error} /> : null}
          {stores.data ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Store</th>
                    <th>State</th>
                    <th className="num">Total units</th>
                    <th className="num">Catalog share</th>
                    <th>Coverage</th>
                  </tr>
                </thead>
                <tbody>
                  {stores.data.items.map((s) => (
                    <tr key={s.store_id}>
                      <td className="mono">{s.store_id}</td>
                      <td>{s.state_id}</td>
                      <td className="num">{num(s.total_units, 0)}</td>
                      <td className="num">{num(s.share_percent, 2)}%</td>
                      <td className="nowrap">
                        {s.first_date} → {s.last_date}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </Panel>
      )}

      {selected ? (
        <Panel
          className="mt-3"
          title={`Series detail — ${selected}`}
          actions={
            <button
              className="icon-btn"
              aria-label="Close series detail"
              onClick={() => setSelected(null)}
            >
              <Icon name="close" />
            </button>
          }
        >
          {detail.loading ? <Spinner label="Loading series…" /> : null}
          {detail.error ? <ErrorBox message={detail.error} /> : null}
          {detail.data ? (
            <div className="grid cols-2">
              <div>
                <div
                  className="chart-box short"
                  role="img"
                  aria-label={`Observed units for ${selected}`}
                >
                  <ResponsiveContainer>
                    <LineChart data={detail.data.history} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                      <CartesianGrid {...gridProps} />
                      <XAxis
                        dataKey="date"
                        {...xAxisProps}
                        minTickGap={40}
                        tickFormatter={(v: string) => fmtDateShort(v)}
                      />
                      <YAxis {...yAxisProps} />
                      <Tooltip
                        {...tooltipProps}
                        formatter={(value: unknown) => [num(Number(value), 0), "Units"]}
                      />
                      <Line
                        type="monotone"
                        dataKey="units"
                        stroke={CHART_COLORS.primary}
                        dot={false}
                        strokeWidth={1.5}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="hint">
                  Observed units, last {detail.data.history.length} days ·
                  zero-demand share{" "}
                  {detail.data.zero_demand_share !== null
                    ? `${num(detail.data.zero_demand_share * 100, 1)}%`
                    : "—"}
                </div>
              </div>
              <div className="table-wrap">
                <table>
                  <tbody>
                    {detail.data.price_stats
                      ? Object.entries(detail.data.price_stats)
                          .filter(([key]) => !["store_id", "item_id"].includes(key))
                          .map(([key, value]) => (
                            <tr key={key}>
                              <td>{key.replaceAll("_", " ")}</td>
                              <td className="num">
                                {typeof value === "number" ? num(value, 2) : String(value)}
                              </td>
                            </tr>
                          ))
                      : (
                          <tr>
                            <td className="muted">No price statistics</td>
                          </tr>
                        )}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </Panel>
      ) : null}

      <Note>
        History shown here comes from the processed 80-day tail used for inference;
        the aggregates on other pages cover the full 1,913-day history for all series.
      </Note>
    </>
  );
}
