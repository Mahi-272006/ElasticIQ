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
import { ErrorBox, Note, PageHead, Spinner, num, useAsync } from "../components/ui";

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
        title="Products & stores"
        subtitle="The full 30,490-series M5 catalog (one series = one item in one store), with observed history and price statistics."
      />

      <div className="controls">
        <button
          className={tab === "products" ? "" : "ghost"}
          onClick={() => setTab("products")}
        >
          Products
        </button>
        <button
          className={tab === "stores" ? "" : "ghost"}
          onClick={() => setTab("stores")}
        >
          Stores
        </button>
      </div>

      {tab === "products" ? (
        <>
          <div className="controls">
            <label className="field">
              Search
              <input
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setOffset(0);
                }}
                placeholder="FOODS_3_090"
                style={{ minWidth: 220 }}
              />
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
          </div>

          {products.loading ? <Spinner /> : null}
          {products.error ? <ErrorBox message={products.error} /> : null}

          {products.data ? (
            <div className="card">
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
                      onClick={() => setSelected(p.series_id)}
                    >
                      <td>{p.series_id}</td>
                      <td>{p.dept_id}</td>
                      <td>{p.cat_id}</td>
                      <td>{p.store_id}</td>
                      <td>{p.state_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="pager">
                <button
                  className="ghost"
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - PAGE))}
                >
                  ← Prev
                </button>
                <span>
                  {offset + 1}–{Math.min(offset + PAGE, products.data.total)} of{" "}
                  {num(products.data.total, 0)}
                </span>
                <button
                  className="ghost"
                  disabled={offset + PAGE >= products.data.total}
                  onClick={() => setOffset(offset + PAGE)}
                >
                  Next →
                </button>
              </div>
            </div>
          ) : null}
        </>
      ) : (
        <div className="card">
          {stores.loading ? <Spinner /> : null}
          {stores.error ? <ErrorBox message={stores.error} /> : null}
          {stores.data ? (
            <table>
              <thead>
                <tr>
                  <th>Store</th>
                  <th>State</th>
                  <th>Total units</th>
                  <th>Catalog share</th>
                  <th>Coverage</th>
                </tr>
              </thead>
              <tbody>
                {stores.data.items.map((s) => (
                  <tr key={s.store_id}>
                    <td>{s.store_id}</td>
                    <td>{s.state_id}</td>
                    <td>{num(s.total_units, 0)}</td>
                    <td>{num(s.share_percent, 2)}%</td>
                    <td>
                      {s.first_date} → {s.last_date}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      )}

      {selected ? (
        <div className="card" style={{ marginTop: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <h3 style={{ margin: 0 }}>Series detail — {selected}</h3>
            <button className="ghost" onClick={() => setSelected(null)}>
              Close
            </button>
          </div>
          {detail.loading ? <Spinner /> : null}
          {detail.error ? <ErrorBox message={detail.error} /> : null}
          {detail.data ? (
            <div className="grid cols-2" style={{ marginTop: 12 }}>
              <div>
                <div className="chart-box short">
                  <ResponsiveContainer>
                    <LineChart data={detail.data.history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a3345" />
                      <XAxis
                        dataKey="date"
                        stroke="#8b98ad"
                        fontSize={10}
                        minTickGap={40}
                        tickFormatter={(v: string) => v.slice(5)}
                      />
                      <YAxis stroke="#8b98ad" fontSize={11} />
                      <Tooltip
                        contentStyle={{ background: "#1a2130", border: "1px solid #2a3345" }}
                        formatter={(value: any) => [num(value, 0), "units"]}
                      />
                      <Line
                        type="monotone"
                        dataKey="units"
                        stroke="#4f8cff"
                        dot={false}
                        strokeWidth={1.5}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="small muted">
                  Observed units, last {detail.data.history.length} days. Zero-demand
                  share:{" "}
                  {detail.data.zero_demand_share !== null
                    ? `${num(detail.data.zero_demand_share * 100, 1)}%`
                    : "—"}
                </div>
              </div>
              <table>
                <tbody>
                  {detail.data.price_stats
                    ? Object.entries(detail.data.price_stats)
                        .filter(([key]) => !["store_id", "item_id"].includes(key))
                        .map(([key, value]) => (
                          <tr key={key}>
                            <td>{key.replaceAll("_", " ")}</td>
                            <td>{typeof value === "number" ? num(value, 2) : String(value)}</td>
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
          ) : null}
        </div>
      ) : null}

      <Note>
        History shown here comes from the processed 80-day tail used for inference; the
        aggregates on other pages cover the full 1,913-day history for all series.
      </Note>
    </>
  );
}
