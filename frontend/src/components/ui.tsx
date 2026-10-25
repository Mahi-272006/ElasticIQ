import type { ReactNode } from "react";
import { useEffect, useState } from "react";

/** Small async loader hook with loading/error state (no libraries). */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    fn()
      .then((result) => {
        if (alive) {
          setData(result);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (alive) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, version]);

  const refresh = () => setVersion((v) => v + 1);

  return { data, error, loading, refresh };
}

export function PageHead({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      {subtitle ? <p>{subtitle}</p> : null}
    </header>
  );
}

/** Open page section: heading with a hairline divider, no container box. */
export function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <section className="section">
      <div className="section-head">
        <h2 className="section-title">{title}</h2>
        {hint ? <span className="section-hint">{hint}</span> : null}
      </div>
      {children}
    </section>
  );
}

/** Functional bordered container for tables, charts, and filter groups. */
export function Panel({
  title,
  tag,
  sub,
  actions,
  children,
  className,
}: {
  title?: string;
  tag?: string;
  sub?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={className ? `panel ${className}` : "panel"}>
      {title || actions ? (
        <div className="panel-head">
          <div className="panel-head-text">
            {title ? <h3 className="panel-title">{title}</h3> : null}
            {tag ? <span className="tag">{tag}</span> : null}
          </div>
          {actions}
        </div>
      ) : null}
      {sub ? <div className="panel-sub">{sub}</div> : null}
      <div className="panel-body">{children}</div>
    </section>
  );
}

/** Open metric: label, tabular value, helper line — never a card. */
export function Metric({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub ? <div className="metric-sub">{sub}</div> : null}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="error-box" role="alert">
      {message}
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Note({
  children,
  honesty = false,
}: {
  children: ReactNode;
  honesty?: boolean;
}) {
  return <div className={honesty ? "note honesty" : "note"}>{children}</div>;
}

const TONE: Record<string, "good" | "warn" | "bad"> = {
  ok: "good",
  stable: "good",
  moderate: "warn",
  stockout_risk: "warn",
  overstock_risk: "warn",
  significant: "bad",
};

/**
 * Status pill. Only known states get a status color; anything else renders as
 * a neutral label so colors never imply meaning the backend did not state.
 */
export function StatusBadge({ status }: { status: string }) {
  const tone = TONE[status] ?? "neutral";
  return <span className={`badge ${tone}`}>{status.replaceAll("_", " ")}</span>;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="spin" role="status">
      {label ?? "Loading…"}
    </div>
  );
}

export function num(value: unknown, digits = 1): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  });
}

/** Percentage with a fixed number format (2 decimals by default). */
export function pct(value: unknown, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${num(value, digits)}%`;
}

/**
 * Compact formatter for axis ticks where full grouped numbers would not fit
 * the axis gutter (e.g. 65700000 -> "65.7M"). Values below one million keep
 * the regular grouped format.
 */
export function numCompact(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${num(value / 1e9, 1)}B`;
  if (abs >= 1e6) return `${num(value / 1e6, 1)}M`;
  if (abs >= 1e5) return `${num(value / 1e3, 1)}k`;
  return num(value, 0);
}
