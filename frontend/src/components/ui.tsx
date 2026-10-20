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
    <div className="page-head">
      <h1>{title}</h1>
      {subtitle ? <p>{subtitle}</p> : null}
    </div>
  );
}

export function Kpi({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="card">
      <h3>{label}</h3>
      <div className="kpi-value">{value}</div>
      {sub ? <div className="kpi-sub">{sub}</div> : null}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="error-box">{message}</div>;
}

export function Note({
  children,
  honesty = false,
}: {
  children: React.ReactNode;
  honesty?: boolean;
}) {
  return <div className={honesty ? "note honesty" : "note"}>{children}</div>;
}

export function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "ok" || status === "stable" || status === "ok"
      ? "good"
      : status === "moderate" || status === "stockout_risk"
        ? "warn"
        : "bad";
  return <span className={`badge ${cls}`}>{status.replace("_", " ")}</span>;
}

export function Spinner({ label }: { label?: string }) {
  return <div className="spin">{label ?? "Loading…"}</div>;
}

export function num(value: unknown, digits = 1): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  });
}
