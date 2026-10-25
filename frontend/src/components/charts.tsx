/**
 * Shared Recharts theming: one restrained palette, subtle gridlines and axes,
 * consistent light tooltips. Every chart in the product spreads these props so
 * colors never drift between pages.
 */

const axisTick = { fill: "#67707d", fontSize: 11 };

export const CHART_COLORS = {
  /** primary series: actuals, model output, forecast */
  primary: "#0d7a63",
  /** secondary series: observed history, mean baseline */
  slate: "#475569",
  /** reference marks, naive baselines */
  amber: "#b45309",
  /** tertiary baselines */
  grey: "#94a3b8",
} as const;

export const xAxisProps = {
  stroke: "#d7dde3",
  tick: axisTick,
  tickLine: false,
  axisLine: { stroke: "#d7dde3" },
};

export const yAxisProps = {
  stroke: "#d7dde3",
  tick: axisTick,
  tickLine: false,
  axisLine: false,
  width: 46,
};

export const gridProps = {
  stroke: "#eef1f4",
  strokeDasharray: "3 3",
  vertical: false,
};

export const tooltipProps = {
  contentStyle: {
    background: "#ffffff",
    border: "1px solid #e3e8ec",
    borderRadius: 6,
    boxShadow: "0 6px 20px rgba(20, 30, 40, 0.10)",
    fontSize: 12,
    padding: "8px 10px",
    color: "#1b2430",
  },
  labelStyle: { color: "#67707d", fontSize: 11 },
  cursor: { stroke: "#cfd6dd", strokeWidth: 1 },
};

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/**
 * Format an ISO date string ("2016-04-25") for axis ticks. Parses the string
 * manually so no timezone conversion can shift the displayed day.
 */
export function fmtDateShort(iso: string, withYear = false): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!match) return iso;
  const month = MONTHS[Number(match[2]) - 1] ?? match[2];
  const day = Number(match[3]);
  return withYear ? `${month} ${day}, ${match[1]}` : `${month} ${day}`;
}

/** Full date for tooltip labels: "Apr 25, 2016". */
export function fmtDateFull(iso: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!match) return iso;
  const month = MONTHS[Number(match[2]) - 1] ?? match[2];
  return `${month} ${Number(match[3])}, ${match[1]}`;
}
