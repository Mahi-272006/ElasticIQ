/**
 * Inline SVG icon set — one consistent stroke style (16px grid, 1.7 stroke,
 * currentColor) with no external icon dependency.
 */

import type { ReactNode, SVGProps } from "react";

export type IconName =
  | "overview"
  | "trends"
  | "forecast"
  | "explorer"
  | "pricing"
  | "inventory"
  | "models"
  | "monitoring"
  | "menu"
  | "close"
  | "chevron-left"
  | "chevron-right"
  | "search";

const PATHS: Record<IconName, ReactNode> = {
  overview: (
    <>
      <rect x="3" y="3" width="7.5" height="7.5" rx="1.5" />
      <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5" />
      <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5" />
      <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5" />
    </>
  ),
  trends: (
    <>
      <path d="M3.5 3.5v17h17" />
      <polyline points="6.5 15 10.5 10.5 14 13.5 19 7" />
    </>
  ),
  forecast: (
    <>
      <rect x="3" y="5" width="18" height="16" rx="2" />
      <path d="M3 10h18" />
      <path d="M8 3v4M16 3v4" />
      <polyline points="8 17 11 14 14 16 17 13" />
    </>
  ),
  explorer: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 9.5h18" />
      <path d="M9.5 9.5V20" />
      <path d="M14 13.5h4M14 16.5h4" />
    </>
  ),
  pricing: (
    <>
      <path d="M20.6 13.4 12.4 21.6a1.7 1.7 0 0 1-2.4 0L3 14.6V4h10.6l7 7a1.7 1.7 0 0 1 0 2.4Z" />
      <circle cx="7.8" cy="7.8" r="1.4" />
    </>
  ),
  inventory: (
    <>
      <path d="M21 8.2 12 3.2 3 8.2v7.6l9 5 9-5Z" />
      <path d="M3 8.2l9 5 9-5" />
      <path d="M12 13.2v7.6" />
    </>
  ),
  models: (
    <>
      <path d="M12 2.8 2.8 7.4 12 12l9.2-4.6Z" />
      <path d="m2.8 12.4 9.2 4.6 9.2-4.6" />
      <path d="m2.8 16.9 9.2 4.6 9.2-4.6" />
    </>
  ),
  monitoring: (
    <>
      <polyline points="2.5 13 7 13 9.6 5.5 14 18.5 16.6 13 21.5 13" />
    </>
  ),
  menu: <path d="M4 6.5h16M4 12h16M4 17.5h16" />,
  close: <path d="M6 6l12 12M18 6 6 18" />,
  "chevron-left": <path d="m14.5 5.5-7 6.5 7 6.5" />,
  "chevron-right": <path d="m9.5 5.5 7 6.5-7 6.5" />,
  search: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4.5 4.5" />
    </>
  ),
};

export function Icon({
  name,
  size = 16,
  ...rest
}: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {PATHS[name]}
    </svg>
  );
}
