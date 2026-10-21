/**
 * Pure selection logic for the Model performance page.
 *
 * Extracted so the version used for the DETAIL FETCH and the version shown
 * in the <select> can never diverge again. Regression guard: the page once
 * displayed items[0] in the dropdown via a value-prop fallback while React
 * state stayed null, so the detail request never fired and the page showed
 * "No trained model found" despite a trained model existing.
 */

export interface VersionRef {
  version: string;
}

/**
 * Resolve the version that should be active.
 *
 * - An explicit selection wins, but only if it still exists in the list
 *   (otherwise we would fetch a version the backend no longer serves).
 * - Otherwise the first available version.
 * - Otherwise null, which keeps the "no trained model" branch honest.
 */
export function effectiveVersion(
  selected: string | null,
  items: VersionRef[] | null | undefined,
): string | null {
  const list = items ?? [];
  if (selected) {
    const known = list.find((item) => item.version === selected);
    if (known) {
      return known.version;
    }
  }
  return list.length > 0 ? list[0].version : null;
}
