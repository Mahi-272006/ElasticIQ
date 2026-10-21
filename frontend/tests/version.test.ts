/**
 * Regression tests for the Model performance page selection bug.
 *
 * Run with: npm test  (node --test, no extra dependencies)
 * Kept outside src/ so the app's tsc pass doesn't require @types/node.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { effectiveVersion } from "../src/lib/version.ts";

const ITEMS = [{ version: "hgb_poisson_v1" }, { version: "older_v0" }];

test("defaults to the first available version when nothing is selected", () => {
  // The original bug: selected stayed null while the dropdown showed items[0],
  // so the detail fetch never ran and the page said "No trained model found".
  assert.equal(effectiveVersion(null, ITEMS), "hgb_poisson_v1");
});

test("an explicit selection always wins over the first item", () => {
  assert.equal(effectiveVersion("older_v0", ITEMS), "older_v0");
});

test("empty string selection falls back to the first item", () => {
  assert.equal(effectiveVersion("", ITEMS), "hgb_poisson_v1");
});

test("null when there are no versions (message branch is legitimate)", () => {
  assert.equal(effectiveVersion(null, []), null);
  assert.equal(effectiveVersion(null, null), null);
  assert.equal(effectiveVersion(null, undefined), null);
});

test("a stale selection is not fetched; falls back safely", () => {
  // Unknown version with an empty list: null, so no 404 fetch is attempted
  // and the legitimate "no trained model" branch renders.
  assert.equal(effectiveVersion("ghost_v9", []), null);
  // Unknown version when other versions exist: fall back to the first one
  // instead of requesting a version the backend no longer serves.
  assert.equal(effectiveVersion("ghost_v9", ITEMS), "hgb_poisson_v1");
});
