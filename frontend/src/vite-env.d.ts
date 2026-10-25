/// <reference types="vite/client" />

/** Build-time environment variables (VITE_* are exposed to the client). */
interface ImportMetaEnv {
  /** Absolute or relative base URL of the FastAPI backend. Default: same origin. */
  readonly VITE_API_BASE_URL?: string;
}
