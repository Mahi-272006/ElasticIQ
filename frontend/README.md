# ElasticIQ frontend

React 19 + TypeScript (strict) + Vite 7 + Recharts. Light theme only; design
tokens live in `src/styles.css`.

## Commands

```bash
npm install        # once
npm run dev        # dev server on http://localhost:5173 (proxies API to :8000)
npm test           # unit tests (node:test, no extra dependencies)
npm run typecheck  # tsc --noEmit
npm run build      # typecheck + production build -> dist/
npm run preview    # serve the production build locally
```

The backend must be running for data (`python -m uvicorn src.api.app:app`
from the repository root). In development, API calls are proxied by Vite.

## Environment variables

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `VITE_API_BASE_URL` | build time (`.env` or shell) | `""` (same origin) | Absolute base URL of the FastAPI backend, e.g. `https://api.example.com`. Leave empty when the API is served from the same origin as the frontend. |
| `VITE_PROXY_TARGET` | dev server (shell) | `http://127.0.0.1:8000` | Backend target for the Vite dev proxy. |

Only `VITE_`-prefixed variables are exposed to the client bundle — never put
secrets in them.

Example `.env.local` (git-ignored by default by Vite for `.local` files):

```
VITE_API_BASE_URL=https://api.example.com
```

## Deployment notes

- Build command: `npm run build` (runs `tsc --noEmit` first). Output: `dist/`.
- Routing is client-side (`BrowserRouter`). The host must serve
  `index.html` for unknown paths (SPA fallback):
  - Netlify: `public/_redirects` is already included in the build output.
  - Vercel: add a `vercel.json` rewrite `{"rewrites": [{"source": "/(.*)", "destination": "/index.html"}]}`.
  - Other hosts: configure an equivalent rewrite.
- When the API is on a different origin, set `VITE_API_BASE_URL` **and**
  configure CORS on the backend for the frontend origin.
- The app calls `/health` and `/api/v1/*`; both follow `VITE_API_BASE_URL`.
