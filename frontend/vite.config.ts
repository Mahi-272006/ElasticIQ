import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API calls to the FastAPI backend. Override the target with
// VITE_PROXY_TARGET when the backend runs elsewhere; the default preserves the
// existing local setup (http://127.0.0.1:8000).
const proxyTarget = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: proxyTarget, changeOrigin: true },
      "/health": { target: proxyTarget, changeOrigin: true },
    },
  },
});
