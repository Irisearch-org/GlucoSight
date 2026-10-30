import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Backend lives at integration/api.py, run separately with uvicorn
// (see integration/README.md: `python -m uvicorn integration.api:app --reload`, port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/predict": "http://127.0.0.1:8000",
      "/contract": "http://127.0.0.1:8000",
    },
  },
});
