import basicSsl from "@vitejs/plugin-basic-ssl";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Backend lives at integration/api.py, run separately with uvicorn
// (see integration/README.md: `python -m uvicorn integration.api:app --reload`, port 8000).
// `npm run dev:phone` serves over HTTPS on the LAN. Phone browsers only allow the
// camera (photo and finger scan) on a secure context, and plain http://<LAN-IP> is not one.
export default defineConfig(({ mode }) => ({
  plugins: [react(), ...(mode === "phone" ? [basicSsl()] : [])],
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/predict": "http://127.0.0.1:8000",
      "/dishes": "http://127.0.0.1:8000",
      "/glucose": "http://127.0.0.1:8000",
      "/meal": "http://127.0.0.1:8000",
      "/contract": "http://127.0.0.1:8000",
      "/auth": "http://127.0.0.1:8000",
      "/user": "http://127.0.0.1:8000",
    },
  },
}));
