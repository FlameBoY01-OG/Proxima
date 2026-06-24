import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server runs on :5173 and talks to the FastAPI backend on :8000.
// CORS is enabled server-side, so we call the backend directly (see src/lib/api.ts).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
