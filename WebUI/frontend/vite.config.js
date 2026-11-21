import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built into WebUI/ui/, which Flask's static catch-all already serves at /ui/.
// Flask serves ui/index.html for /home and /map-interpreter (see map_interpreter_api.py).
export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  build: { outDir: "../ui", emptyOutDir: true },
  server: {
    port: 5173,
    // `npm run dev` talks to the running Flask app for the API and the existing pages' assets
    proxy: { "/api": "http://127.0.0.1:4041", "/icon": "http://127.0.0.1:4041" },
  },
});
