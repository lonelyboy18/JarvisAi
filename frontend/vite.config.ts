import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ws": {
        target: "ws://localhost:8756",
        ws: true,
      },
      "/api": {
        target: "http://localhost:8756",
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
