import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// FastAPI backend (cctv_dfd/api/server.py) listens on :8000 by default.
// In dev (`npm run dev`) Vite serves React on :3000 and proxies /predict
// to the backend so the browser doesn't need CORS configured.
// To talk to a different backend, edit .env (VITE_API_URL).

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
