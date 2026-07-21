import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

declare const process: { env: Record<string, string | undefined> }

const apiPort = process.env.CASHGAP_API_PORT ?? '8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': `http://127.0.0.1:${apiPort}`,
    },
  },
})
