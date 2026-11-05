import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
  plugins: [vue()],
  build: { rollupOptions: { input: { main: 'index.html' } } },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': process.env.VITE_API_TARGET || env.VITE_API_TARGET || 'http://localhost:8001',
    },
  },
  }
})
