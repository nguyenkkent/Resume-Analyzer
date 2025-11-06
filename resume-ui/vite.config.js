import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/agent': {
        target: 'http://localhost:8100',
        changeOrigin: true,
      },
    },
  },
})
