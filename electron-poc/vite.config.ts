import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  base: './',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    // IMPORTANT:
    // Keep renderer output separate from electron-builder output dir.
    // electron-builder writes installers/archives into its own output folder (configured in package.json).
    // If we share the same folder, Vite will try to empty it and can fail on locked NSIS artifacts (EPERM).
    outDir: 'dist-renderer',
  },
  server: {
    port: 5173,
    strictPort: true,
    host: true,
  },
  optimizeDeps: {
    include: ['date-fns', 'wavesurfer.js'],
  },
})
