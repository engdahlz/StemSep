import esbuild from 'esbuild'
import fs from 'fs'
import path from 'path'
import { createRequire } from 'module'

const require = createRequire(import.meta.url)

// Build main process
await esbuild.build({
  entryPoints: ['electron/main.ts'],
  bundle: true,
  platform: 'node',
  target: 'node18',
  external: ['electron'],
  format: 'cjs',
  outfile: 'dist-electron/main.js',
})

// Build preload script
await esbuild.build({
  entryPoints: ['electron/preload.ts'],
  bundle: true,
  platform: 'node',
  target: 'node18',
  outfile: 'dist-electron/preload.cjs',
  external: ['electron'],
  format: 'cjs',
})

// Ensure a runnable ffmpeg binary exists alongside the bundled Electron main.
// NOTE: `ffmpeg-static` computes its binary path relative to __dirname; when we bundle
// it into dist-electron/main.js, that becomes dist-electron/. Copying the binary here
// avoids runtime ENOENT for non-WAV decoding.
try {
  const ffmpegStaticPath = require('ffmpeg-static')
  if (typeof ffmpegStaticPath === 'string' && ffmpegStaticPath && fs.existsSync(ffmpegStaticPath)) {
    const outDir = path.resolve('dist-electron')
    fs.mkdirSync(outDir, { recursive: true })

    const outName = process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg'
    const outPath = path.join(outDir, outName)
    fs.copyFileSync(ffmpegStaticPath, outPath)
  }
} catch {
  // Optional dependency; decoding will fall back to `ffmpeg` on PATH.
}

console.log('✅ Electron scripts built successfully')
