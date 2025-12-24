import esbuild from 'esbuild'

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

console.log('✅ Electron scripts built successfully')
