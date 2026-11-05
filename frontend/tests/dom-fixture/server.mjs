// Isolated component fixture: original Vue component, synthetic state, no API proxy.
import {createServer} from 'vite'
import {fileURLToPath} from 'node:url'
import path from 'node:path'
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const fixture = path.join(root, 'tests/dom-fixture')
const server = await createServer({
  root, configFile: false, cacheDir: path.join(fixture, '.cache'), server: {host: '127.0.0.1', port: 5189, strictPort: true},
  plugins: [
    (await import('@vitejs/plugin-vue')).default(),
    {name: 'isolated-state', enforce: 'pre', resolveId(id, importer) {
      if (importer?.endsWith('/src/components/ui/RecordsDrawer.vue')) {
        if (id === '../../fitzen/state') return path.join(fixture, 'state.ts')
        if (id === '../../fitzen/metrics') return path.join(fixture, 'metrics.ts')
      }
    }},
    {name: 'deny-api', configureServer(s) {
      s.middlewares.use((req, res, next) => {
        if (req.url.startsWith('/api')) {res.statusCode = 403; res.end('fixture forbids real API'); return}
        next()
      })
    }}
  ]
})
await server.listen()
console.log('Isolated DOM fixture: http://127.0.0.1:5189/tests/dom-fixture/index.html')
