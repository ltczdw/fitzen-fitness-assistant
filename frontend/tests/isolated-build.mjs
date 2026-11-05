// Production source build with existing dependencies. No real .env/proxy/config
// loading, and cache/output stay inside this execution copy.
import {build} from 'vite'
import vue from '@vitejs/plugin-vue'
import {fileURLToPath} from 'node:url'
import path from 'node:path'
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const evidence = path.resolve(root, '..', process.argv[2] || 'evidence/quality-acceptance-20261003')
if (!evidence.startsWith(path.resolve(root,'..')+path.sep)) throw new Error('evidence outside workspace')
const output = path.join(evidence, process.argv[3] || 'frontend-dist')
if (!output.startsWith(path.resolve(root, '..') + path.sep)) throw new Error('output outside workspace')
process.chdir(root)
await build({root, configFile:false, envDir:path.join(evidence,'empty-env'),
  cacheDir:path.join(evidence,'vite-cache'), plugins:[vue()],
  build:{outDir:output, emptyOutDir:false, rollupOptions:{input:{main:path.join(root,'index.html')}}}})
