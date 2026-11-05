import {createServer} from 'vite'
import vue from '@vitejs/plugin-vue'
import {fileURLToPath} from 'node:url'
import path from 'node:path'
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..')
const fixture=path.join(root,'tests/round2-dom'),evidence=path.resolve(root,'../evidence/quality-round2-20261003')
const server=await createServer({root,configFile:false,envDir:path.join(evidence,'empty-env'),cacheDir:path.join(evidence,'dom-cache'),
 server:{host:'127.0.0.1',port:5190,strictPort:true},plugins:[vue(),
 {name:'round2-synthetic-state',enforce:'pre',resolveId(id,importer){
  if((importer?.endsWith('/src/components/ui/NutritionEditModal.vue')&&id==='../../fitzen/state')||(importer?.endsWith('/src/components/ui/WorkoutSessionManager.vue')&&id==='../../fitzen/state'))return path.join(fixture,'state.ts')
 }},
 {name:'deny-api',configureServer(s){s.middlewares.use((req,res,next)=>{if(req.url.startsWith('/api')){res.statusCode=403;res.end('fixture forbids real API');return}next()})}}
 ]})
await server.listen();console.log('Round 2 fixture: http://127.0.0.1:5190/tests/round2-dom/index.html')
