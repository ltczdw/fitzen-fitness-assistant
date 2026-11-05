import {createServer} from 'vite'
import vue from '@vitejs/plugin-vue'
import {fileURLToPath} from 'node:url'
import path from 'node:path'
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..')
const fixture=path.join(root,'tests/round4-dom'),evidence=path.resolve(root,'../evidence/quality-round4-20261004')
const server=await createServer({root,configFile:false,envDir:path.join(evidence,'empty-env'),cacheDir:path.join(evidence,'dom-cache'),
 server:{host:'127.0.0.1',port:5194,strictPort:true},plugins:[vue(),
 {name:'round4-synthetic-state',enforce:'pre',resolveId(id,importer){
  if((importer?.endsWith('/src/components/ui/NutritionEditModal.vue')&&id==='../../fitzen/state')||(importer?.endsWith('/src/components/ui/WorkoutSessionManager.vue')&&id==='../../fitzen/state'))return path.join(fixture,'state.ts')
 }},
 {name:'deny-api',configureServer(s){s.middlewares.use((req,res,next)=>{if(req.url.startsWith('/api')){res.statusCode=403;res.end('fixture forbids real API');return}next()})}}
 ]})
await server.listen();console.log('Round 4 fixture: http://127.0.0.1:5194/tests/round4-dom/index.html')
