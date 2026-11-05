import {createServer} from 'vite'
import vue from '@vitejs/plugin-vue'
import {fileURLToPath} from 'node:url'
import path from 'node:path'
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..')
const evidence=path.resolve(root,'../evidence/chat-write-ui-20261013')
const server=await createServer({root,configFile:false,envDir:path.join(evidence,'empty-env'),cacheDir:path.join(evidence,'dom-cache'),
 server:{host:'127.0.0.1',port:5194,strictPort:true},plugins:[vue(),{name:'deny-real-api',configureServer(s){
  s.middlewares.use((req,res,next)=>{if(req.url.startsWith('/api')){res.statusCode=403;res.end('No real API in fixture');return}next()})}}]})
await server.listen()
console.log('Isolated chat fixture http://127.0.0.1:5194/tests/chat-write-dom/index.html')
