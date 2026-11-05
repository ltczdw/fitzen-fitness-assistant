import {createServer} from 'vite'
import vue from '@vitejs/plugin-vue'
import {fileURLToPath} from 'node:url'
import path from 'node:path'
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..')
const evidence=path.resolve(root,'../evidence/chat-write-e2e-20261013')
const server=await createServer({root,configFile:false,envDir:path.join(evidence,'empty-env'),cacheDir:path.join(evidence,'vite-cache'),plugins:[vue()],
 server:{host:'127.0.0.1',port:5196,strictPort:true,proxy:{'/api':{target:'http://127.0.0.1:8196'},'/__fixture__':{target:'http://127.0.0.1:8196'}}}})
await server.listen()
console.log('Isolated browser→API http://127.0.0.1:5196/tests/chat-write-e2e/index.html')
