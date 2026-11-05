import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import ts from 'typescript'
import {isolatedState} from './isolated-state.mjs'

const source=readFileSync(new URL('../src/fitzen/recent-conversations.ts',import.meta.url),'utf8')
const js=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ES2022}}).outputText
const {recentConversations}=await import('data:text/javascript;base64,'+Buffer.from(js).toString('base64'))
const settle=()=>new Promise(setImmediate)

test('empty new chats are hidden; first user message immediately supplies one current row',()=>{
 const saved=[{conversation_id:'saved',title:'旧会话'}]
 assert.deepEqual(recentConversations(saved,'current',[]),saved)
 const messages=[{role:'user',content:'第一条测试消息'},{role:'assistant',content:''}]
 const list=recentConversations(saved,'current',messages)
 assert.equal(list[0].conversation_id,'current')
 assert.equal(list[0].title,'第一条测试消息')
 assert.equal(saved.length,1)
 assert.equal(recentConversations(saved,'next-empty',[]).length,1)
 assert.equal(recentConversations(saved,'current',messages,'测试').length,1)
 assert.equal(recentConversations(saved,'current',messages,'不存在').length,0)
})

test('current row deduplicates server entry, preserves canonical title and obeys list limit',()=>{
 const saved=Array.from({length:35},(_,i)=>({conversation_id:'c'+i,title:'标题'+i}))
 const list=recentConversations(saved,'c8',[{role:'user',content:'本地消息'}])
 assert.equal(list.length,30)
 assert.equal(list[0].title,'标题8')
 assert.equal(list.filter(item=>item.conversation_id==='c8').length,1)
 assert.equal(saved[0].conversation_id,'c0')
})

for(const protocol of ['SSE','JSON'])test(protocol+' completion refreshes recent conversations without a second New Chat',async t=>{
 const f=await isolatedState(t);let release,serverRows=[],listReads=0
 globalThis.fetch=async(url,init)=>{
  if(url==='/api/chat/stream')return await new Promise(resolve=>{release=()=>{
   serverRows=[{conversation_id:f.ui.conversationId,title:'服务端标题'}]
   resolve(protocol==='SSE'?new Response('event: token\ndata: {"content":"ok"}\n\nevent: done\ndata: {}\n\n',{headers:{'Content-Type':'text/event-stream'}}):Response.json({answer:'ok'}))
  }})
  if(url.includes('/chat/conversations')){listReads++;return Response.json({conversations:serverRows})}
  return Response.json({}, {status:503})
 }
 await f.newChat();await settle()
 assert.equal(f.ui.conversations.length,0)
 const id=f.ui.conversationId
 f.ui.message='首条测试消息';const sending=f.sendChat()
 assert.equal(f.ui.sending,true)
 assert.equal(recentConversations(f.ui.conversations,id,f.ui.messages)[0].title,'首条测试消息')
 release();await sending;await settle()
 assert.equal(f.ui.conversationId,id)
 assert.equal(f.ui.conversations[0].conversation_id,id)
 assert.equal(f.ui.conversations[0].title,'服务端标题')
 assert.equal(listReads,2)
 assert.equal(recentConversations(f.ui.conversations,id,f.ui.messages).length,1)
})

test('interrupted stream still refreshes persisted conversation while keeping retry state',async t=>{
 const f=await isolatedState(t);let reads=0
 globalThis.fetch=async(url)=>{
  if(url==='/api/chat/stream')return new Response('event: token\ndata: {"content":"partial"}\n\n',{headers:{'Content-Type':'text/event-stream'}})
  if(url.includes('/chat/conversations')){reads++;return Response.json({conversations:[{conversation_id:f.ui.conversationId,title:'已持久化的部分会话'}]})}
  return Response.json({}, {status:503})
 }
 f.ui.message='模拟断流';await f.sendChat();await settle()
 assert.equal(reads,1)
 assert.equal(f.ui.canRetryChat,true)
 assert.equal(f.ui.conversations[0].title,'已持久化的部分会话')
})

test('newer list request wins; network and invalid list payloads preserve existing sessions',async t=>{
 const f=await isolatedState(t);let release
 globalThis.fetch=async()=>await new Promise(resolve=>{release=resolve})
 const old=f.loadConversations()
 globalThis.fetch=async()=>Response.json({conversations:[{conversation_id:'newer',title:'新列表'}]})
 await f.loadConversations()
 release(Response.json({conversations:[{conversation_id:'stale',title:'过期列表'}]}));await old
 assert.equal(f.ui.conversations[0].conversation_id,'newer')
 globalThis.fetch=async()=>{throw Error('offline')};await f.loadConversations()
 assert.equal(f.ui.conversations[0].conversation_id,'newer')
 globalThis.fetch=async()=>Response.json({});await f.loadConversations()
 assert.equal(f.ui.conversations[0].conversation_id,'newer')
})

test('successful deletion stays removed if the subsequent list reload fails',async t=>{
 const f=await isolatedState(t)
 f.ui.conversations=[{conversation_id:'remove',title:'删除项'},{conversation_id:'keep',title:'保留项'}]
 globalThis.fetch=async(url,init)=>init?.method==='DELETE'?Response.json({deleted:true}):Response.json({}, {status:503})
 await f.deleteConversation('remove')
 assert.deepEqual(f.ui.conversations.map(item=>item.conversation_id),['keep'])
})
