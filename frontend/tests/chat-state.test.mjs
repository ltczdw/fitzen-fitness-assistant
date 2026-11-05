import {test, beforeEach} from 'node:test'
import assert from 'node:assert/strict'
import {isolatedState} from './isolated-state.mjs'
let ui, sendChat, retryChat, deleteConversation, requests, outcome
beforeEach(async t => {
 ({ui,sendChat,retryChat,deleteConversation} = await isolatedState(t))
 requests=[]; outcome='disconnect'
 globalThis.fetch=async (url,init)=>{
 if(url==='/api/chat/stream'){
  requests.push(JSON.parse(init.body))
  if(outcome==='network')throw new Error('network offline')
  const body=outcome==='done'?'event: token\ndata: {"content":"ok"}\n\nevent: done\ndata: {}\n\n':'event: token\ndata: {"content":"partial"}\n\n'
  return new Response(body,{headers:{'Content-Type':'text/event-stream'}})
 }
 return new Response(JSON.stringify({detail:'offline'}),{status:503})
}
})

test('SSE/network retries retain ID; new edited message gets new ID',async()=>{
 ui.message='记录两个鸡蛋';await sendChat()
 assert.equal(ui.canRetryChat,true)
 assert.match(ui.chatError,/连接中断/)
 outcome='network';await retryChat()
 assert.equal(requests[0].request_id,requests[1].request_id)
 outcome='done';await retryChat()
 assert.equal(requests[0].request_id,requests[2].request_id)
 assert.equal(ui.messages.length,2)
 assert.equal(ui.messages[1].content,'ok')
 assert.equal(ui.canRetryChat,false)
 ui.message='记录三个鸡蛋';await sendChat()
 assert.notEqual(requests[2].request_id,requests[3].request_id)
 outcome='network';ui.message='再记录一个鸡蛋';await sendChat()
 ui.message='修改成两个鸡蛋';await retryChat()
 assert.notEqual(requests[4].request_id,requests[5].request_id)
 assert.equal(requests[5].message,'修改成两个鸡蛋')
})

test('default chat mode is react_readonly and needs no record-capability probe',async()=>{
 assert.equal(ui.chatMode,'react_readonly')
 ui.message='今天训练多久合适';await sendChat()
 assert.equal(requests.length,1)
 assert.equal(requests[0].mode,'react_readonly')
 assert.equal(ui.messages.length,2)
})

test('failed deletion preserves current conversation, messages and list',async()=>{
 ui.conversationId='synthetic-delete-failure'; ui.messages=[{role:'user',content:'synthetic'}]
 const before={id:ui.conversationId,messages:JSON.stringify(ui.messages)}
 ui.conversations=[{id:before.id}];ui.pendingDeleteConvId=before.id
 await deleteConversation(before.id)
 assert.equal(ui.conversationId,before.id)
 assert.equal(JSON.stringify(ui.messages),before.messages)
 assert.deepEqual(ui.conversations,[{id:before.id}])
 assert.equal(ui.pendingDeleteConvId,before.id)
 assert.match(ui.notice,/offline/)
})
