import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import {isolatedState} from './isolated-state.mjs'
// Captured from the actual FastAPI JSON/SSE routes and production serializer.
const wire=JSON.parse(readFileSync(new URL('./chat-wire-contract.generated.json',import.meta.url),'utf8'))
const sse=(text,request)=>new Response(text.replaceAll(request.confirmation?'synthetic-confirmation':'synthetic-original',request.request_id),{headers:{'Content-Type':'text/event-stream'}})
function json(data,request){const body=structuredClone(data);if(body.request_id)body.request_id=request.request_id;return Response.json(body)}
test('real HTTP wire pending SSE → exact confirmation JSON → receipt and consumed target',async t=>{
 const state=await isolatedState(t),requests=[]
 globalThis.fetch=async(url,init)=>{
  if(url==='/api/health')return Response.json({ready:true,capabilities:{chat_record:{available:true,reason:'available'}}})
  if(url!=='/api/chat/stream')return Response.json({}, {status:503})
  const body=JSON.parse(init.body);requests.push(body)
  return body.confirmation?json(wire.completed,body):sse(wire.pending_sse,body)
 }
 state.ui.chatMode='react_write';state.ui.message='模拟午餐';await state.sendChat();const original=state.ui.messages[1]
 assert.equal(original.write.status,'pending');assert.equal(original.write.pending.length,1)
 await state.confirmPending(1,0)
 assert.equal(requests[1].confirmation.pending_request_id,requests[0].request_id)
 assert.deepEqual(Object.keys(requests[1].confirmation).sort(),['pending_digest','pending_op_id','pending_request_id'])
 assert.equal(state.ui.messages[3].write.status,'completed');assert.deepEqual(state.ui.messages[3].write.receipts,wire.completed.metrics.receipt_refs)
 assert.equal(state.pendingUnavailable(original,original.write.pending[0]),'已确认保存')
})
test('real confirmation privacy SSE clears old identities and disables all recovery',async t=>{
 const state=await isolatedState(t),requests=[]
 globalThis.fetch=async(url,init)=>{
  if(url==='/api/health')return Response.json({ready:true,capabilities:{chat_record:{available:true,reason:'available'}}})
  if(url!=='/api/chat/stream')return Response.json({}, {status:503})
  const body=JSON.parse(init.body);requests.push(body)
  return sse(body.confirmation?wire.confirmation_privacy_sse:wire.pending_sse,body)
 }
 state.ui.chatMode='react_write';state.ui.message='模拟午餐';await state.sendChat();const original=state.ui.messages[1]
 assert.ok(original.run_id);await state.confirmPending(1,0)
 assert.equal(original.write.blocked,true);assert.equal(original.run_id,undefined);assert.equal(state.ui.canRetryChat,false)
 assert.equal(state.pendingUnavailable(original,original.write.pending[0]),'隐私设置已变更')
 await state.retryChat();await state.resumeChat(3);await state.confirmPending(1,0);assert.equal(requests.length,2)
})
test('real closed-switch wire stays limited and dispatches no fallback request',async t=>{
 const state=await isolatedState(t),requests=[]
 globalThis.fetch=async(url,init)=>{
  if(url==='/api/health')return Response.json({ready:true,capabilities:{chat_record:{available:true,reason:'available'}}})
  if(url!=='/api/chat/stream')return Response.json({}, {status:503});const body=JSON.parse(init.body);requests.push(body);return json(wire.disabled,body)}
 state.ui.chatMode='react_write';state.ui.message='模拟记录';await state.sendChat()
 assert.equal(state.ui.messages[1].write.reason,'react_write_disabled');assert.equal(requests.length,1);assert.equal(requests[0].mode,'react_write')
})
