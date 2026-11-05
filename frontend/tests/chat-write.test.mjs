import {test,beforeEach} from 'node:test'
import assert from 'node:assert/strict'
import {isolatedState} from './isolated-state.mjs'
let state,requests,outcome,transport,expiry,requestGate
const op='op2_'+'a'.repeat(32),digest='b'.repeat(64)
const receipt='nutrition:c2-2:'+op+':g0:17'
const target=id=>({pending_request_id:id,pending_op_id:op,pending_digest:digest,expires_at:expiry})
const response=body=>({answer:'模拟结果',request_id:body.request_id,diagnostics_status:'unavailable',metrics:{execution_mode:'react_write',react_write:outcome,receipt_refs:outcome==='completed'||outcome==='commit_unknown'?[receipt]:[],pending_confirmations:outcome==='pending'?[target(body.request_id)]:[]}})
beforeEach(async t=>{
 state=await isolatedState(t);requests=[];outcome='pending';transport='sse';expiry=new Date(Date.now()+900000).toISOString();requestGate=null
 globalThis.fetch=async(url,init)=>{
  if(url==='/api/health')return Response.json({ready:true,capabilities:{chat_record:{available:true,reason:'available'}}})
  if(url!=='/api/chat/stream')return Response.json({detail:'isolated no data'},{status:503})
  const body=JSON.parse(init.body);requests.push(body)
  if(requestGate)await requestGate
  if(transport==='network')throw new Error('fixture offline')
  if(transport==='409'||transport==='422')return Response.json({detail:'synthetic refusal'},{status:Number(transport)})
  const data=response(body)
  if(transport==='privacy')data.metrics={react_write:'limited',mode_reason:'privacy_invalidated',receipt_refs:[],pending_confirmations:[]}
  if(transport==='disabled')data.metrics={execution_mode:'limited',mode_reason:'react_write_disabled'}
  if(transport==='bad-target')data.metrics.pending_confirmations=[{...target(body.request_id),pending_request_id:'other'}]
  if(transport==='bad-id')data.request_id='other'
  if(transport==='json'||transport==='privacy'||transport==='disabled'||transport==='bad-target'||transport==='bad-id')return Response.json(data)
  if(transport==='conflict')return new Response('event: error\ndata: {"message":"conflict","code":"request_id_conflict"}\n\nevent: done\ndata: {"status":"failed"}\n\n',{headers:{'Content-Type':'text/event-stream'}})
  if(transport==='failed')return new Response(`event: meta\ndata: ${JSON.stringify(data)}\n\nevent: error\ndata: {"message":"failed","code":"execution_failed"}\n\nevent: done\ndata: {"status":"failed"}\n\n`,{headers:{'Content-Type':'text/event-stream'}})
  if(transport==='abort')return new Response(new ReadableStream({start(c){c.enqueue(new TextEncoder().encode(`event: meta\ndata: ${JSON.stringify(data)}\n\n`));init.signal.addEventListener('abort',()=>c.error(new DOMException('aborted','AbortError')))}}),{headers:{'Content-Type':'text/event-stream'}})
  const {answer,...meta}=data
  return new Response(`event: meta\ndata: ${JSON.stringify(meta)}\n\nevent: token\ndata: ${JSON.stringify({content:answer})}\n\n${transport==='disconnect'?'':`event: done\ndata: ${JSON.stringify(meta)}\n\n`}`,{headers:{'Content-Type':'text/event-stream'}})
 }
})
async function send(){state.ui.chatMode='react_write';state.ui.message='模拟午餐：两个鸡蛋';await state.sendChat();return state.ui.messages[1]}
test('explicit mode, exact selector, fresh confirmation ID; completed receipt consumes target',async()=>{
 const original=await send();assert.equal(requests[0].mode,'react_write');assert.equal(original.write.status,'pending');assert.equal(original.write.settled,true)
 outcome='completed';await state.confirmPending(1,0)
 assert.deepEqual(requests[1].confirmation,{pending_request_id:requests[0].request_id,pending_op_id:op,pending_digest:digest})
 assert.equal(requests[1].message,'确认保存');assert.notEqual(requests[1].request_id,requests[0].request_id)
 assert.deepEqual(state.ui.messages[3].write.receipts,[receipt]);assert.equal(state.ui.messages[3].run_id,undefined)
 assert.equal(state.pendingUnavailable(original,original.write.pending[0]),'已确认保存')
 await state.confirmPending(1,0);assert.equal(requests.length,2)
})
test('confirmation double click and failed confirmation preserve one payload/ID across mode changes',async()=>{
 const original=await send();transport='network'
 const first=state.confirmPending(1,0);await state.confirmPending(1,0);await first
 assert.equal(requests.length,2);assert.equal(state.pendingUnavailable(original,original.write.pending[0]),'确认结果待恢复')
 await state.confirmPending(1,0);assert.equal(requests.length,2)
 state.ui.chatMode='legacy';outcome='completed';transport='json';await state.retryChat()
 assert.deepEqual(requests[2],requests[1]);assert.equal(state.ui.messages.length,4);assert.equal(state.ui.canRetryChat,false)
})
test('disconnect blocks premature confirmation; retry restores original pending safely',async()=>{
 transport='disconnect';const original=await send()
 assert.equal(original.write.settled,false);await state.confirmPending(1,0);assert.equal(requests.length,1)
 transport='sse';state.ui.chatMode='legacy';await state.retryChat()
 assert.deepEqual(requests[1],requests[0]);assert.equal(original.write.settled,true)
})
test('JSON and SSE produce equivalent typed write result',async()=>{
 outcome='commit_unknown';const original=await send();const before=structuredClone(original.write)
 transport='json';await state.resumeChat(1);assert.deepEqual(original.write,before);assert.deepEqual(requests[1],requests[0])
 assert.equal(original.write.status,'commit_unknown');assert.deepEqual(original.write.receipts,[receipt])
})
test('stop preserves known receipt and retries same ID, never reports rollback',async()=>{
 transport='abort';outcome='commit_unknown'
 state.ui.chatMode='react_write';state.ui.message='模拟训练'
 const flight=state.sendChat()
 // Let the stream meta reach the real decoder before stopping.
 for(let i=0;i<100&&!state.ui.messages[1]?.write?.receipts.length;i++)await Promise.resolve()
 assert.deepEqual(state.ui.messages[1].write.receipts,[receipt]);state.stopChat();await flight
 assert.match(state.ui.chatError,/保存可能已执行/);assert.equal(state.ui.messages[1].write.settled,false)
 transport='json';outcome='completed';await state.retryChat();assert.deepEqual(requests[1],requests[0])
})
test('expired selector and malformed source target never dispatch confirmation',async()=>{
 expiry=new Date(Date.now()-1000).toISOString();const original=await send();await state.confirmPending(1,0)
 assert.equal(requests.length,1);assert.match(state.pendingUnavailable(original,original.write.pending[0]),/过期/)
 transport='bad-target';await state.resumeChat(1);assert.equal(original.write.pending.length,0);await state.confirmPending(1,0);assert.equal(requests.length,2)
})
for(const refusal of ['409','422','conflict','privacy'])test(refusal+' invalidates previous targets and stops recovery',async()=>{
 const original=await send();original.run_id='synthetic-private-run';original.message_id='synthetic-private-answer';original.citations=[{title:'synthetic',source:'synthetic'}];transport=refusal;await state.confirmPending(1,0)
 assert.equal(state.ui.canRetryChat,false);assert.equal(original.write.blocked,true)
 await state.confirmPending(1,0);await state.resumeChat(1);await state.retryChat();assert.equal(requests.length,2)
 if(refusal==='privacy'){assert.deepEqual(original.write.receipts,[]);assert.deepEqual(original.citations,[]);assert.equal(original.run_id,undefined);assert.equal(original.message_id,undefined);assert.equal(state.pendingUnavailable(original,original.write.pending[0]),'隐私设置已变更')}
})
test('disabled feature is shown as limited with no automatic legacy fallback',async()=>{
 transport='disabled';const original=await send();assert.equal(original.write.status,'limited');assert.equal(original.write.reason,'react_write_disabled');assert.equal(requests.length,1);assert.equal(requests[0].mode,'react_write');assert.equal(state.ui.canRetryChat,false)
})
test('SSE execution failure retains original retry and disables selectors',async()=>{
 transport='failed';const original=await send();assert.equal(state.ui.canRetryChat,true);assert.equal(original.write.settled,false)
 await state.confirmPending(1,0);assert.equal(requests.length,1)
})
test('completed text alone cannot create a receipt and mismatched response ID cannot supply one',async()=>{
 transport='json';outcome='unrecognized';const original=await send();assert.deepEqual(original.write.receipts,[])
 transport='bad-id';outcome='completed';await state.resumeChat(1);assert.deepEqual(original.write.receipts,[]);assert.equal(original.write.settled,false);assert.match(state.ui.chatError,/标识不匹配/)
})
test('conversation change discards actionable state; stale history cannot replace new send',async()=>{
 const old=await send();state.ui.conversationId='new-conversation';await state.confirmPending(1,0);await state.resumeChat(1);assert.equal(requests.length,1)
 let resolve;globalThis.fetch=async()=>new Promise(r=>resolve=r)
 state.ui.messages=[];const history=state.loadHistory();state.ui.messages.push({role:'assistant',content:'current',request:{}})
 resolve(Response.json({messages:[{role:'assistant',content:'old'}]}));await history;assert.equal(state.ui.messages[0].content,'current')
 assert.match(state.pendingUnavailable(old,old.write.pending[0]),/会话已变更/)
})
test('stale history failure cannot overwrite an active message status',async()=>{
 let reject;globalThis.fetch=async()=>new Promise((_,r)=>reject=r)
 const history=state.loadHistory();state.ui.messages.push({role:'assistant',content:'current',request:{}});state.ui.chatError='current request status'
 reject(new Error('synthetic old history failure'));await history;assert.equal(state.ui.chatError,'current request status')
})
test('successful deletion removes pending confirmations and original recovery request',async()=>{
 await send();const id=state.ui.conversationId
 globalThis.fetch=async(url)=>url.includes('/chat/conversations/')?Response.json({deleted:true}):Response.json({conversations:[]})
 await state.deleteConversation(id);assert.notEqual(state.ui.conversationId,id);assert.deepEqual(state.ui.messages,[])
 await state.confirmPending(1,0);await state.resumeChat(1);await state.retryChat();assert.equal(requests.length,1)
})
