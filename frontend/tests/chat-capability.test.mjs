import {test} from 'node:test'
import assert from 'node:assert/strict'
import {isolatedState} from './isolated-state.mjs'

const health=(available,reason=available?'available':'react_write_disabled')=>Response.json({ready:true,capabilities:{chat_record:{available,reason}}})
for(const failure of ['disabled','old-server','network','invalid','database'])test('new record stays in draft when capability is '+failure,async t=>{
 const state=await isolatedState(t),sent=[]
 globalThis.fetch=async(url,init)=>{
  sent.push(url)
  assert.equal(url,'/api/health')
  if(failure==='network')throw new Error('offline')
  if(failure==='old-server')return Response.json({ready:true})
  if(failure==='invalid')return Response.json({ready:true,capabilities:{chat_record:{available:'true'}}})
  if(failure==='database')return Response.json({ready:false,capabilities:{chat_record:{available:true}}})
  return health(false)
 }
 state.ui.chatMode='react_write';state.ui.message='合成午餐'
 await state.sendChat()
 assert.deepEqual(sent,['/api/health']);assert.equal(state.ui.message,'合成午餐')
 assert.equal(state.ui.messages.length,0);assert.equal(state.ui.chatMode,'react_write')
 assert.equal(state.ui.chatRecordCapability.available,false);assert.ok(state.ui.notice)
})

test('availability changing between sends is checked again; no silent legacy request',async t=>{
 const state=await isolatedState(t),requests=[];let available=true
 globalThis.fetch=async(url,init)=>{
  if(url==='/api/health')return health(available)
  if(url==='/api/chat/stream'){requests.push(JSON.parse(init.body));return Response.json({answer:'完成',metrics:{execution_mode:'react_write',react_write:'completed',receipt_refs:['synthetic-receipt']}})}
  return Response.json({}, {status:503})
 }
 state.ui.chatMode='react_write';state.ui.message='第一餐';await state.sendChat()
 assert.equal(requests.length,1);assert.equal(requests[0].mode,'react_write')
 available=false;state.ui.message='第二餐';await state.sendChat()
 assert.equal(requests.length,1);assert.equal(state.ui.message,'第二餐')
 assert.equal(state.ui.messages.length,2);assert.equal(state.ui.chatMode,'react_write')
})

test('concurrent readiness checks are coalesced and draft changes cannot submit stale text',async t=>{
 const state=await isolatedState(t);let resolve,checks=0
 globalThis.fetch=async url=>{assert.equal(url,'/api/health');checks++;return new Promise(r=>resolve=r)}
 state.ui.chatMode='react_write';state.ui.message='旧草稿'
 const first=state.sendChat(),second=state.loadChatCapabilities()
 state.ui.message='新草稿';resolve(health(true));await Promise.all([first,second])
 assert.equal(checks,1);assert.equal(state.ui.messages.length,0);assert.equal(state.ui.message,'新草稿')
})

test('recovering an already sent request preserves identity even after capability closes',async t=>{
 const state=await isolatedState(t),sent=[];let checks=0
 globalThis.fetch=async(url,init)=>{
  if(url==='/api/health'){checks++;return health(true)}
  if(url==='/api/chat/stream'){sent.push(JSON.parse(init.body));return new Response('event: token\ndata: {"content":"部分回复"}\n\n',{headers:{'Content-Type':'text/event-stream'}})}
  return Response.json({}, {status:503})
 }
 state.ui.chatMode='react_write';state.ui.message='合成训练';await state.sendChat()
 state.ui.chatRecordCapability={checked:true,available:false,reason:'react_write_disabled'}
 await state.retryChat()
 assert.equal(checks,1);assert.equal(sent.length,2);assert.deepEqual(sent[0],sent[1])
})
