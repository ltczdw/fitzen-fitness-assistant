import {test, beforeEach} from 'node:test'
import assert from 'node:assert/strict'
import {isolatedState} from './isolated-state.mjs'
let ui, rateAnswer, loadHistory, switchConversation
beforeEach(async t => {
 ({ui,rateAnswer,loadHistory,switchConversation} = await isolatedState(t))
})

test('feedback binds persisted answer, prevents duplicate clicks and can retry network failure',async()=>{
 let calls=0,finish
 globalThis.fetch=async(url,init)=>{
  calls++;assert.equal(url,'/api/feedback')
  assert.deepEqual(JSON.parse(init.body),{run_id:'run-one',message_id:'message-one',rating:'down'})
  await new Promise(resolve=>{finish=resolve})
  return new Response('{}')
 }
 ui.messages=[{role:'assistant',content:'answer',run_id:'run-one',message_id:'message-one'}]
 const pending=rateAnswer(0,'down')
 await rateAnswer(0,'down');assert.equal(calls,1)
 finish();await pending;assert.equal(ui.messages[0].feedback,'down')
 await rateAnswer(0,'up');assert.equal(calls,1)
 ui.messages=[{role:'assistant',content:'answer',run_id:'run-one',message_id:'message-one'}]
 globalThis.fetch=async()=>{throw new Error('offline')}
 await rateAnswer(0,'down');assert.equal(ui.messages[0].feedback,undefined);assert.equal(ui.messages[0].feedbackPending,false)
 globalThis.fetch=async()=>new Response('{}')
 await rateAnswer(0,'down');assert.equal(ui.messages[0].feedback,'down')
})

test('an answer without persisted identifiers cannot submit feedback',async()=>{
 ui.messages=[{role:'assistant',content:'answer'}]
 globalThis.fetch=()=>{throw new Error('must not send')}
 await rateAnswer(0,'down');assert.equal(ui.messages[0].feedback,undefined)
})

test('history restores run id, citations and feedback and blocks a second rating',async()=>{
 globalThis.fetch=async(url)=>{
  assert.match(String(url),/\/api\/chat\/history\?/)
  return new Response(JSON.stringify({messages:[
   {role:'user',content:'问题'},
   {role:'assistant',content:'回答一',run_id:'run-history',message_id:'mid-history',citations:[{title:'训练指南',source:'knowledge'}]},
   {role:'assistant',content:'回答二',run_id:'run-rated',message_id:'mid-rated',feedback:'down'}
  ]}))
 }
 await loadHistory()
 assert.equal(ui.messages[1].run_id,'run-history')
 assert.equal(ui.messages[1].citations[0].title,'训练指南')
 assert.equal(ui.messages[2].feedback,'down')
 let calls=0
 globalThis.fetch=async()=>{calls++;return new Response('{}')}
 // Already rated history stays rated; an unrated historical answer can submit.
 await rateAnswer(2,'up');assert.equal(calls,0)
 await rateAnswer(1,'up');assert.equal(calls,1)
 assert.equal(ui.messages[1].feedback,'up')
})

test('history without diagnostics stays visible and offers no feedback id',async()=>{
 globalThis.fetch=async()=>new Response(JSON.stringify({messages:[
  {role:'user',content:'旧问题'},
  {role:'assistant',content:'旧回答',message_id:'mid-plain'}
 ]}))
 await loadHistory()
 assert.equal(ui.messages[1].content,'旧回答')
 assert.equal(ui.messages[1].run_id,undefined)
 let calls=0
 globalThis.fetch=async()=>{calls++;return new Response('{}')}
 await rateAnswer(1,'up');assert.equal(calls,0)
})

test('switching conversations reloads the restored feedback identity',async()=>{
 globalThis.fetch=async(url)=>{
  if(String(url).includes('/chat/history'))return new Response(JSON.stringify({messages:[
   {role:'assistant',content:'另一会话回答',run_id:'run-switch',message_id:'mid-switch'}
  ]}))
  return new Response('{}')
 }
 ui.sending=false;ui.conversationId='conv-current';ui.messages=[]
 await switchConversation('conv-other')
 assert.equal(ui.conversationId,'conv-other')
 assert.equal(ui.messages.at(-1).run_id,'run-switch')
})
