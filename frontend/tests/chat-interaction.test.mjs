import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import vm from 'node:vm'
import ts from 'typescript'
import {parse} from 'vue/compiler-sfc'
import {ref,reactive,computed,watch,nextTick,effectScope,markRaw} from 'vue'
import {isolatedState} from './isolated-state.mjs'
const compile=source=>ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
const helper=compile(readFileSync(new URL('../src/fitzen/chat-interaction.ts',import.meta.url),'utf8'))
const {createCompositionGuard,createChatScrollPolicy}=await import('data:text/javascript;base64,'+Buffer.from(helper).toString('base64'))
function page(t){
 let sends=0,top=700,height=1000,writes=0
 const compositionTasks=[]
 const ui=reactive({message:'中文消息',sending:false,conversationId:'one',messages:[],sessions:[],workouts:[],conversations:[],chatRecordCapability:{available:true},data:{}})
 const useFitzen=()=>({ui,sendChat:async()=>{sends++},stopChat(){},openFood(){},openWorkout(){},openRecords(){},exportChat(){}})
 const cleanup=[]
 const context={ref,reactive,computed,watch,nextTick,useFitzen,loadChatCapabilities(){},createCompositionGuard:()=>createCompositionGuard(done=>{compositionTasks.push(done);return 0}),createChatScrollPolicy,onMounted(){},onBeforeUnmount:fn=>cleanup.push(fn),window:{matchMedia:()=>({matches:false}),removeEventListener(){},dispatchEvent(){}},document:{removeEventListener(){}},CustomEvent:class{},getComputedStyle:()=>({lineHeight:'20px'})}
 const source=parse(readFileSync(new URL('../src/fitzen/screens/Chat.vue',import.meta.url),'utf8')).descriptor.scriptSetup.content
 const scope=effectScope();t.after(()=>{scope.stop()})
 scope.run(()=>vm.runInNewContext(compile(source).replace(/^import .*$/gm,'')+'\nglobalThis.page={inputKey,composition,sendMessage,chatStream,chatContent,composerWrap,latestBottom,trackScroll,pauseFollow,scrollLatest,followContent,showLatest};',context))
 const p=context.page
 p.chatStream.value=markRaw({get scrollTop(){return top},set scrollTop(v){writes++;top=Math.min(v,height-300)},get scrollHeight(){return height},clientHeight:300})
 p.chatContent.value=markRaw({querySelector:()=>({})})
 return {...p,ui,unmount(){cleanup.forEach(fn=>fn());scope.stop()},flushComposition(){compositionTasks.splice(0).forEach(done=>done())},get sends(){return sends},get top(){return top},set top(value){top=value},get writes(){return writes},set height(value){height=value}}
}
function key(p,patch={}){const e={key:'Enter',shiftKey:false,isComposing:false,keyCode:13,repeat:false,prevented:false,stopped:false,preventDefault(){this.prevented=true},stopPropagation(){this.stopped=true},...patch};p.inputKey(e);return e}

test('candidate Enter and Safari commit event never send; first distinct Enter sends immediately',t=>{
 const p=page(t)
 p.composition.start();assert.equal(key(p).prevented,false);assert.equal(p.sends,0)
 p.composition.end();assert.equal(key(p).prevented,true);assert.equal(p.sends,0)
 p.composition.keyup({key:'Enter'})
 key(p);assert.equal(p.sends,1)
})
test('space or pointer candidate selection needs no extra Enter or 500ms wait',t=>{
 const p=page(t);p.composition.start();p.composition.end();p.flushComposition()
 key(p,{keyCode:229});key(p,{isComposing:true});assert.equal(p.sends,0)
 key(p);assert.equal(p.sends,1)
})
test('composition guards slash commands; release allows first Enter selection',t=>{
 const p=page(t);p.ui.message='/train';p.composition.start();key(p);assert.equal(p.ui.message,'/train')
 p.composition.end();key(p);assert.equal(p.ui.message,'/train')
 p.composition.keyup({key:'Enter'});key(p);assert.notEqual(p.ui.message,'/train');assert.equal(p.sends,0)
})
test('blur resets composition, repeated Enter never sends, and ShiftEnter remains newline',t=>{
 const p=page(t);p.composition.start();p.composition.cancel()
 assert.equal(key(p,{shiftKey:true}).prevented,false)
 key(p,{repeat:true});assert.equal(p.sends,0)
 key(p);assert.equal(p.sends,1)
})
test('explicit Send remains usable during the commit event',async t=>{
 const p=page(t);p.composition.start();p.composition.end();await p.sendMessage();assert.equal(p.sends,1)
})
test('composition handlers are bound and reply actions contain no audit entry',()=>{
 const chat=readFileSync(new URL('../src/fitzen/screens/Chat.vue',import.meta.url),'utf8')
 assert.ok(chat.includes('@keyup="composition.keyup"'))
 const replies=readFileSync(new URL('../src/components/ui/ChatMessages.vue',import.meta.url),'utf8')
 assert.ok(!replies.includes('质量与审核'));assert.ok(!replies.includes('href="/api/quality"'))
 assert.ok(replies.includes('copyReply(index)'));assert.ok(replies.includes("rateAnswer(index,'up')"))
})
test('one and three actual line thresholds govern resuming and latest button',()=>{
 const policy=createChatScrollPolicy(),metrics={scrollHeight:1000,clientHeight:300,scrollTop:700}
 policy.followed(metrics);policy.pause(metrics)
 metrics.scrollTop=690;policy.track(metrics,20);assert.equal(policy.following,false)
 metrics.scrollTop=691;policy.track(metrics,20);assert.equal(policy.following,true)
 metrics.scrollTop=680;policy.track(metrics,20);assert.equal(policy.following,false)
 metrics.scrollTop=640;assert.equal(policy.showLatest(metrics,20),false)
 metrics.scrollTop=639;assert.equal(policy.showLatest(metrics,20),true)
})
test('wheel pause cancels queued scroll; appended streaming messages never reset reading position',async t=>{
 const p=page(t);p.scrollLatest();p.pauseFollow();await nextTick();assert.equal(p.writes,0)
 p.top=500;p.trackScroll();p.height=1400;p.ui.messages.push({role:'assistant',content:'新增回复'});await nextTick();await nextTick()
 assert.equal(p.top,500);assert.equal(p.writes,0);assert.equal(p.showLatest.value,true)
 p.ui.messages[0].content+='继续输出';await nextTick();assert.equal(p.top,500)
 p.scrollLatest();await nextTick();assert.equal(p.top,1100);assert.equal(p.showLatest.value,false)
 p.height=1500;p.ui.messages[0].content+='自动跟随';await nextTick();await nextTick();assert.equal(p.top,1200)
})
test('switching conversation restores initial follow',async t=>{
 const p=page(t);p.pauseFollow();p.top=100;p.ui.conversationId='two';await nextTick();await nextTick();assert.equal(p.top,700)
})
test('unmount invalidates queued scrolling',async t=>{
 const p=page(t);p.scrollLatest();p.unmount();await nextTick();assert.equal(p.writes,0)
})
test('latest button stays above the resized multiline composer',t=>{
 const p=page(t);p.composerWrap.value=markRaw({offsetHeight:270});p.pauseFollow();assert.equal(p.latestBottom.value,282)
})
for(const transport of ['json','sse'])test(`${transport} real state transport does not write scrollTop`,async t=>{
 const state=await isolatedState(t);let writes=0
 globalThis.document.querySelector=()=>({scrollHeight:1000,set scrollTop(v){writes++}})
 globalThis.fetch=async(url,init)=>{
  if(url!=='/api/chat/stream')return Response.json({detail:'isolated'},{status:503})
  const request=JSON.parse(init.body),data={request_id:request.request_id,answer:'模拟流式文本',metrics:{}}
  return transport==='json'?Response.json(data):new Response(`event: token\ndata: {"content":"模拟流式文本"}\n\nevent: done\ndata: ${JSON.stringify(data)}\n\n`,{headers:{'Content-Type':'text/event-stream'}})
 }
 state.ui.message='隔离问题';await state.sendChat();assert.equal(writes,0);assert.equal(state.ui.messages[1].content,'模拟流式文本')
})
