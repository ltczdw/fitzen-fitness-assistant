import {createApp,h,ref} from 'vue'
import App from '../../src/fitzen/App.vue'
import {ui} from '../../src/fitzen/state'
import actualWire from '../chat-wire-contract.generated.json'
import '../../src/fitzen/fitzen.css'
const requests:any[]=[]
const count=ref(0),lastRequest=ref<any>(null)
const confirmed=new Set<string>()
let nextOutcome='pending'
const pendingOp='op2_'+'1'.repeat(32),digest='2'.repeat(64)
window.fetch=async(input:any,init?:RequestInit)=>{
 const url=String(input)
 if(url==='/api/chat/stream'){
  const body=JSON.parse(String(init?.body));requests.push(body);count.value=requests.length;lastRequest.value=body
  if(nextOutcome==='disconnect'){nextOutcome='pending';return new Response('event: token\ndata: {"content":"连接中断前的回复"}\n\n',{headers:{'Content-Type':'text/event-stream'}})}
  const confirmation=!!body.confirmation
  if(nextOutcome==='privacy'){
   nextOutcome='pending'
   return new Response(actualWire.confirmation_privacy_sse.replaceAll('synthetic-confirmation',body.request_id),{headers:{'Content-Type':'text/event-stream'}})
  }
  if(confirmation&&nextOutcome!=='limited')confirmed.add(body.confirmation.pending_request_id)
  const completed=confirmation||confirmed.has(body.request_id)
  const result={answer:completed?'已按确认保存饮食记录。':'这项饮食记录使用估算，需要确认后才会保存。',request_id:body.request_id,message_id:body.request_id+':assistant',run_id:'synthetic-run',diagnostics_status:'ready',metrics:{react_write:completed?'completed':'pending',execution_mode:'react_write',mode_reason:'explicit_write',receipt_refs:completed?['nutrition:c2-2:'+pendingOp+':g0:17']:[],pending_confirmations:completed?[]:[{pending_request_id:body.request_id,pending_op_id:pendingOp,pending_digest:digest,expires_at:new Date(Date.now()+(nextOutcome==='expired'?-1000:900000)).toISOString()}]}}
  if(nextOutcome==='partial'){result.metrics.react_write='commit_unknown';result.metrics.receipt_refs=['training:c2-1:'+pendingOp+':g0:11'];result.metrics.pending_confirmations=[];result.answer='一项训练有回执，其余结果尚未核对。'}
  if(nextOutcome==='limited'){result.metrics.react_write='limited';result.metrics.execution_mode='limited';result.metrics.mode_reason='react_write_disabled';result.metrics.pending_confirmations=[];result.answer='记录功能尚未开放。'}
  const json=nextOutcome==='json';nextOutcome='pending'
  if(json)return Response.json(result)
  const {answer,...meta}=result
  return new Response(`event: meta\ndata: ${JSON.stringify(meta)}\n\nevent: token\ndata: ${JSON.stringify({content:answer})}\n\nevent: done\ndata: ${JSON.stringify(meta)}\n\n`,{headers:{'Content-Type':'text/event-stream'}})
 }
 if(url.includes('/chat/conversations'))return Response.json({conversations:[]})
 if(url.includes('/chat/history'))return Response.json({messages:[]})
 if(url.includes('/dashboard'))return Response.json({profile:{display_name:'界面验证'},workouts:[],sessions:[],nutrition:[],today:{},food_catalog:[{name:'鸡蛋',kcal:144}]})
 if(url.includes('/nutrition/summary'))return Response.json({target_ready:false})
 throw new Error('Fixture forbids unsupported fetch')
}
ui.online=true;ui.profile.display_name='界面验证';location.hash='/ai-chat'
createApp({setup(){return()=>h('div',[h('details',{style:'position:fixed;top:4px;right:4px;z-index:1000;background:#18181b;padding:4px;font-size:11px'},[
 h('summary','模拟环境'),...['disconnect','json','partial','limited','privacy','expired'].map(value=>h('button',{style:'display:block;padding:4px',onClick:()=>nextOutcome=value},'夹具：'+value)),h('p',{id:'fixture-request-count','data-request-id':lastRequest.value?.request_id},`模拟请求 ${count.value} · ${lastRequest.value?.mode||'无'} · 确认字段 ${Object.keys(lastRequest.value?.confirmation||{}).length}`)]),h(App)])}}).mount('#app')
