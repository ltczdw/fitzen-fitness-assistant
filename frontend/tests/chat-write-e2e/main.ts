import {createApp,h,ref} from 'vue'
import App from '../../src/fitzen/App.vue'
import {ui} from '../../src/fitzen/state'
import '../../src/fitzen/fitzen.css'
// Native fetch reaches Vite's loopback proxy and the real API. No response mocks.
location.hash='/ai-chat'
const report=ref<any>(null)
async function recoverPrivate(){
 report.value=null
 const init={method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}
 const before=await (await fetch('/__fixture__/report',init)).json()
 const original=[...before.requests].reverse().find((r:any)=>r.events.some((e:any)=>e.metrics.react_write==='pending'&&e.metrics.pending_confirmations?.length))
 if(!original)throw new Error('No real pending request to recover')
 const response=await fetch('/api/chat',{...init,body:JSON.stringify({message:'帮我记录今天午餐：300克神秘盖饭。谢谢',user_id:'demo-user',conversation_id:original.conversation_id,request_id:original.request_id,mode:original.mode})})
 const payload=await response.json()
 const after=await (await fetch('/__fixture__/report',init)).json()
 report.value={...after,privacy_recovery:{request_id:original.request_id,status:response.status,before,response:payload}}
}
async function control(path:string,body?:object){
 report.value=null
 const response=await fetch('/__fixture__/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})})
 if(!response.ok)throw new Error('Fixture control failed')
 report.value=await response.json()
}
createApp({setup(){return()=>h('div',[
 h('details',{style:'position:fixed;top:4px;right:4px;z-index:200;background:#18181b;padding:5px;font-size:11px;max-width:350px;max-height:80vh;overflow:auto'},[
 h('summary','隔离 SQLite · 模型模拟 · 真实 API'),
 ...[['正常传输','normal'],['下次提交后断流','drop-after-meta'],['下次提交后暂停','pause-after-meta']].map(([label,value])=>h('button',{style:'display:block;padding:5px',onClick:()=>control('transport',{kind:value})},label)),
 h('button',{style:'display:block;padding:5px',onClick:()=>control('privacy',{conversation_id:ui.conversationId})},'撤销测试会话隐私权限'),
 h('button',{style:'display:block;padding:5px',onClick:recoverPrivate},'验证原请求隐私拒绝'),
 h('button',{style:'display:block;padding:5px',onClick:()=>control('report')},'核对隔离数据库'),
 h('button',{style:'display:block;padding:5px',onClick:()=>control('finish')},'结束并验收整链'),
 h('pre',{id:'e2e-report',style:'white-space:pre-wrap;overflow-wrap:anywhere'},JSON.stringify(report.value,null,2))]),h(App)])}}).mount('#app')
