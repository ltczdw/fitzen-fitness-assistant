<script setup lang="ts">
import {computed,onBeforeUnmount,ref} from 'vue'
import {renderReply,replyPlainText} from './chat-format'
import { useFitzen, type ChatMessage } from '../../fitzen/state'
const {ui,notify,stopChat,retryChat,rateAnswer,confirmPending,resumeChat,pendingUnavailable}=useFitzen()
const replyCache=new WeakMap<ChatMessage,{content:string,html:string}>()
const renderedReplies=computed(()=>ui.messages.map(item=>{
 if(item.role!=='assistant'||!item.content)return ''
 const cached=replyCache.get(item)
 if(cached?.content===item.content)return cached.html
 const html=renderReply(item.content)
 replyCache.set(item,{content:item.content,html})
 return html
}))
const copiedIndex=ref<number|null>(null)
let copyTimer:ReturnType<typeof setTimeout>|undefined
async function copyReply(index:number){try{await navigator.clipboard.writeText(replyPlainText(ui.messages[index]?.content||''));copiedIndex.value=index;clearTimeout(copyTimer);copyTimer=setTimeout(()=>copiedIndex.value=null,2000)}catch{notify('复制失败，请选择回答文字后手动复制')}}
const clock=ref(Date.now())
const expiryTimer=setInterval(()=>clock.value=Date.now(),1000)
onBeforeUnmount(()=>{clearInterval(expiryTimer);clearTimeout(copyTimer)})
function writeLabel(item:ChatMessage){
 const w=item.write!
 if(w.blocked)return '请求已失效'
 if(!w.settled)return w.receipts.length?'已有保存回执 · 回复未完整接收':'保存结果待核对'
 if(w.status==='completed')return w.receipts.length?'保存回执已收到':'执行完成 · 未收到保存回执'
 if(w.status==='pending'&&w.pending.length&&w.pending.every(t=>pendingUnavailable(item,t,clock.value)==='已确认保存'))return '此项已确认保存 · 可核对原请求'
 if(w.receipts.length)return '已有保存回执 · 本次执行尚未完成'
 if(w.status==='pending')return '等待你的确认'
 if(w.status==='commit_unknown')return '保存状态待核对'
 if(w.status==='limited')return '记录功能当前受限'
 return '本次执行尚未完成'
}
function pendingSource(item:ChatMessage){
 const id=item.request?.confirmation?.pending_request_id
 return id?ui.messages.find(m=>m.request?.requestId===id)?.request?.message||'原始记录':item.request?.message
}
function canResume(item:ChatMessage){return item.request&&!item.write?.blocked&&(!item.write?.settled||!['completed','pending'].includes(item.write.status)||(item.write.status==='pending'&&!item.write.pending.length)||(item.write.pending.length>0&&item.write.pending.every(t=>pendingUnavailable(item,t,clock.value)==='已确认保存')))}
</script>
<template>
 <div v-if="!ui.messages.length" class="flex flex-col items-center justify-center py-20 text-center gap-4">
  <span class="material-symbols-outlined text-primary text-4xl">auto_awesome</span><h2 class="text-xl font-semibold">今天有什么可以帮你？</h2>
  <p class="text-on-surface-variant text-sm">记录训练、分析饮食，或讨论你的下一步目标。</p>
 </div>
 <div v-for="(item,index) in ui.messages" :key="index" class="agent-message flex gap-3" :class="item.role==='user'?'justify-end':'justify-start'">
  <span v-if="item.role!=='user'" class="material-symbols-outlined text-primary p-2 bg-surface-container-high rounded-full h-10 w-10">auto_awesome</span>
  <div class="agent-message-content max-w-[85%] rounded-2xl p-4 text-sm leading-7 whitespace-pre-wrap break-words" :class="[item.role==='user'?'bg-surface-container-high':'bg-surface-container-low',item.role!=='user'&&ui.sending&&index===ui.messages.length-1?'chat-streaming-bubble':'']">
   <div v-if="item.role==='assistant'&&item.content" class="chat-rich-text" v-html="renderedReplies[index]"></div>
   <span v-else>{{item.content || (ui.sending?'正在思考…':'未返回内容')}}</span><span v-if="item.role!=='user'&&ui.sending&&index===ui.messages.length-1" class="chat-stream-cursor" aria-hidden="true"></span>
   <section v-if="item.write" @click.stop class="chat-write-state" :class="{'has-receipts':item.write.receipts.length}" aria-label="记录执行状态" aria-live="polite">
    <strong>{{writeLabel(item)}}</strong>
    <p v-if="item.write.receipts.length">以下回执对应已保存部分；其余记录以各自状态为准。</p>
    <details v-if="item.write.receipts.length"><summary>查看 {{item.write.receipts.length}} 条保存回执</summary><ul><li v-for="receipt in item.write.receipts" :key="receipt"><code>{{receipt}}</code></li></ul></details>
    <div v-for="(target,targetIndex) in item.write.pending" :key="target.pending_op_id+target.pending_digest" class="chat-pending">
     <span class="chat-state-caption">待确认记录 · {{targetIndex+1}}</span>
     <blockquote>{{pendingSource(item)}}</blockquote>
     <p>确认后保存这项记录。服务未提供估算明细；如需修改，请重新描述。</p>
     <p>有效期至 <time :datetime="target.expires_at">{{new Date(target.expires_at).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}}</time> · {{new Date(target.expires_at).toLocaleDateString('zh-CN')}}</p>
     <button class="fz-button fz-primary" :disabled="ui.sending||!!pendingUnavailable(item,target,clock)" @click="confirmPending(index,targetIndex)">{{pendingUnavailable(item,target,clock)||'确认保存这项记录'}}</button>
    </div>
    <p v-if="item.write.status==='pending'&&!item.write.pending.length">确认标识缺失或不匹配，请先用原请求核对。</p>
    <p v-if="item.write.status==='limited'">{{item.write.reason==='react_write_disabled'||item.write.reason==='react_write_disabled_frozen'?'服务端尚未开放聊天记录功能。':'请核对已有记录；当前请求不能继续保存。'}}</p>
    <p v-if="!item.write.settled&&!item.write.blocked">连接停止不代表保存取消。恢复将沿用本次请求标识。</p>
    <div class="chat-state-actions">
     <a v-if="item.write.receipts.length" class="fz-button" href="#/nutrition">查看饮食记录</a>
     <a v-if="item.write.receipts.length" class="fz-button" href="#/training">查看训练记录</a>
     <button v-if="canResume(item)" class="fz-button" :disabled="ui.sending" @click="resumeChat(index)">用原请求核对 / 恢复</button>
    </div>
   </section>
   <div v-if="item.role==='assistant'&&item.content&&!ui.sending" class="agent-reply-actions mt-3 flex gap-3 text-xs whitespace-normal">
    <button type="button" @click.stop="copyReply(index)" :aria-label="copiedIndex===index?'回答已复制':'复制回答'">{{copiedIndex===index?'已复制':'复制'}}</button>
    <button v-if="item.run_id&&!item.write?.blocked" :disabled="!!item.feedback||item.feedbackPending" @click="rateAnswer(index,'up')">{{item.feedback==='up'?'已反馈有帮助':'有帮助'}}</button>
    <button v-if="item.run_id&&!item.write?.blocked" :disabled="!!item.feedback||item.feedbackPending" @click="rateAnswer(index,'down')">{{item.feedback==='down'?'已反馈未解决':'未解决'}}</button>
   </div>
  </div>
 </div>
 <p v-if="ui.chatError" role="alert" class="text-error text-sm">{{ui.chatError}} <button v-if="ui.canRetryChat&&!ui.sending" @click.stop="retryChat" class="underline">{{ui.message.trim()?'发送修改后的消息':'重试原请求'}}</button></p>
 <button v-if="ui.sending" @click.stop="stopChat" class="fz-button self-start">停止生成</button>
</template>
<style scoped>
.agent-message-content{min-width:0}
.chat-rich-text{white-space:normal;overflow-wrap:anywhere}
.chat-rich-text :deep(p),.chat-rich-text :deep(blockquote){margin:0 0 12px;white-space:pre-line}
.chat-rich-text :deep(h3){margin:16px 0 8px;font-size:1em;font-weight:600;line-height:1.7}
.chat-rich-text :deep(ul),.chat-rich-text :deep(ol){margin:8px 0 12px;padding-left:1.5em;list-style-position:outside}
.chat-rich-text :deep(ul){list-style-type:disc}.chat-rich-text :deep(ol){list-style-type:decimal}
.chat-rich-text :deep(li){padding-left:2px;margin:3px 0;white-space:pre-line}
.chat-rich-text :deep(strong){font-weight:600}.chat-rich-text :deep(a){color:var(--fz-primary);text-decoration:underline;text-underline-offset:3px}
.chat-rich-text :deep(blockquote){padding-left:12px;border-left:1px solid var(--fz-outline-variant);color:var(--fz-on-surface-variant)}
.chat-rich-text :deep(code){padding:1px 4px;border-radius:4px;background:var(--fz-surface-container-low);font-size:.95em}
.chat-rich-text :deep(pre){max-width:100%;overflow-x:auto;padding:12px;margin:12px 0;border-radius:8px;background:var(--fz-surface-container-low);white-space:pre}
.chat-rich-text :deep(pre code){padding:0;background:transparent}
.chat-rich-text :deep(.chat-answer-table){max-width:100%;overflow-x:auto;margin:12px 0}
.chat-rich-text :deep(table){width:100%;border-collapse:collapse;text-align:left;font-size:.95em;line-height:1.6}
.chat-rich-text :deep(th),.chat-rich-text :deep(td){min-width:72px;padding:8px 10px;border-bottom:1px solid var(--fz-outline-variant);vertical-align:top;white-space:normal}
.chat-rich-text :deep(th){font-weight:600;background:var(--fz-surface-container-low)}
.chat-rich-text :deep(> :first-child){margin-top:0}.chat-rich-text :deep(> :last-child){margin-bottom:0}

.chat-write-state{margin-top:16px;padding-top:0;border:0;white-space:normal;line-height:1.65}
.chat-write-state strong{font-size:13px;color:var(--fz-on-surface)}.chat-write-state.has-receipts strong{color:var(--fz-primary)}
.chat-write-state p,.chat-write-state summary{margin:6px 0;font-size:12px;color:var(--fz-on-surface-variant)}
.chat-write-state summary{cursor:pointer}.chat-write-state code{font-size:11px;overflow-wrap:anywhere;color:var(--fz-on-surface-variant)}
.chat-pending{--fz-chat-frame-radius:16px;--fz-chat-frame-inset:12px;margin-top:12px;padding:var(--fz-chat-frame-inset);border:0;border-radius:var(--fz-chat-frame-radius);background:var(--fz-surface-container-low);box-shadow:inset 0 0 0 var(--fz-chat-border-width) var(--fz-outline-variant)}
.chat-state-caption{font-size:11px;color:var(--fz-on-surface-variant)}.chat-pending blockquote{margin:6px 0;padding-left:0;border:0;white-space:pre-wrap;overflow-wrap:anywhere}
.chat-pending button{margin-top:8px;min-height:40px;border:0;border-radius:calc(var(--fz-chat-frame-radius) - var(--fz-chat-frame-inset));box-shadow:none;text-align:left}.chat-state-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.chat-state-actions a{text-decoration:none}

.chat-state-actions .fz-button{border:0;background:transparent;box-shadow:none}.chat-state-actions .fz-button:hover{background:var(--fz-surface-container-low)}
</style>
