<script setup lang="ts">
import {computed,nextTick,onBeforeUnmount,onMounted,ref,watch} from 'vue'
import {createCompositionGuard,createChatScrollPolicy} from '../chat-interaction'
import {useFitzen,loadHistory,loadChatCapabilities,chatRecordStatusText} from '../state'
import {ChatMessages} from '../../components/ui'
import {fmt,today,editRecord} from '../metrics'
const {ui,sendChat,stopChat,newChat,openFood,openWorkout,openRecords,exportChat}=useFitzen()
const chatStream=ref<HTMLElement|null>(null),messageInput=ref<HTMLTextAreaElement|null>(null),contextPanel=ref<HTMLElement|null>(null),contextToggle=ref<HTMLButtonElement|null>(null),actionMenu=ref<HTMLDetailsElement|null>(null)
const contextOpen=ref(false),contextTab=ref<'today'|'records'>('today'),selectedCommand=ref(0),suppressedCommand=ref('')
const currentTitle=computed(()=>ui.conversations.find(c=>c.conversation_id===ui.conversationId)?.title||'新对话')
const commands=[{id:'train',name:'训练',icon:'fitness_center',description:'分析已有训练，安排下一步',prompt:'请根据我的实际训练记录，分析下一步的训练安排',mode:'react_readonly' as const},{id:'food',name:'饮食',icon:'restaurant',description:'描述一餐，核对后记录',prompt:'帮我记录今天的一餐：',mode:'react_write' as const},{id:'recover',name:'恢复',icon:'bedtime',description:'结合记录讨论恢复和休息',prompt:'请根据我的训练记录，给出恢复与休息建议',mode:'react_readonly' as const}]
const visibleCommands=computed(()=>ui.message.startsWith('/')&&!ui.sending&&suppressedCommand.value!==ui.message?commands.filter(c=>(c.name+c.id+c.description).toLowerCase().includes(ui.message.slice(1).toLowerCase())):[])
const recentRecords=computed(()=>[...ui.sessions.map(record=>({key:'session:'+record.id,record})),...ui.workouts.map(record=>({key:'workout:'+record.id,record}))].sort((a,b)=>{
 const parsedA=Date.parse(a.record.session_date||a.record.date||''),parsedB=Date.parse(b.record.session_date||b.record.date||'')
 const stampA=Number.isFinite(parsedA)?parsedA:-Infinity,stampB=Number.isFinite(parsedB)?parsedB:-Infinity
 return stampA===stampB?0:stampA>stampB?-1:1
}).slice(0,5))
const narrowContext=ref(window.matchMedia('(max-width:1279px)').matches)
function resizeContext(){narrowContext.value=window.matchMedia('(max-width:1279px)').matches}
function signalContext(){window.dispatchEvent(new CustomEvent('fitzen-context-overlay',{detail:contextOpen.value&&narrowContext.value}))}
function searchOpening(){contextOpen.value=false;closeActions()}
watch([contextOpen,narrowContext],signalContext)

function resizeInput(){const el=messageInput.value;if(el){el.style.height='auto';el.style.height=Math.min(180,Math.max(64,el.scrollHeight))+'px'}}
const composition=createCompositionGuard(),scrollPolicy=createChatScrollPolicy(),showLatest=ref(false)
const chatContent=ref<HTMLElement|null>(null),composerWrap=ref<HTMLElement|null>(null),latestBottom=ref(200)
let followQueued=false,contentObserver:ResizeObserver|undefined,disposed=false
function messageLine(){const el=chatContent.value?.querySelector<HTMLElement>('.agent-message.justify-start .agent-message-content')||chatContent.value?.querySelector<HTMLElement>('.agent-message-content');return el?parseFloat(getComputedStyle(el).lineHeight)||26:26}
function updateLatest(){latestBottom.value=(composerWrap.value?.offsetHeight||188)+12;const el=chatStream.value;showLatest.value=!!el&&scrollPolicy.showLatest(el,messageLine())}
function followContent(){
 if(followQueued||disposed)return
 followQueued=true
 void nextTick(()=>{followQueued=false;if(disposed)return;const el=chatStream.value;if(el&&scrollPolicy.following){el.scrollTop=el.scrollHeight;scrollPolicy.followed(el)}updateLatest()})
}
function scrollLatest(){scrollPolicy.reset();followContent()}
function trackScroll(){const el=chatStream.value;if(el)scrollPolicy.track(el,messageLine());updateLatest()}
function pauseFollow(){const el=chatStream.value;if(el)scrollPolicy.pause(el);updateLatest()}
function scrollKey(event:KeyboardEvent){if(['ArrowUp','ArrowDown','PageUp','PageDown','Home','End',' '].includes(event.key))pauseFollow()}
async function sendMessage(){if(ui.sending||!ui.message.trim())return;scrollLatest();await sendChat()}
function fillCommand(command:typeof commands[number]){if(ui.sending||(command.mode==='react_write'&&!ui.chatRecordCapability.available))return;ui.chatMode=command.mode;ui.message=command.prompt;selectedCommand.value=0;void nextTick(()=>messageInput.value?.focus())}
function inputKey(event:KeyboardEvent){
 if(composition.active(event))return
 if(event.key==='Enter'&&composition.blocksCommitEnter()){event.preventDefault();event.stopPropagation();return}
 if(event.shiftKey)return
 if(event.key==='Enter'&&event.repeat){event.preventDefault();return}
 if(visibleCommands.value.length&&['ArrowDown','ArrowUp','Enter','Escape'].includes(event.key)){
  event.preventDefault();if(event.key==='Escape')suppressedCommand.value=ui.message
  else if(event.key==='Enter')fillCommand(visibleCommands.value[selectedCommand.value%visibleCommands.value.length]!)
  else selectedCommand.value=(selectedCommand.value+(event.key==='ArrowDown'?1:visibleCommands.value.length-1))%visibleCommands.value.length
  return
 }
 if(event.key==='Enter'){event.preventDefault();void sendMessage()}
}
function closeActions(){if(actionMenu.value)actionMenu.value.open=false}
function closeContext(){contextOpen.value=false;void nextTick(()=>contextToggle.value?.focus())}
function outside(event:MouseEvent){if(actionMenu.value&&!actionMenu.value.contains(event.target as Node))closeActions()}
function key(event:KeyboardEvent){if(event.defaultPrevented)return;const dialog=(event.target as Element)?.closest?.('[role="dialog"]');if(dialog&&dialog.id!=='agent-context')return;if(event.key==='Escape'){closeActions();if(contextOpen.value)closeContext()}if(event.key==='Tab'&&contextOpen.value&&narrowContext.value){const controls=Array.from(contextPanel.value?.querySelectorAll<HTMLElement>('button,a[href]')||[]);const first=controls[0],last=controls[controls.length-1];if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus()}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus()}}}
watch(()=>ui.message,()=>{selectedCommand.value=0;suppressedCommand.value='';void nextTick(resizeInput)})
watch(()=>ui.conversationId,scrollLatest)
watch(()=>[ui.messages.length,ui.messages[ui.messages.length-1]?.content],followContent)
watch(contextOpen,open=>{if(open)void nextTick(()=>contextPanel.value?.querySelector<HTMLButtonElement>('button')?.focus())})
onMounted(()=>{if(typeof ResizeObserver!=='undefined'){contentObserver=new ResizeObserver(followContent);if(chatContent.value)contentObserver.observe(chatContent.value);if(chatStream.value)contentObserver.observe(chatStream.value);if(composerWrap.value)contentObserver.observe(composerWrap.value)}void loadChatCapabilities();window.addEventListener('focus',loadChatCapabilities);void nextTick(()=>{resizeInput();scrollLatest()});document.addEventListener('click',outside);window.addEventListener('resize',resizeContext);window.addEventListener('fitzen-page-search-open',searchOpening);window.addEventListener('keydown',key)})
onBeforeUnmount(()=>{disposed=true;composition.dispose();contentObserver?.disconnect();window.removeEventListener('focus',loadChatCapabilities);document.removeEventListener('click',outside);window.removeEventListener('resize',resizeContext);window.removeEventListener('fitzen-page-search-open',searchOpening);window.dispatchEvent(new CustomEvent('fitzen-context-overlay',{detail:false}));window.removeEventListener('keydown',key)})
</script>
<template>
<section class="screen-chat agent-chat" :class="{'agent-chat-empty':!ui.messages.length,'agent-context-visible':contextOpen}" aria-label="AI 工作区">
 <div class="agent-session-toolbar" :inert="contextOpen&&narrowContext"><div><span aria-hidden="true" class="material-symbols-outlined">chat_bubble_outline</span><h1>{{currentTitle}}</h1><span v-if="ui.sending" class="agent-run-state" role="status">正在回复</span></div><div><button type="button" class="agent-icon-button" :disabled="!ui.messages.length" @click.stop="exportChat" aria-label="导出当前对话"><span aria-hidden="true" class="material-symbols-outlined">ios_share</span></button><button ref="contextToggle" type="button" class="agent-text-button" aria-label="上下文" :class="{'is-selected':contextOpen}" :aria-expanded="contextOpen" aria-controls="agent-context" @click.stop="contextOpen=!contextOpen"><span aria-hidden="true" class="material-symbols-outlined">right_panel_open</span><span>上下文</span></button></div></div>
 <div class="agent-chat-layout"><div class="agent-conversation" :inert="contextOpen&&narrowContext" :class="{'has-messages':ui.messages.length}">
  <p v-if="!ui.messages.length&&ui.chatError" role="alert" class="agent-history-error">{{ui.chatError}} <button @click.stop="loadHistory">重试</button></p>
  <div v-if="!ui.messages.length" class="agent-welcome"><svg class="agent-welcome-mark" viewBox="0 0 40 40" aria-hidden="true"><path d="M3 21h9l5-14 7 27 5-13h8" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" /></svg><h2>今天想先完成什么？</h2><p>聊聊训练、记下一餐，或一起调整你的下一步。</p></div>
  <div v-show="ui.messages.length" ref="chatStream" id="chat-stream" class="agent-transcript" aria-label="对话内容" tabindex="0" @scroll="trackScroll" @wheel.passive="pauseFollow" @touchstart.passive="pauseFollow" @pointerdown="pauseFollow" @keydown="scrollKey"><div ref="chatContent" class="agent-transcript-content"><ChatMessages /></div></div>
  <button v-if="ui.messages.length&&showLatest" type="button" class="agent-scroll-latest" :style="{bottom:latestBottom+'px'}" aria-label="跳至最新消息" @click.stop="scrollLatest"><span aria-hidden="true" class="material-symbols-outlined">arrow_downward</span>跳至最新</button>
  <div ref="composerWrap" class="agent-composer-wrap"><div class="agent-composer" :class="{'is-sending':ui.sending}">
   <div v-if="visibleCommands.length" id="agent-command-options" class="agent-command-menu" role="listbox" aria-label="快捷指令"><button v-for="(command,index) in visibleCommands" :key="command.id" :id="'agent-command-'+command.id" role="option" :disabled="command.mode==='react_write'&&!ui.chatRecordCapability.available" :title="command.mode==='react_write'&&!ui.chatRecordCapability.available?chatRecordStatusText():command.description" :aria-selected="selectedCommand===index" @click.stop="fillCommand(command)"><span aria-hidden="true" class="material-symbols-outlined">{{command.icon}}</span><span><strong>/{{command.name}}</strong><small>{{command.description}}</small></span><kbd v-if="selectedCommand===index">↵</kbd></button></div>
   <textarea ref="messageInput" v-model="ui.message" @keydown="inputKey" @keyup="composition.keyup" @compositionstart="composition.start" @compositionend="composition.end" @blur="composition.cancel" id="chat-input" aria-label="发送给 AI 的消息" :aria-controls="visibleCommands.length?'agent-command-options':undefined" :aria-activedescendant="visibleCommands.length?'agent-command-'+visibleCommands[selectedCommand]?.id:undefined" :placeholder="ui.chatMode==='react_write'?'描述要记录的训练或饮食，包含数量、时间…':'向 FitZen 提问，输入 / 查看快捷指令…'" rows="2" />
   <div class="agent-composer-toolbar"><div><details ref="actionMenu" class="agent-add-menu"><summary class="agent-icon-button" aria-label="添加到工作区"><span aria-hidden="true" class="material-symbols-outlined">add</span></summary><div class="agent-popover"><button type="button" @click.stop="closeActions();openWorkout()"><span aria-hidden="true" class="material-symbols-outlined">fitness_center</span>添加训练记录</button><button type="button" @click.stop="closeActions();openFood()"><span aria-hidden="true" class="material-symbols-outlined">restaurant</span>添加饮食记录</button><button type="button" @click.stop="closeActions();openRecords('training')"><span aria-hidden="true" class="material-symbols-outlined">folder_open</span>查看全部记录</button></div></details></div><div><span class="agent-model-label">FitZen AI</span><button v-if="ui.sending" id="btn-stop" type="button" class="agent-send-button" @click.stop="stopChat" aria-label="停止生成"><span aria-hidden="true" class="material-symbols-outlined">stop</span></button><button v-else id="btn-send" type="button" class="agent-send-button" :disabled="!ui.message.trim()||(ui.chatMode==='react_write'&&!ui.chatRecordCapability.available)" @click.stop="sendMessage" :aria-label="ui.chatMode==='react_write'?'发送记录':'发送消息'"><span aria-hidden="true" class="material-symbols-outlined">arrow_upward</span></button></div></div>
  </div><div class="agent-composer-caption"><span>{{ui.chatMode==='react_write'?chatRecordStatusText():'使用已有记录作为对话上下文'}}</span><span>Enter 发送 · Shift+Enter 换行</span></div>
  <div v-if="!ui.messages.length" class="agent-suggested-actions"><button v-for="command in commands" :key="command.id" type="button" :disabled="ui.sending||(command.mode==='react_write'&&!ui.chatRecordCapability.available)" :title="command.mode==='react_write'&&!ui.chatRecordCapability.available?chatRecordStatusText():command.description" @click.stop="fillCommand(command)"><span aria-hidden="true" class="material-symbols-outlined">{{command.icon}}</span>{{command.name==='训练'?'分析训练':command.name==='饮食'?'记录一餐':'恢复建议'}}<span aria-hidden="true" class="material-symbols-outlined">north_east</span></button></div>
 </div></div>
 <button v-if="contextOpen" class="agent-context-scrim" aria-label="关闭上下文面板" @click.stop="closeContext" />
 <aside v-if="contextOpen" ref="contextPanel" id="agent-context" class="agent-context-panel" aria-label="记录上下文" :role="narrowContext?'dialog':'complementary'" :aria-modal="narrowContext?true:undefined"><div class="agent-context-heading"><strong>记录上下文</strong><button class="agent-icon-button" aria-label="关闭上下文" @click.stop="closeContext"><span aria-hidden="true" class="material-symbols-outlined">close</span></button></div><div class="agent-context-tabs" role="group" aria-label="上下文内容"><button :aria-pressed="contextTab==='today'" @click.stop="contextTab='today'">今日概况</button><button :aria-pressed="contextTab==='records'" @click.stop="contextTab='records'">最近记录</button></div><div class="agent-context-content"><template v-if="contextTab==='today'"><h2>已保存的今日数据</h2><dl><div><dt>完成训练</dt><dd>{{fmt(today.completed_sets)}} <small>组</small></dd></div><div><dt>训练容量</dt><dd>{{fmt(today.training_volume)}} <small>kg</small></dd></div><div><dt>热量摄入</dt><dd>{{fmt(today.calories)}} <small>kcal</small></dd></div><div><dt>蛋白质</dt><dd>{{fmt(today.protein_g,1)}} <small>g</small></dd></div></dl><p>缺失数据显示为 —，不会用估算值补齐你的记录。</p><button class="agent-context-link" @click.stop="openRecords('nutrition')">查看饮食记录<span aria-hidden="true" class="material-symbols-outlined">arrow_forward</span></button><h2>当前训练计划</h2><p>{{ui.data?.plan?.plan_name||'尚未保存训练计划'}}</p></template><template v-else><h2>最近保存的训练</h2><p v-if="!recentRecords.length">还没有训练记录，可以从输入框旁的加号添加。</p><button v-for="record in recentRecords" :key="record.key" class="agent-context-record" @click.stop="editRecord(record.record)"><strong>{{record.record.title||record.record.exercise||'训练记录'}}</strong><span>{{(record.record.session_date||record.record.date||'').slice(0,10)||'日期未记录'}}</span><span aria-hidden="true" class="material-symbols-outlined">chevron_right</span></button><button class="agent-context-link" @click.stop="openRecords('training')">查看全部记录<span aria-hidden="true" class="material-symbols-outlined">arrow_forward</span></button></template></div></aside>
 </div>
</section>
</template>
