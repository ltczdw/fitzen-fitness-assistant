<script setup lang="ts">
import {recentConversations} from './recent-conversations'
import {computed,onMounted,onBeforeUnmount,nextTick,defineAsyncComponent,ref,watch} from 'vue'
import Dashboard from './screens/Dashboard.vue'
import {DeleteButton,FoodModal,RecordsDrawer,WorkoutSessionManager,WorkoutRecordManager,NutritionEditModal as NutritionRecordManager} from '../components/ui'
import {useFitzen,loadHistory} from './state'
import {deleteConfirmation} from '../components/ui/delete-confirmation'
const {ui,navigate,notify,refresh,openFood,openWorkout,openRecords,exportChat,exportRecords,saveProfileData,newChat,loadConversations,switchConversation,deleteConversation}=useFitzen()
const components:Record<string,any>={dashboard:Dashboard,training:defineAsyncComponent(()=>import('./screens/Training.vue')),nutrition:defineAsyncComponent(()=>import('./screens/Nutrition.vue')),stats:defineAsyncComponent(()=>import('./screens/Stats.vue')),'ai-chat':defineAsyncComponent(()=>import('./screens/Chat.vue')),profile:defineAsyncComponent(()=>import('./screens/Profile.vue')),settings:defineAsyncComponent(()=>import('./screens/Settings.vue')),status:defineAsyncComponent(()=>import('./screens/Status.vue')),'nutrition-modal':defineAsyncComponent(()=>import('./screens/Nutrition.vue'))}
const current=computed(()=>components[ui.route]||components.status)
const destinations=[{id:'dashboard',name:'首页',icon:'grid_view'},{id:'training',name:'训练记录',icon:'fitness_center'},{id:'nutrition',name:'饮食记录',icon:'restaurant'},{id:'stats',name:'数据统计',icon:'trending_up'},{id:'ai-chat',name:'AI 工作区',icon:'auto_awesome'},{id:'profile',name:'个人信息',icon:'person'},{id:'settings',name:'设置与偏好',icon:'settings'},{id:'status',name:'系统状态总览',icon:'monitor_heart'}]
const searchResults=computed(()=>destinations.filter(x=>x.name.toLowerCase().includes(ui.searchText.toLowerCase())))
function routeChanged(){ui.route=location.hash.slice(2)||'ai-chat';ui.menu=false;window.scrollTo(0,0);if(ui.route==='nutrition-modal'){ui.route='nutrition';openFood()}if(ui.route==='ai-chat')void loadHistory()}
function key(e:KeyboardEvent){if(e.defaultPrevented)return;const activeDialog=(e.target as Element)?.closest?.('[role="dialog"]');if(activeDialog&&!['agent-page-search','agent-sidebar','agent-context'].includes(activeDialog.id))return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();ui.search=!ui.search;void nextTick(()=>document.querySelector<HTMLInputElement>('#global-search')?.focus())}if(e.key==='Escape'){ui.menu=false;ui.search=false;if(ui.foodOpen&&!ui.savingFood)ui.foodOpen=false;else if(ui.records)ui.records=false}}
function rootInput(event:Event){const input=event.target as HTMLInputElement;if(input.id==='searchInput'){document.querySelectorAll<HTMLTableRowElement>('#tableBody > tr[data-row-id]').forEach(row=>{row.hidden=!row.textContent?.toLowerCase().includes(input.value.toLowerCase());const detail=document.getElementById('detail-'+row.dataset.rowId);if(row.hidden&&detail)detail.classList.add('hidden')})}}
async function action(event:MouseEvent){
 const target=event.target as HTMLElement,link=target.closest<HTMLAnchorElement>('[data-path]')
 if(link){event.preventDefault();navigate(link.dataset.path||'dashboard');return}
 if(target.closest('header')&&target.closest('kbd')){ui.search=true;return}
 const b=target.closest<HTMLElement>('button,a');if(!b)return
 if(ui.route==='settings')return
 const id=b.id,text=(b.textContent||'').trim(),icon=b.querySelector('.material-symbols-outlined')?.textContent?.trim(),label=b.getAttribute('aria-label')
 if(b.closest('#add-food-modal-overlay')){event.preventDefault();if(icon==='close'||text==='取消')navigate('nutrition');else{navigate('nutrition');openFood()};return}
 if(label==='Switch Account'||icon==='unfold_more'){ui.menu=!ui.menu;return}
 if(label==='Notifications'){navigate('status');return}
 if(id==='btn-export'){exportChat();return}
 if(id==='btn-sync-meal'){openFood();return}
 if(id==='saveBioBtn'||text.includes('保存修改')){
 const height=(document.getElementById('inputHeight') as HTMLInputElement)?.value,weight=(document.getElementById('inputWeight') as HTMLInputElement)?.value
 if(height&&weight)await saveProfileData({height_cm:Number(height),body_weight_kg:Number(weight)});return}
 if(icon==='settings'||text.includes('编辑个人档案')){navigate('settings');return}
 if(text.includes('退出登录')||text.includes('切换账户')||text.includes('返回登录')){notify('当前为本地账户，尚未配置登录服务。');return}
 if(text.includes('检查更新')){notify('当前界面版本：FitZen 2.4 · 本地版本');return}
 if(text.includes('导出')){ui.recordsTab=ui.route==='nutrition'?'nutrition':'training';exportRecords();return}
 if(text.includes('快速记录')||text.includes('录入动作')||text.includes('开始训练')){openWorkout();return}
 if(ui.route==='nutrition'&&(icon==='add'||text.includes('添加')||text.includes('补记'))){openFood();return}
 if(ui.route==='nutrition'&&(icon==='edit'||icon==='delete')){openRecords('nutrition');notify('请在真实记录中选择要编辑的条目。');return}
 if(text.includes('拍照')){notify('当前后端没有图像识别接口，可通过添加食物或 AI 文本记录饮食。');return}
 if(text.includes('返回首页')){navigate('dashboard');return}
 if(text.includes('重试')||text.includes('重新加载')){await refresh();notify(ui.online?'服务连接正常':'服务暂不可用，请检查后端');return}
 if(b.classList.contains('gender-btn')){b.parentElement?.querySelectorAll('button').forEach(el=>el.classList.remove('text-primary','bg-surface-container-high'));b.classList.add('text-primary','bg-surface-container-high');return}
 if(b.classList.contains('goal-tab')){b.parentElement?.querySelectorAll('button').forEach(el=>el.classList.remove('text-primary','bg-surface-container-high'));b.classList.add('text-primary','bg-surface-container-high');ui.profile.goal=text.replace(/^[a-z_]+\s*/,'').trim();notify('目标已选择，点击保存修改写入个人资料');return}
 const row=b.closest('tr[data-row-id]')
 if(row){const detail=document.getElementById('detail-'+(row as HTMLElement).dataset.rowId);if(detail){detail.classList.toggle('hidden');return}}
 if(text.includes('上一页')||text.includes('下一页')||/^\d+$/.test(text)){openRecords('training');return}
 if(ui.route==='ai-chat'&&id!=='btn-send'){if(text.includes('建议')||text.includes('训练')||text.includes('饮食')||text.includes('恢复')){ui.message=text.replace(/^[a-z_]+\s*/,'');document.querySelector<HTMLTextAreaElement>('#chat-input')?.focus();return}}
 if(text.includes('AI')||text.includes('教练')||text.includes('对话')){navigate('ai-chat');return}
 if(text.includes('全部')||text.includes('详情')||text.includes('记录')){openRecords(ui.route==='nutrition'?'nutrition':'training');return}
 if(b.tagName==='A'&&b.getAttribute('href')==='#')event.preventDefault()
}
function submit(event:Event){if(ui.route!=='settings'){event.preventDefault();notify('请使用保存按钮提交修改。')}}
onMounted(()=>{routeChanged();window.addEventListener('resize',shellResize);window.addEventListener('fitzen-context-overlay',contextOverlay);window.addEventListener('keydown',shellKey);void loadConversations();window.addEventListener('hashchange',routeChanged);window.addEventListener('keydown',key);window.addEventListener('fitness-workout-records-changed',refresh);window.addEventListener('fitness-nutrition-records-changed',refresh);window.addEventListener('focus',refresh);void refresh()})
onBeforeUnmount(()=>{window.removeEventListener('resize',shellResize);window.removeEventListener('fitzen-context-overlay',contextOverlay);window.removeEventListener('keydown',shellKey);window.removeEventListener('hashchange',routeChanged);window.removeEventListener('keydown',key);window.removeEventListener('fitness-workout-records-changed',refresh);window.removeEventListener('fitness-nutrition-records-changed',refresh);window.removeEventListener('focus',refresh)})
const searchPanel=ref<HTMLElement|null>(null),contextModalActive=ref(false)
let searchReturnFocus:HTMLElement|null=null
const sidebarCollapsed=ref(false),sidebarMobileOpen=ref(false),sidebarPanel=ref<HTMLElement|null>(null),sidebarToggle=ref<HTMLButtonElement|null>(null)
const compactScreen=ref(window.matchMedia('(max-width:1023px)').matches)
const sessionSearchOpen=ref(false),sessionQuery=ref(''),sessionSearchInput=ref<HTMLInputElement|null>(null),deletingConversation=ref('')
try{sidebarCollapsed.value=localStorage.getItem('fitzen-sidebar-collapsed')==='1'}catch{}
const workspaceDestinations=computed(()=>[destinations[4],...destinations.slice(0,4),destinations[5]])
const routeTitle=computed(()=>destinations.find(x=>x.id===ui.route)?.name||'工作区')
const sidebarExpanded=computed(()=>compactScreen.value?sidebarMobileOpen.value:!sidebarCollapsed.value)
const sidebarInert=computed(()=>(compactScreen.value&&!sidebarMobileOpen.value)||contextModalActive.value)
const visibleConversations=computed(()=>recentConversations(ui.conversations,ui.conversationId,ui.messages,sessionQuery.value))
function closeSidebar(){if(compactScreen.value)sidebarMobileOpen.value=false;else sidebarCollapsed.value=true;void nextTick(()=>sidebarToggle.value?.focus())}
function toggleSidebar(){if(compactScreen.value){sidebarMobileOpen.value=!sidebarMobileOpen.value;if(sidebarMobileOpen.value)void nextTick(()=>sidebarPanel.value?.querySelector<HTMLButtonElement>('button')?.focus())}else sidebarCollapsed.value=!sidebarCollapsed.value}
function shellResize(){compactScreen.value=window.matchMedia('(max-width:1023px)').matches;if(!compactScreen.value)sidebarMobileOpen.value=false}
function goTo(route:string){sidebarMobileOpen.value=false;navigate(route)}
async function startConversation(){if(ui.sending)return;await newChat();goTo('ai-chat');void nextTick(()=>document.querySelector<HTMLTextAreaElement>('#chat-input')?.focus())}
async function selectConversation(id:string){if(ui.sending)return;goTo('ai-chat');await switchConversation(id)}
async function removeConversation(conv:any){const id=String(conv.conversation_id||'');if(!id||ui.sending||deletingConversation.value)return;deletingConversation.value=id;try{await deleteConversation(id)}finally{deletingConversation.value=''}}
function contextOverlay(event:Event){contextModalActive.value=Boolean((event as CustomEvent).detail)}
watch(()=>ui.search,open=>{
 if(open){searchReturnFocus=document.activeElement instanceof HTMLElement?document.activeElement:null;sidebarMobileOpen.value=false;window.dispatchEvent(new Event('fitzen-page-search-open'));void nextTick(()=>document.querySelector<HTMLInputElement>('#global-search')?.focus())}
 else void nextTick(()=>{const target=searchReturnFocus;if(target?.isConnected&&!target.closest('[inert]')&&target.getClientRects().length)target.focus();else sidebarToggle.value?.focus()})
})
function searchKey(event:KeyboardEvent){
 if(event.key==='Escape'){event.preventDefault();event.stopPropagation();ui.search=false;return}
 if(event.key==='Tab'){const nodes=Array.from(searchPanel.value?.querySelectorAll<HTMLElement>('button:not(:disabled),input,a[href]')||[]);const first=nodes[0],last=nodes[nodes.length-1];if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus()}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus()}}
}
watch(sidebarCollapsed,value=>{try{localStorage.setItem('fitzen-sidebar-collapsed',value?'1':'0')}catch{}})
watch(()=>ui.route,()=>{sidebarMobileOpen.value=false;deleteConfirmation.clear()})
function shellKey(e:KeyboardEvent){
 if(e.defaultPrevented||ui.search)return;const dialog=(e.target as Element)?.closest?.('[role="dialog"]');if(dialog&&dialog.id!=='agent-sidebar')return;
 if((e.ctrlKey||e.metaKey)&&!e.altKey&&e.key.toLowerCase()==='b'){e.preventDefault();toggleSidebar()}
 if(e.ctrlKey&&e.altKey&&e.key.toLowerCase()==='n'){e.preventDefault();void startConversation()}
 if(e.key==='Escape'&&sidebarMobileOpen.value){e.preventDefault();closeSidebar()}
 if(e.key==='Tab'&&sidebarMobileOpen.value){const items=Array.from(sidebarPanel.value?.querySelectorAll<HTMLElement>('a[href],button:not([disabled]),input')||[]).filter(el=>el.getClientRects().length);const first=items[0],last=items[items.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus()}}
}

</script>
<template>
<div class="agent-shell" :inert="ui.search" :class="{'agent-shell-collapsed':sidebarCollapsed,'agent-shell-mobile-open':sidebarMobileOpen}" @click="action" @input="rootInput" @submit="submit">
  <button v-if="sidebarMobileOpen" class="agent-sidebar-scrim" aria-label="关闭导航" @click.stop="closeSidebar" />
  <aside ref="sidebarPanel" id="agent-sidebar" class="agent-sidebar" aria-label="工作区导航" :role="compactScreen&&sidebarMobileOpen?'dialog':'complementary'" :aria-hidden="compactScreen&&!sidebarMobileOpen?true:undefined" :aria-modal="compactScreen&&sidebarMobileOpen?true:undefined" :inert="sidebarInert">
    <div class="agent-brand-row"><a href="#/ai-chat" class="agent-brand" @click.stop.prevent="goTo('ai-chat')" aria-label="FitZen AI 工作区"><svg viewBox="0 0 32 32" aria-hidden="true"><path d="M3 17h7l4-11 6 21 4-10h5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" /></svg><strong>FitZen</strong></a><button class="agent-icon-button agent-sidebar-close" aria-label="收起侧栏" @click.stop="closeSidebar"><span aria-hidden="true" class="material-symbols-outlined">left_panel_close</span></button></div>
    <button class="agent-new-session" aria-label="新建对话" title="新建对话（Ctrl+Alt+N）" :disabled="ui.sending" @click.stop="startConversation"><span aria-hidden="true" class="material-symbols-outlined">edit_square</span><span>新建对话</span><kbd>Ctrl Alt N</kbd></button>
    <nav class="agent-primary-nav" aria-label="主要页面"><a v-for="item in workspaceDestinations" :key="item.id" :href="'#/'+item.id" :aria-label="item.name" :title="item.name" :aria-current="ui.route===item.id?'page':undefined" @click.stop.prevent="goTo(item.id)"><span aria-hidden="true" class="material-symbols-outlined">{{item.icon}}</span><span>{{item.name}}</span><span v-if="item.id==='ai-chat'&&ui.sending" class="agent-running-dot" aria-label="正在回复" /></a></nav>
    <section class="agent-session-section" aria-label="最近对话"><div class="agent-session-heading"><span>最近对话</span><button class="agent-icon-button" @click.stop="sessionSearchOpen=!sessionSearchOpen; if(sessionSearchOpen) nextTick(()=>sessionSearchInput?.focus())" :aria-expanded="sessionSearchOpen" aria-label="搜索会话"><span aria-hidden="true" class="material-symbols-outlined">filter_list</span></button></div><input v-if="sessionSearchOpen" ref="sessionSearchInput" v-model="sessionQuery" class="agent-session-search" type="search" aria-label="搜索会话标题" placeholder="搜索对话…" />
      <p v-if="!visibleConversations.length" class="agent-session-empty">{{sessionQuery?'没有匹配的对话':'开始新对话后，会显示在这里。'}}</p>
      <div v-for="conv in visibleConversations" :key="conv.conversation_id" class="agent-session-row" :class="{'is-current':ui.route==='ai-chat'&&conv.conversation_id===ui.conversationId}"><button type="button" :disabled="ui.sending" :aria-current="ui.route==='ai-chat'&&conv.conversation_id===ui.conversationId?'true':undefined" :title="conv.title||'未命名对话'" @click.stop="selectConversation(conv.conversation_id)"><span class="agent-session-dot" /><span>{{conv.title||'未命名对话'}}</span></button><DeleteButton :action="()=>removeConversation(conv)" :subject="'对话 '+(conv.title||'未命名对话')" :disabled="ui.sending||!!deletingConversation" :loading="deletingConversation===conv.conversation_id" icon-only data-conversation-delete /></div>
    </section>
    <div class="agent-sidebar-footer"><button class="agent-status-link" @click.stop="goTo('status')" :title="ui.online?'服务已连接':'查看连接状态'"><span :class="ui.online?'is-online':''" class="agent-status-dot" /><span>{{ui.loading?'正在同步':ui.online?'本地服务已连接':'服务未连接'}}</span></button><button class="agent-account" @click.stop="goTo('profile')" aria-label="打开个人信息"><span class="agent-account-avatar">{{(ui.profile.display_name||'我')[0]}}</span><span><strong>{{ui.profile.display_name||'我的账户'}}</strong><small>{{ui.profile.goal||'个人健身空间'}}</small></span></button><button class="agent-icon-button" @click.stop="goTo('settings')" aria-label="设置与偏好"><span aria-hidden="true" class="material-symbols-outlined">settings</span></button></div>
  </aside>
  <div class="agent-shell-main" :inert="sidebarMobileOpen"><header class="agent-global-header" :inert="contextModalActive"><div class="agent-header-left"><button ref="sidebarToggle" class="agent-icon-button" aria-label="切换侧栏（Ctrl+B）" :aria-expanded="sidebarExpanded" aria-controls="agent-sidebar" @click.stop="toggleSidebar"><span aria-hidden="true" class="material-symbols-outlined">left_panel_open</span></button><span class="agent-breadcrumb">个人空间<span>/</span><strong>{{routeTitle}}</strong></span></div><div class="agent-global-actions"><button @click.stop="openWorkout" class="agent-text-button" aria-label="添加训练记录"><span aria-hidden="true" class="material-symbols-outlined">fitness_center</span><span>添加训练</span></button><button @click.stop="openFood()" class="agent-text-button" aria-label="添加饮食记录"><span aria-hidden="true" class="material-symbols-outlined">restaurant</span><span>添加饮食</span></button><button @click.stop="openRecords(ui.route==='nutrition'?'nutrition':'training')" class="agent-icon-button" aria-label="查看全部记录"><span aria-hidden="true" class="material-symbols-outlined">folder_open</span></button></div></header>
    <div class="agent-page-host" :class="{'agent-page-chat':ui.route==='ai-chat'}"><component :is="current" :key="ui.route" /></div>
  </div>
</div>


<div v-if="ui.menu" class="fixed bottom-20 right-4 z-[70] w-56 bg-surface-container-lowest border border-outline-variant rounded-xl shadow-2xl p-2"><button v-for="item in destinations" :key="item.id" class="w-full flex gap-3 px-3 py-2.5 text-sm rounded-lg hover:bg-surface-container text-on-surface" @click="navigate(item.id)"><span class="material-symbols-outlined text-primary text-lg">{{item.icon}}</span>{{item.name}}</button></div>
<div v-if="ui.search" class="fixed inset-0 z-[100] bg-scrim/70 backdrop-blur-md flex justify-center items-start pt-[15vh] px-4" @click.self="ui.search=false"><section ref="searchPanel" id="agent-page-search" role="dialog" aria-modal="true" aria-label="页面搜索" @keydown="searchKey" class="max-w-lg w-full bg-surface-container-lowest border border-outline-variant rounded-2xl p-4"><input id="global-search" v-model="ui.searchText" aria-label="搜索页面" placeholder="搜索页面、记录…" class="fz-input w-full mb-3"><button v-for="item in searchResults" :key="item.id" class="block w-full text-left p-3 text-sm rounded-lg hover:bg-surface-container" @click="navigate(item.id)">{{item.name}}</button><p v-if="!searchResults.length" class="p-4 text-sm text-on-surface-variant">未找到匹配页面</p><button class="fz-button mt-3" @click="ui.search=false">关闭</button></section></div>
<div v-if="ui.notice" role="status" class="fixed top-20 left-1/2 -translate-x-1/2 z-[150] rounded-xl px-5 py-3 border border-primary/30 bg-surface-container-lowest text-sm shadow-2xl max-w-[90vw]">{{ui.notice}}</div>
<div v-if="ui.loadError" role="alert" class="fixed top-16 right-4 z-[75] max-w-sm bg-error-container text-error rounded-lg p-3 text-xs">{{ui.loadError}} {{ui.data?"当前为上次同步结果。":""}} <button @click="refresh" class="underline">重试</button></div>
<FoodModal/><RecordsDrawer/><WorkoutSessionManager/><WorkoutRecordManager/><NutritionRecordManager/>
</template>
