<script setup lang="ts">
import {computed,ref,watch,nextTick} from 'vue'
import DeleteButton from './DeleteButton.vue'
import {deleteConfirmation} from './delete-confirmation'
import {useFitzen,api,type RecordItem} from '../../fitzen/state'
import {records} from '../../fitzen/metrics'
const {ui,refresh,openFood,openWorkout,exportRecords,notify}=useFitzen()
const list=computed(()=>{const items=ui.recordsTab==='nutrition'?ui.nutrition:records.value;return items.filter(item=>JSON.stringify(item).toLowerCase().includes(ui.recordSearch.toLowerCase())).sort((a,b)=>(b.date||b.session_date||'').localeCompare(a.date||a.session_date||''))})
const pages=computed(()=>Math.max(1,Math.ceil(list.value.length/10)))
const pageItems=computed(()=>list.value.slice((ui.recordPage-1)*10,ui.recordPage*10))
function recordDateTime(value:string){
 if(!value)return '时间未记录'
 const parsed=new Date(value)
 if(!Number.isFinite(parsed.getTime()))return value.slice(0,10)
 return parsed.toLocaleString('zh-CN',{year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false})
}
function clearDeleteConfirmation(){deleteConfirmation.clear()}
watch(()=>[ui.recordSearch,ui.recordsTab],()=>{ui.recordPage=1;clearDeleteConfirmation()})
function edit(item:RecordItem){
 if(ui.recordsTab==='nutrition'){ui.records=false;window.dispatchEvent(new CustomEvent('fitness-edit-nutrition-record',{detail:item}));return}
 if(ui.recordsTab==='training'&&item.exercises){window.dispatchEvent(new CustomEvent('fitness-edit-workout-session',{detail:item}));return}
 window.dispatchEvent(new Event(ui.recordsTab==='nutrition'?'fitness-open-nutrition-records':'fitness-open-workout-records'))
}
async function remove(item:RecordItem){
 clearDeleteConfirmation()
 try{const p=ui.recordsTab==='nutrition'?(item.source==='ai_food_log'?'/nutrition/food/':'/nutrition/'):(item.exercises?'/workout-sessions/':'/workouts/');await api(p+encodeURIComponent(item.id)+'?user_id=demo-user',{method:'DELETE'});await refresh();ui.recordPage=Math.min(ui.recordPage,pages.value);notify('记录已删除')}catch(error){notify(error instanceof Error?error.message:'删除失败')}
}
const dialog=ref<HTMLElement|null>(null)
let previous:HTMLElement|null=null
watch(()=>ui.records,async open=>{if(open){previous=document.activeElement as HTMLElement;await nextTick();dialog.value?.focus()}else previous?.focus()})
function key(e:KeyboardEvent){if(e.key==='Escape')ui.records=false;if(e.key==='Tab'){const items=dialog.value?.querySelectorAll<HTMLElement>('button:not(:disabled),input');if(!items?.length)return;const first=items[0],last=items[items.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus()}}}
</script>
<template>
<Teleport to="body"><div v-if="ui.records" class="fixed inset-0 z-[80] bg-scrim/70 backdrop-blur-sm flex items-center justify-center p-4" @click.self="ui.records=false" @keydown="key">
<section ref="dialog" tabindex="-1" role="dialog" aria-modal="true" aria-labelledby="records-title" class="bg-surface-container-lowest border border-outline-variant rounded-2xl w-full max-w-4xl max-h-[90vh] p-6 flex flex-col text-on-surface outline-none">
<header class="flex justify-between items-center mb-5"><div><p class="text-primary text-[11px] tracking-widest mb-1">MY RECORDS</p><h2 id="records-title" class="text-xl font-semibold">我的真实记录</h2></div><button class="fz-button" aria-label="关闭记录" @click="ui.records=false">✕</button></header>
<div class="flex flex-wrap gap-2 mb-4"><button v-for="(label,key) in {training:'训练记录',nutrition:'饮食记录'}" :key="key" class="fz-button" :class="ui.recordsTab===key?'fz-primary':''" @click="ui.recordsTab=key">{{label}}</button><input v-model="ui.recordSearch" class="fz-input flex-1 min-w-28" placeholder="搜索记录…" aria-label="搜索真实记录"><button @click="refresh" :disabled="ui.loading" class="fz-button">{{ui.loading?'加载中…':'刷新'}}</button><button @click="ui.recordsTab==='nutrition'?openFood():openWorkout()" class="fz-button fz-primary">+ 新增</button><button @click="exportRecords" class="fz-button">导出</button></div>
<p v-if="ui.loadError" role="alert" class="text-error text-sm p-3 bg-error/10 rounded-lg mb-4">{{ui.loadError}}</p>
<div class="overflow-y-auto flex-1"><div v-if="ui.loading&&!list.length" class="text-center py-20 text-on-surface-variant">正在读取记录…</div><div v-else-if="!list.length" class="text-center py-20 text-on-surface-variant"><span class="material-symbols-outlined text-4xl block mb-3">inbox</span>暂无符合条件的记录</div>
<article v-for="item in pageItems" :key="(item.exercises?'session-':'record-')+item.id" class="p-4 mb-2 rounded-xl border border-outline-variant/40 bg-surface-container-lowest">
<div class="flex justify-between items-start gap-3"><div><h3 class="font-semibold text-sm">{{item.title||item.exercise||(ui.recordsTab==='nutrition'?(item.meal_type||'本餐记录'):item.description)}}</h3><p class="text-xs text-on-surface-variant mt-1">{{ui.recordsTab==='nutrition'?recordDateTime(item.date||''):(item.date||item.session_date||'').slice(0,10)}} · {{item.meal_type||'训练'}}<template v-if="ui.recordsTab==='nutrition'"> · {{item.items?.length||1}} 项食物</template></p></div><div class="flex gap-2"><button v-if="ui.recordsTab==='nutrition'&&item.source==='ai_food_log'" class="fz-button text-primary" @click="edit(item)">+ 食物</button><button class="fz-button" @click="edit(item)">{{ui.recordsTab==='nutrition'?'编辑整餐':'编辑'}}</button><DeleteButton :action="()=>remove(item)" :subject="String(item.title||item.exercise||item.meal_type||'记录')" compact /></div></div>
<div v-if="item.exercises" class="text-xs text-on-surface-variant mt-3 space-y-2"><p v-for="ex in item.exercises" :key="ex.id">{{ex.exercise_name}} · <span v-for="(set,i) in ex.sets" :key="i" class="inline-block mr-3"><template v-if="set.duration_minutes != null">{{set.distance_km}} km · {{set.duration_minutes}} 分钟</template><template v-else>{{set.weight_kg}}kg × {{set.reps}}次</template> {{set.completed?'✓':''}}</span></p></div>
<div v-else-if="ui.recordsTab==='nutrition'" class="mt-3">
 <div v-if="item.items?.length" class="grid gap-1.5 mb-2"><div v-for="food in item.items" :key="food.id" class="flex items-center justify-between gap-3 rounded-lg bg-surface-container px-3 py-2 text-xs"><span class="truncate">{{food.food_name||food.raw_food_name}} <em class="not-italic text-on-surface-variant">{{food.weight_g}}g</em></span><span class="text-on-surface-variant whitespace-nowrap">{{food.calories??'—'}} kcal</span></div></div>
 <p class="text-xs text-primary">{{item.calories??'—'}} kcal · 蛋白质 {{item.protein_g??'—'}}g · 碳水 {{item.carbs_g??'—'}}g · 脂肪 {{item.fat_g??'—'}}g</p>
</div>
<p v-else class="text-xs text-primary mt-3">{{item.sets}} 组 × {{item.reps}} 次 · {{item.weight_kg}} kg</p>
</article></div><footer class="flex justify-between items-center pt-4 text-xs text-on-surface-variant"><span>{{list.length}} 条记录 · 第 {{ui.recordPage}} / {{pages}} 页</span><div class="flex gap-2"><button class="fz-button" :disabled="ui.recordPage<=1" @click="ui.recordPage--">上一页</button><button class="fz-button" :disabled="ui.recordPage>=pages" @click="ui.recordPage++">下一页</button></div></footer>
</section></div></Teleport>
</template>
