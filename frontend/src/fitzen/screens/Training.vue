<script setup lang="ts">
import {computed,onMounted,ref,watch} from 'vue'
import {api,ensureExerciseLibrary,exerciseLibrary,notify,refresh,useFitzen} from '../state'
import {DeleteButton} from '../../components/ui'
import {deleteConfirmation} from '../../components/ui/delete-confirmation'
import {records,setsFor,fmt,weekDays,weekSets,weekVolume,pct,bar,editRecord,todayKey} from '../metrics'
const {ui,openWorkout,exportRecords}=useFitzen()
const page=ref(1),expanded=ref<string|null>(null),searchTerm=ref(''),dateRange=ref('30')
const deleting=ref(false)
onMounted(()=>{void ensureExerciseLibrary()})
const exerciseMetadata=computed(()=>new Map(exerciseLibrary.map(item=>[item.name.trim().toLowerCase(),item])))
function exerciseMeta(name:string=''){return exerciseMetadata.value.get(String(name||'').trim().toLowerCase())}
function resetDelete(){deleteConfirmation.clear()}
async function deleteAction(row:typeof rows.value[number]){
 if(deleting.value)return
 deleting.value=true
 try{
  const path=row.exerciseId!=null?'/workout-sessions/'+encodeURIComponent(row.record.id)+'/exercises/'+encodeURIComponent(row.exerciseId):'/workouts/'+encodeURIComponent(row.record.id)
  await api(path+'?user_id=demo-user',{method:'DELETE'})
  notify('训练动作已删除')
  await refresh()
  page.value=Math.min(page.value,pages.value)
 }catch(error){notify(error instanceof Error?error.message:'训练动作删除失败，请重试')}
 finally{deleting.value=false}
}
function dateKey(value:any){return String(value||'').slice(0,10)}
function dateFromToday(range:string){if(range==='all')return null;const days=Number(range);if(!Number.isFinite(days))return null;const date=new Date(todayKey.value+'T12:00:00');date.setDate(date.getDate()-(days-1));return dateKey(date.toISOString())}
function cardIcon(name:string=''){
  const category=exerciseMeta(name)?.category||''
  if(category==='有氧')return 'directions_run'
  if(category==='腿部'||category==='核心')return 'accessibility_new'
  if(category==='胸部'||category==='肩部'||category==='背部'||category==='手臂')return 'fitness_center'
  return 'help_outline'
}
function muscleGroup(name:string=''){
  const meta=exerciseMeta(name)
  return meta?.target_muscles||'未知'
}
function cardioRow(row:any){return row.sets.some((set:any)=>set.duration_minutes!=null)}
function cardioTotal(row:any,key:string){return row.sets.reduce((sum:number,set:any)=>sum+(Number(set[key])||0),0)}
function rowVolume(row:any){return row.sets.reduce((sum:number,set:any)=>sum+(Number(set.weight_kg)||0)*(Number(set.reps)||0),0)}
function rowReps(row:any){const reps=row.sets.map((set:any)=>Number(set.reps)).filter((value:number)=>Number.isFinite(value)&&value>0);return reps.length?Math.round(reps.reduce((sum:number,value:number)=>sum+value,0)/reps.length):null}
function relativeDay(row:any){const key=dateKey(row.date),today=todayKey.value;const delta=key&&today?Math.round((Date.parse(today+'T12:00:00')-Date.parse(key+'T12:00:00'))/86400000):null;if(delta===0)return '今天';if(delta===1)return '昨天';if(delta!==null&&delta>=2&&delta<7)return ['周日','周一','周二','周三','周四','周五','周六'][new Date(key+'T12:00:00').getDay()];return key?Number(key.slice(5,7))+'月'+Number(key.slice(8,10))+'日':''}
function shortDate(row:any){const key=dateKey(row.date);return key?key.slice(5).replace('-','-'):'--'}
function recordTime(row:any){return String(row.record?.updated_at||row.record?.created_at||'').slice(11,16)}
function completionLabel(row:any){return [relativeDay(row),recordTime(row),'完成'].filter(Boolean).join(' ')}
function bestWeight(name:string=''){return Number(ui.data?.exercise_bests?.find((item:any)=>item.exercise===name)?.weight_kg||0)}
function isBest(row:any){return !cardioRow(row)&&Number(row.peak?.weight_kg||0)>0&&Number(row.peak?.weight_kg||0)>=bestWeight(row.name)}
function rowStatus(row:any){if(!row.sets.length)return '未记录';if(row.completed<row.sets.length)return '部分完成';return isBest(row)?'新纪录':'已完成'}
function rowStatusClass(row:any){const status=rowStatus(row);return status==='新纪录'?'training-status-pr':status==='已完成'?'training-status-done':'training-status-muted'}
function difficultyFeedback(value:any){
  const normalized=String(value||'').trim()
  if(normalized==='轻松'||normalized==='很轻松')return '😊 很轻松'
  if(normalized==='困难'||normalized==='很吃力')return '😫 很吃力'
  if(normalized==='正常'||normalized==='刚好')return '😐 刚好'
  return '—'
}
function detailSummary(row:any){return row.completed===row.sets.length?'全部组已完成':row.completed+' / '+row.sets.length+' 组已完成'}
const rows=computed(()=>records.value.flatMap(record=>record.exercises?record.exercises.map((exercise:any)=>({key:'s'+record.id+'e'+exercise.id,record,exerciseId:exercise.id,name:exercise.exercise_name,note:exercise.note||record.note,date:record.session_date,sets:exercise.sets||[]})):[{key:'w'+record.id,record,exerciseId:null,name:record.exercise,note:record.note,date:record.date,sets:setsFor(record)}]).map(row=>({...row,peak:row.sets.reduce((best:any,set:any)=>!best||set.weight_kg>best.weight_kg?set:best,null),completed:row.sets.filter((s:any)=>s.completed!==false).length})))
const filteredRows=computed(()=>{const term=searchTerm.value.trim().toLowerCase(),from=dateFromToday(dateRange.value);return rows.value.filter(row=>{const key=dateKey(row.date),text=[row.name,row.note,row.record?.title].filter(Boolean).join(' ').toLowerCase();return (!term||text.includes(term))&&(!from||(key>=from&&key<=todayKey.value))})})
const pageRows=computed(()=>filteredRows.value.slice((page.value-1)*6,page.value*6)),pages=computed(()=>Math.max(1,Math.ceil(filteredRows.value.length/6)))
watch(rows,items=>{
  if(!items.length){expanded.value=null;return}
  if(expanded.value!==null && items.some(item=>item.key===expanded.value))return
  expanded.value=items[0].key
},{immediate:true})
watch([searchTerm,dateRange],()=>{page.value=1;expanded.value=null})

</script>

<template>
<div class="screen-training bg-background font-body-md text-on-surface antialiased selection:bg-primary selection:text-on-primary">
  <!-- 侧边栏 -->


  <div class=" flex flex-col min-h-screen">
    <!-- 顶部导航 -->


    <main class="w-full pt-0 bg-background pb-20 lg:pb-12">
      <div class="fz-page-inset max-w-7xl mx-auto">
        <div class="fz-section-stack flex flex-col">
          <!-- 标题区 -->
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-space-md">
            <div class="flex flex-col gap-1">
              <div class="flex items-center gap-space-sm">
                <h1 class="font-headline-lg text-headline-lg text-on-surface font-bold tracking-tight">训练记录</h1>
                <span class="font-label-sm text-label-sm px-2.5 py-0.5 rounded-full bg-surface-container text-primary font-semibold flex items-center gap-1">
                  <span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>实时数据
                </span>
              </div>
              <p class="font-body-md text-body-md text-on-surface-variant">本周期累计总容量 <span class="text-primary font-medium">{{fmt(weekVolume)}} kg</span> · 根据本周已完成训练组汇总</p>
            </div>
            <div class="flex items-center gap-2 self-start md:self-auto">
              <button @click.stop="exportRecords" class="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface text-label-md font-label-md transition-all">
                <span class="material-symbols-outlined text-[16px]">download</span><span>导出训练日志</span>
              </button>
              <button @click.stop="openWorkout" class="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary-container text-on-primary-container hover:brightness-110 active:scale-[0.98] text-label-md font-label-md font-semibold transition-all shadow-sm">
                <span class="material-symbols-outlined text-[18px]">add</span><span>快速记录动作 (+)</span>
              </button>
            </div>
          </div>

          <!-- 三个统计卡片 -->
          <div class="grid grid-cols-1 md:grid-cols-3 fz-module-grid">
            <!-- 卡片1：训练频次 -->
            <div class="bg-surface-container-lowest rounded-2xl p-space-lg flex flex-col justify-between card-lift shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">
              <div class="flex items-start justify-between">
                <div class="flex flex-col">
                  <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">周期训练频次</span>
                  <span class="font-body-sm text-body-sm text-on-surface-variant mt-0.5">每周达标率监测</span>
                </div>
                <div class="w-10 h-10 rounded-xl bg-surface-container flex items-center justify-center text-primary">
                  <span class="material-symbols-outlined text-[20px]">calendar_today</span>
                </div>
              </div>
              <div class="my-space-md flex items-baseline gap-2">
                <span class="font-display-lg text-display-lg text-on-surface font-bold tracking-tight">{{weekDays}}</span>
                <span class="font-body-md text-body-md text-on-surface-variant">/ 目标 {{ui.profile.available_days??'—'}} 天</span>
              </div>
              <div class="flex flex-col gap-1.5">
                <div class="flex justify-between font-label-sm text-label-sm text-on-surface-variant">
                  <span>周目标达成度</span>
                  <span class="font-medium text-primary">{{fmt(pct(weekDays,ui.profile.available_days))}}%</span>
                </div>
                <div class="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                  <div class="h-full bg-primary rounded-full transition-all duration-700" :style="{width: (bar(weekDays,ui.profile.available_days)||0)+'%'}"></div>
                </div>
              </div>
            </div>

            <!-- 卡片2：总组数 -->
            <div class="bg-surface-container-lowest rounded-2xl p-space-lg flex flex-col justify-between card-lift shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">
              <div class="flex items-start justify-between">
                <div class="flex flex-col">
                  <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">累计总组数</span>
                  <span class="font-body-sm text-body-sm text-on-surface-variant mt-0.5">负荷刺激强度</span>
                </div>
                <div class="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-on-surface-variant">
                  <span class="material-symbols-outlined text-[20px]">format_list_numbered</span>
                </div>
              </div>
              <div class="my-space-md flex items-baseline gap-2">
                <span class="font-display-lg text-display-lg text-on-surface font-bold tracking-tight">{{weekSets}}</span>
                <span class="font-body-md text-body-md text-on-surface-variant">组完成</span>
              </div>
            </div>

            <!-- 卡片3：总容量 -->
            <div class="bg-surface-container-lowest rounded-2xl p-space-lg flex flex-col justify-between card-lift shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">
              <div class="flex items-start justify-between">
                <div class="flex flex-col">
                  <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">累计总负荷容量</span>
                  <span class="font-body-sm text-body-sm text-on-surface-variant mt-0.5">绝对做功总量</span>
                </div>
                <div class="w-10 h-10 rounded-xl bg-surface-container flex items-center justify-center text-primary">
                  <span class="material-symbols-outlined text-[20px]">fitness_center</span>
                </div>
              </div>
              <div class="my-space-md flex items-baseline gap-2">
                <span class="font-display-lg text-display-lg text-primary font-bold tracking-tight">{{fmt(weekVolume/1000,1)}}</span>
                <span class="font-body-md text-body-md text-on-surface-variant">吨 ({{fmt(weekVolume)}} kg)</span>
              </div>
            </div>
          </div>

          <!-- 搜索与日期筛选 -->
          <div class="bg-surface-container-lowest rounded-2xl border border-outline-variant/30 p-3 flex flex-col lg:flex-row gap-3">
            <label class="flex-1 min-w-0 flex items-center gap-2 px-3 py-2 rounded-xl bg-surface-container border border-outline-variant/30 focus-within:border-primary/40 focus-within:bg-primary/[.04] text-on-surface-variant">
              <span class="material-symbols-outlined text-[18px]">search</span>
              <input v-model="searchTerm" aria-label="搜索训练记录" placeholder="搜索动作名称或备注（如：卧推、深蹲、背部）…" class="w-full bg-transparent outline-none text-on-surface placeholder:text-on-surface-variant text-body-sm">
            </label>
            <label class="flex items-center gap-2 px-3 py-2 rounded-xl bg-surface-container border border-outline-variant/30 focus-within:border-primary/40 focus-within:bg-primary/[.04] text-on-surface-variant lg:min-w-[180px]">
              <span class="material-symbols-outlined text-[18px] text-primary">calendar_month</span>
              <span class="text-label-sm whitespace-nowrap">日期</span>
              <select v-model="dateRange" aria-label="日期筛选" class="min-w-0 flex-1 bg-transparent outline-none text-on-surface text-body-sm">
                <option value="all">全部日期</option>
                <option value="7">最近 7 天</option>
                <option value="30">最近 30 天</option>
                <option value="90">最近 90 天</option>
                <option value="365">最近 1 年</option>
              </select>
            </label>
            <button v-if="searchTerm||dateRange!=='30'" @click.stop="searchTerm='';dateRange='30'" class="px-3 py-2 rounded-lg text-label-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors">清除筛选</button>
          </div>

          <!-- 记录列表 -->
          <!-- 记录列表 -->
          <div class="training-list-shell" id="training-card-list">
            <div v-if="!filteredRows.length" class="training-empty-state bg-surface-container-lowest rounded-xl border border-outline-variant/30 text-on-surface-variant">
              <span class="material-symbols-outlined text-[42px]">fitness_center</span>
              <p v-if="rows.length">当前筛选范围没有训练记录。<button @click.stop="searchTerm='';dateRange='30'" class="text-primary ml-1">清除筛选</button></p>
              <p v-else>暂无训练记录 <button @click.stop="openWorkout" class="text-primary ml-1">添加训练</button></p>
            </div>
            <template v-for="(row,idx) in pageRows" :key="row.key">
              <article class="training-row" :class="{ 'training-row-expanded': expanded===row.key }">
                <div class="training-row-head fz-data-row">
                  <button type="button" class="training-date" @click.stop="expanded=(expanded===row.key?'':row.key)" aria-label="展开训练详情">
                    <span>{{relativeDay(row)}}</span><strong>{{shortDate(row)}}</strong>
                  </button>
                  <button type="button" class="training-action" @click.stop="expanded=(expanded===row.key?'':row.key)">
                    <div class="training-action-title"><strong>{{row.name||'训练动作'}}</strong><span class="training-muscle">{{muscleGroup(row.name)}}</span><span v-if="rowStatus(row)==='新纪录'" class="training-pr-badge">PR 新纪录 <span>🏆</span></span><span v-else-if="rowStatus(row)==='已完成'" class="training-done-badge">目标达成</span></div>
                    <small>{{row.note||'已保存训练动作记录'}}</small>
                  </button>
                  <div class="training-peak"><strong>{{cardioRow(row)?fmt(cardioTotal(row,'duration_minutes'),1):fmt(row.peak?.weight_kg)}}</strong><span>{{cardioRow(row)?'分钟':'kg'}}</span><small>{{cardioRow(row)?fmt(cardioTotal(row,'distance_km'),2)+' km':'× '+(row.peak?.reps||rowReps(row)||'—')+' 次'}}</small></div>
                  <div class="training-set-chips"><span v-for="(set,i) in row.sets" :key="i" :class="{'training-chip-active':set.completed!==false&&i===row.sets.length-1}">{{set.duration_minutes!=null?fmt(set.distance_km,2)+' km · '+fmt(set.duration_minutes,1)+' 分钟':fmt(set.weight_kg)+'kg × '+set.reps}}</span></div>
                  <span class="training-status" :class="rowStatusClass(row)">{{rowStatus(row)}}</span>
                  <div class="training-row-actions"><button type="button" class="training-expand" aria-label="展开训练详情" @click.stop="expanded=(expanded===row.key?'':row.key)"><span>{{expanded===row.key?'收起':'展开'}}</span><span class="material-symbols-outlined">expand_more</span></button>
                   <DeleteButton data-training-delete :action="()=>deleteAction(row)" :subject="'动作 '+row.name" :disabled="deleting" compact />
                  </div>
                </div>
                <div v-if="expanded===row.key" class="training-detail">
                  <div class="training-detail-top"><div><span class="training-detail-dot"></span><strong>逐组执行明细与完成状况</strong><span class="training-count">{{detailSummary(row)}}</span></div><div class="training-detail-meta"><span>● {{rowStatus(row)==='部分完成'?'仍有未完成组':'完成状态已记录'}}</span><span>◎ 数据来自训练记录</span></div></div>
                  <div class="training-detail-table"><div class="training-detail-head"><span>组序</span><span>{{cardioRow(row)?'距离（km）':'负荷重量'}}</span><span>{{cardioRow(row)?'时间（分钟）':'有效次数'}}</span><span>体感反馈</span></div><div v-for="(set,i) in row.sets" :key="i" class="training-detail-row fz-data-row"><span>#{{Number(i)+1}}</span><span>{{set.duration_minutes!=null?fmt(set.distance_km,2)+' km':fmt(set.weight_kg)+' kg'}}</span><span>{{set.duration_minutes!=null?fmt(set.duration_minutes,1)+' 分钟':set.reps+' 次'}}</span><span>{{difficultyFeedback(set.difficulty)}}</span></div></div>
                  <div class="training-detail-actions"><button @click.stop="editRecord(row.record)" class="fz-button">编辑这条训练</button></div>
                </div>
              </article>
            </template>
            <div v-if="filteredRows.length" class="training-pagination flex flex-col sm:flex-row items-center justify-between gap-3 pt-3"><span class="text-on-surface-variant font-body-sm text-body-sm">显示第 {{(page-1)*6+1}} 至 {{Math.min(page*6,filteredRows.length)}} 条，共 {{filteredRows.length}} 条动作记录</span><div class="flex items-center gap-1.5"><button :disabled="page<=1" @click.stop="page--" class="px-3 py-1.5 rounded-lg bg-surface-container text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors font-label-md text-label-md disabled:opacity-40 disabled:cursor-not-allowed">上一页</button><button :disabled="page>=pages" @click.stop="page++" class="px-3 py-1.5 rounded-lg bg-surface-container text-on-surface hover:bg-surface-container-high transition-colors font-label-md text-label-md disabled:opacity-40 disabled:cursor-not-allowed">下一页</button></div></div>
          </div>
        </div>
      </div>
    </main>

    <!-- 手机端底部导航 -->

  </div>
</div>
</template>

<style scoped>
html, .screen-training { margin: 0; padding: 0; }
.screen-training { overscroll-behavior: none; }
main > :first-child { margin-top: 0 !important; }
main > :last-child { margin-bottom: 0 !important; }
::-webkit-scrollbar { display: none; }

.training-list-shell{display:flex;flex-direction:column;gap:0}.training-row{background:var(--fz-surface-container-lowest);border:1px solid var(--fz-outline-variant);border-radius:0;overflow:hidden;color:var(--fz-on-surface)}.training-row-expanded{border-color:var(--fz-outline-variant)}.training-row-head{display:grid;grid-template-columns:58px minmax(160px,1.25fr) 112px minmax(120px,1.15fr) 70px 112px;align-items:center;gap:12px;height:var(--fz-table-row-height);min-height:var(--fz-table-row-height);padding:0 16px}.training-date{height:44px;width:58px;border:1px solid var(--fz-outline-variant);border-radius:10px;background:var(--fz-surface-card);box-shadow:0 1px 3px rgb(var(--fz-rgb-shadow) / .06);color:var(--fz-on-surface);display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer}.training-date span{font-size:10px;color:var(--fz-primary);font-weight:700;line-height:1.1}.training-date strong{font-size:15px;line-height:1.2;color:var(--fz-on-surface)}.training-action{border:0;background:transparent;color:inherit;text-align:left;min-width:0;padding:0;cursor:pointer}.training-action-title{display:flex;align-items:center;gap:7px;flex-wrap:nowrap;overflow:hidden}.training-action-title strong{font-size:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.training-action small{display:block;margin-top:3px;color:var(--fz-on-surface-variant);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.training-muscle{font-size:10px;color:var(--fz-on-surface-variant)}.training-pr-badge,.training-done-badge{font-size:10px;border-radius:5px;padding:3px 7px;font-weight:700}.training-pr-badge{color:var(--fz-primary);background:rgb(var(--fz-rgb-primary) / .16)}.training-done-badge{color:var(--fz-primary);background:rgb(var(--fz-rgb-primary) / .11)}.training-peak{display:flex;align-items:baseline;gap:5px;white-space:nowrap}.training-peak strong{font-size:30px;line-height:1;color:var(--fz-on-surface)}.training-row-expanded .training-peak strong{color:var(--fz-primary)}.training-peak span{font-size:11px}.training-peak small{font-size:11px;color:var(--fz-on-surface)}.training-set-chips{display:flex;flex-wrap:nowrap;gap:6px;min-width:0;overflow-x:auto}.training-set-chips span{font-size:10px;color:var(--fz-on-surface);background:var(--fz-surface-container-low);border:1px solid var(--fz-surface-container-high);border-radius:4px;padding:5px 7px;white-space:nowrap}.training-set-chips .training-chip-active{color:var(--fz-primary);border-color:var(--fz-primary);background:rgb(var(--fz-rgb-primary) / .11)}.training-status{justify-self:end;font-size:10px;border-radius:10px;padding:4px 8px;white-space:nowrap;font-weight:700}.training-status-pr,.training-status-done{color:var(--fz-primary);background:rgb(var(--fz-rgb-primary) / .12)}.training-status-muted{color:var(--fz-on-surface-variant);background:var(--fz-surface-container)}.training-expand{border:0;background:transparent;color:var(--fz-on-surface);display:flex;align-items:center;justify-content:center;gap:1px;font-size:10px;cursor:pointer}.training-expand .material-symbols-outlined{font-size:18px;transition:transform .2s}.training-row-expanded .training-expand .material-symbols-outlined{transform:rotate(180deg)}.training-detail{margin:0 16px 16px;background:var(--fz-surface-card);border:1px solid var(--fz-outline-variant);border-radius:12px;padding:14px 16px;box-shadow:0 1px 4px rgb(var(--fz-rgb-shadow) / .05)}.training-detail-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px;font-size:11px}.training-detail-top>div:first-child{display:flex;align-items:center;gap:7px}.training-detail-dot{width:7px;height:7px;border-radius:50%;background:var(--fz-primary)}.training-count{font-size:10px;color:var(--fz-on-surface);background:var(--fz-surface-container-high);padding:3px 7px;border-radius:4px}.training-detail-meta{display:flex;gap:14px;color:var(--fz-primary);font-size:10px}.training-detail-table{overflow:hidden;border:1px solid var(--fz-surface-container-high);border-radius:6px}.training-detail-head,.training-detail-row{display:grid;grid-template-columns:72px 1.2fr 1fr 1fr;gap:10px;align-items:center;padding:8px 10px;font-size:11px}.training-detail-head{background:var(--fz-surface-container-low);color:var(--fz-on-surface-variant);font-size:10px}.training-detail-row{height:var(--fz-table-row-height);padding-block:0;background:var(--fz-table-row-odd);border-top:1px solid var(--fz-surface-container-high);color:var(--fz-on-surface)}.training-detail-row:nth-child(odd){background:var(--fz-table-row-even)}.training-detail-row:hover{background:var(--fz-surface-container)}.training-set-complete,.training-set-incomplete{font-size:10px;padding:3px 7px;border-radius:8px;background:var(--fz-outline-variant);color:var(--fz-on-surface)}.training-detail-actions{display:flex;justify-content:flex-end;margin-top:10px}@media(max-width:1150px){.training-row-head{grid-template-columns:58px minmax(0,1fr) 90px 112px;gap:8px;padding:0 12px}.training-set-chips,.training-status,.training-peak small{display:none}.training-peak{justify-self:end}.training-detail-head,.training-detail-row{grid-template-columns:46px 1.2fr 1fr 1fr}}@media(max-width:560px){.training-date{width:48px}.training-row-head{grid-template-columns:48px minmax(0,1fr) 70px}.training-peak{display:none}.training-action-title strong{font-size:14px}.training-action small{font-size:10px}.training-detail-meta{display:none}}
.training-row-actions{display:flex;gap:6px;align-items:center}.training-row-actions>.training-expand{height:32px;min-width:40px;flex:1}

.training-row:first-of-type{border-radius:12px 12px 0 0}
.training-row:last-of-type{border-radius:0 0 12px 12px}
.training-row:only-of-type{border-radius:12px}
.training-row + .training-row{border-top:0}
.training-row:nth-of-type(even) .training-row-head{background:var(--fz-table-row-even)}
.training-row-head:hover{background:var(--fz-surface-container)}
.training-pagination{margin-top:var(--fz-module-gap);padding-top:0}
.training-empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:180px;padding:32px 24px;gap:12px;text-align:center}
.training-empty-state > .material-symbols-outlined{display:block;line-height:1}
.training-empty-state > p{margin:0}
@media(max-width:560px){
  .training-date strong{font-size:14px;white-space:nowrap}
  .training-muscle,.training-pr-badge,.training-done-badge{display:none}
  .training-row-actions>.training-expand{min-width:32px}
  .training-row-actions>.training-expand>span:not(.material-symbols-outlined){display:none}
  .training-detail{padding:12px;margin:0 12px 12px}
  .training-detail-top>div:first-child{flex-wrap:wrap}
  .training-detail-head,.training-detail-row{grid-template-columns:32px 1fr 1fr 1fr;gap:4px;padding-inline:8px}
}
@media (prefers-reduced-motion: reduce) {
  .card-lift, .card-lift:hover { transition: none; transform: none; }
}
</style>
