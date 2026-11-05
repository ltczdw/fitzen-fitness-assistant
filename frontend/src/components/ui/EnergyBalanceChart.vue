<script setup lang="ts">
import {computed} from 'vue'
const props=defineProps<{daily:any[]}>()
const scale=computed(()=>Math.max(1,...props.daily.flatMap(d=>[d.calories??0,d.tdee??0])))
const known=computed(()=>props.daily.filter(d=>d.calories!==null))
const average=computed(()=>known.value.length?Math.round(known.value.reduce((n,d)=>n+d.calories,0)/known.value.length).toLocaleString('zh-CN'):'—')
const days=computed(()=>[...props.daily].sort((a,b)=>String(b?.date||'').localeCompare(String(a?.date||''))).map(d=>({...d,label:d.date.slice(5).replace('-','/'),intakeHeight:d.calories===null?0:d.calories/scale.value*173,expenditureHeight:d.tdee===null?0:d.tdee/scale.value*173})))
</script>
<template>
<section class="energy-balance" aria-labelledby="energy-balance-title">
<header class="energy-heading"><div class="energy-heading-copy"><span class="energy-status">已记录 {{known.length}} / {{daily.length}} 天摄入</span><h2 id="energy-balance-title">热量摄入对比每日总消耗</h2></div>
<div class="energy-summaries"><div class="energy-summary"><span class="energy-swatch energy-swatch-intake"></span><div><span class="energy-label">记录日均摄入</span><strong>{{average}} kcal</strong></div></div><div class="energy-summary"><span class="energy-swatch energy-swatch-expenditure"></span><div><span class="energy-label">日均消耗 (TDEE)</span><strong class="energy-green">暂无数据</strong></div></div><div class="energy-summary"><span class="material-symbols-outlined energy-flame">local_fire_department</span><div><span class="energy-label">平均赤字差值</span><strong class="energy-pink">— kcal/天</strong></div></div></div></header>
<div class="energy-scroll"><div class="energy-chart" :style="{gridTemplateColumns:'repeat('+Math.max(days.length,1)+',minmax(56px,1fr))'}" role="img" :aria-label="'近 '+days.length+' 天热量摄入；尚无总消耗数据'"><div v-for="day in days" :key="day.date" class="energy-day" :class="{'energy-day-highlight':day.date===days[0]?.date}" :title="day.date+'：'+(day.calories===null?'未记录':day.calories+' kcal')"><div class="energy-bars"><span v-if="day.calories!==null" class="energy-bar energy-bar-intake" :style="{height:day.intakeHeight+'px'}"></span><span v-if="day.tdee!==null" class="energy-bar energy-bar-expenditure" :style="{height:day.expenditureHeight+'px'}"></span></div><div class="energy-day-caption"><span class="energy-difference">{{day.calories===null?'—':Math.round(day.calories)}}</span><span class="energy-date">{{day.label}}</span></div></div></div></div>
<footer class="energy-footer"><span class="energy-model-summary"><span class="material-symbols-outlined energy-verified">info</span><span>柱状图显示已记录摄入（kcal）。尚未记录每日消耗，暂不计算热量赤字和脂肪变化。</span></span><span class="energy-model-version">实际饮食记录</span></footer>
</section>
</template>
<style scoped>
.energy-balance {
  padding: 24px;
  margin-bottom: 32px;
  border-radius: 16px;
  background: var(--fz-surface-container-lowest);
  color: var(--fz-on-surface);
  font-family: Inter, "Microsoft YaHei", sans-serif;
  font-variant-numeric: tabular-nums;
}
.energy-heading { display: flex; align-items: center; justify-content: space-between; gap: 24px; min-height: 62px; }
.energy-heading-copy { flex-shrink: 0; }
.energy-status { display: inline-block; padding: 3px 9px; border-radius: 5px; background: rgb(var(--fz-rgb-primary) / .1); color: var(--fz-primary); font-size: 13px; line-height: 18px; font-weight: 600; }
.energy-heading h2 { margin: 5px 0 0; font-size: 24px; line-height: 30px; font-weight: 700; letter-spacing: -.02em; }
.energy-summaries { display: flex; align-items: center; gap: 18px; }
.energy-summary { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 10px; background: var(--fz-surface-container-low); white-space: nowrap; }
.energy-label { display: block; color: var(--fz-on-surface-variant); font-size: 13px; line-height: 17px; font-weight: 600; }
.energy-summary strong { display: block; font-size: 16px; line-height: 20px; font-weight: 700; }
.energy-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
.energy-swatch-intake { background: var(--fz-outline); }
.energy-swatch-expenditure { background: var(--fz-primary); box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06); }
.energy-green { color: var(--fz-primary); }
.energy-pink,.energy-flame { color: var(--fz-tertiary); }
.energy-flame { font-size: 21px; }
.energy-chart { position: relative; display: grid; grid-template-columns: repeat(7,minmax(0,1fr)); gap: 20px; margin-top: 64px; }
.energy-chart::after { content: ""; position: absolute; bottom: 10px; left: 0; right: 0; height: 1px; background: rgb(var(--fz-rgb-outline-variant) / .23); pointer-events: none; }
.energy-day { min-width: 0; display: flex; flex-direction: column; align-items: center; }
.energy-bars { height: 204px; display: flex; align-items: flex-end; justify-content: center; gap: 7px; }
.energy-bar { display: block; width: 19px; border-radius: 3px 3px 0 0; }
.energy-bar-intake { background: var(--fz-outline); }
.energy-bar-expenditure { background: var(--fz-primary); box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06); }
.energy-day-highlight .energy-bar-expenditure { background: var(--fz-primary); box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06); }
.energy-day-caption { display: flex; flex-direction: column; align-items: center; height: 46px; padding-top: 8px; gap: 3px; }
.energy-difference { padding: 3px 7px; color: var(--fz-tertiary); background: rgb(var(--fz-rgb-tertiary) / .1); border-radius: 4px; font-size: 13px; line-height: 17px; font-weight: 600; }
.energy-day-highlight .energy-difference { box-shadow: inset 0 0 0 1px rgb(var(--fz-rgb-tertiary) / .2); }
.energy-date { color: var(--fz-on-surface-variant); font-size: 13px; line-height: 16px; font-weight: 600; }
.energy-day-highlight .energy-date { color: var(--fz-primary); font-weight: 700; }
.energy-footer { min-height: 38px; display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 8px 14px; border-radius: 10px; background: var(--fz-background); color: var(--fz-on-surface-variant); font-size: 13px; line-height: 20px; font-weight: 600; }
.energy-model-summary { display: flex; align-items: center; gap: 8px; }
.energy-verified { font-size: 17px; color: var(--fz-primary); flex-shrink: 0; }
.energy-model-version { white-space: nowrap; }
@media (max-width: 1200px) {
  .energy-heading { align-items: flex-start; flex-direction: column; gap: 16px; }
  .energy-summaries { flex-wrap: wrap; gap: 12px; }
  .energy-chart { margin-top: 36px; }
  .energy-footer { align-items: flex-start; }
}
@media (max-width: 600px) {
  .energy-balance { padding: 16px; }
  .energy-heading h2 { font-size: 20px; line-height: 28px; }
  .energy-summaries { gap: 8px; width: 100%; }
  .energy-summary { padding: 8px; gap: 6px; flex: 1 1 auto; }
  .energy-label { font-size: 11px; }
  .energy-summary strong { font-size: 13px; }
  .energy-chart { gap: 4px; margin-top: 20px; }
  .energy-bars { gap: 4px; height: 190px; }
  .energy-bar { width: 12px; }
  .energy-difference { font-size: 11px; padding: 3px 4px; }
  .energy-date { font-size: 11px; }
  .energy-footer { flex-direction: column; gap: 4px; font-size: 11px; line-height: 18px; padding: 10px; }
  .energy-model-version { align-self: flex-end; }
}
.energy-scroll{overflow-x:auto}.energy-chart{padding-bottom:10px}.energy-difference{color:var(--fz-on-surface-variant);background:var(--fz-surface-container)}.energy-chart{gap:10px}
</style>
