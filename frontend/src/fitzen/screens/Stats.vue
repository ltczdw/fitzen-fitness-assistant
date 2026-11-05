<script setup lang="ts">
import {computed,ref} from 'vue'
import {EnergyBalanceChart} from '../../components/ui'
import {useFitzen} from '../state'
import {daily,fmt,bar} from '../metrics'
const {ui}=useFitzen()
const period=ref(30)
const periodDays=computed(()=>daily.value.slice(-period.value))
const volume=computed(()=>periodDays.value.reduce((n,d)=>n+d.training_volume,0))
const buckets=computed(()=>Array.from({length:7},(_,i)=>{const rows=periodDays.value.slice(Math.floor(i*periodDays.value.length/7),Math.floor((i+1)*periodDays.value.length/7));return {date:rows[0]?.date||'',label:rows[0]?.date.slice(5)||'—',value:rows.reduce((n,d)=>n+d.training_volume,0)}}))
const peak=(pattern:string)=>{const items=(ui.data?.exercise_bests||[]).filter((r:any)=>r.exercise.includes(pattern));return items.length?Math.max(...items.map((r:any)=>r.weight_kg)):null}

</script>
<template>
<div class="screen-stats bg-background font-body-md text-on-surface antialiased selection:bg-primary selection:text-on-primary"><div class=" flex flex-col min-h-screen"><main class="w-full pt-0 bg-background pb-20 lg:pb-12"><div class="fz-page-inset max-w-7xl mx-auto"><div class="flex flex-col w-full">
<!-- Subtle ambient glow background element -->
<div class="relative w-full">
<div class="absolute -top-24 right-1/4 w-96 h-96 bg-surface-container rounded-full blur-3xl pointer-events-none -z-10"></div>
<div class="absolute top-96 left-10 w-80 h-80 bg-secondary-container/5 rounded-full blur-3xl pointer-events-none -z-10"></div>
<!-- Header & Range Pill Switcher -->
<header class="flex flex-col md:flex-row md:items-end justify-between gap-space-md mb-space-xl">
<div class="flex flex-col gap-1">
<div class="flex items-center gap-space-xs text-on-surface-variant font-label-sm text-label-sm uppercase tracking-widest"><span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span><span class="">实时体征与训练负荷分析</span></div>
<h1 class="font-headline-lg text-headline-lg font-semibold text-on-surface tracking-tight">
          数据统计与趋势分析
        </h1>
<p class="font-body-sm text-body-sm text-on-surface-variant"> 依据近 {{period}} 天已保存的训练和饮食记录汇总 </p>
</div>
<!-- Time Range Pill Switcher -->
<div class="inline-flex items-center p-1 rounded-full bg-surface-container-high shadow-inner"><button @click.stop="period=7" :class="period===7?'bg-primary text-on-primary font-medium':'text-on-surface-variant'" class="time-range-btn px-4 py-1.5 rounded-full font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-all duration-200" type="button">周</button><button @click.stop="period=30" :class="period===30?'bg-primary text-on-primary font-medium':'text-on-surface-variant'" class="time-range-btn px-4 py-1.5 rounded-full font-label-md text-label-md  transition-all duration-200" type="button">月</button><button @click.stop="period=365" :class="period===365?'bg-primary text-on-primary font-medium':'text-on-surface-variant'" class="time-range-btn px-4 py-1.5 rounded-full font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-all duration-200" type="button">年</button></div>
</header>
<!-- Row 1: Dual Macro Charts (1:1 Layout) -->
<section class="grid grid-cols-1 lg:grid-cols-2 gap-space-lg mb-space-xl">
<!-- Weight & Body Fat Trend Chart Card -->
<article class="bg-surface-container-lowest rounded-xl p-space-md lg:p-space-lg shadow-xl relative overflow-hidden flex flex-col justify-between group hover:bg-surface-container transition-colors duration-300">
<div>
<div class="flex items-start justify-between gap-space-sm mb-space-sm">
<div class="flex flex-col">

<h2 class="font-headline-sm text-headline-sm text-on-surface font-semibold">体重与体脂率变化趋势</h2>
</div>
<!-- Legend -->
<div class="flex items-center gap-space-md bg-surface-container-lowest px-3 py-1.5 rounded-lg shadow-sm">
<div class="flex items-center gap-1.5">
<span class="w-3 h-0.5 rounded-full bg-primary shadow-sm"></span>
<span class="font-label-sm text-label-sm text-on-surface">体重 (kg)</span>
</div>
<div class="flex items-center gap-1.5">
<span class="w-3 h-0.5 rounded-full bg-info border-b border-dashed border-info"></span>
<span class="font-label-sm text-label-sm text-on-surface-variant">体脂率 (%)</span>
</div>
</div>
</div>
<!-- Key Metrics Headline in Card -->
<div class="flex flex-wrap items-baseline gap-x-space-lg gap-y-1 mb-space-md font-body-sm text-body-sm">
<div class="flex items-baseline gap-2">
<span class="font-metric-display text-metric-display text-on-surface font-semibold tabular-nums">{{fmt(ui.profile.body_weight_kg,1)}}</span>
<span class="font-label-md text-label-md text-on-surface-variant">kg</span>
<span class="inline-flex items-center text-primary font-code-inline text-code-inline bg-surface-container px-1.5 py-0.5 rounded">
<span class="material-symbols-outlined text-[14px]">trending_down</span> 暂无变化记录 </span>
</div>
<div class="flex items-baseline gap-2">
<span class="font-headline-sm text-headline-sm text-on-surface font-semibold tabular-nums">—</span>
<span class="font-label-md text-label-md text-on-surface-variant">%</span>
<span class="inline-flex items-center text-info font-code-inline text-code-inline bg-info/10 px-1.5 py-0.5 rounded">
<span class="material-symbols-outlined text-[14px]">trending_down</span> — </span>
</div>
</div>
</div>
<!-- SVG Line Chart (30-day progression) -->
<div class="w-full relative mt-2">
<svg class="w-full h-48 lg:h-56 overflow-visible select-none" preserveAspectRatio="none" viewBox="0 0 540 210"><line stroke="var(--fz-outline-variant)" x1="0" x2="540" y1="170" y2="170"/><text fill="var(--fz-on-surface-variant)" text-anchor="middle" x="270" y="100" font-size="13">暂无体重与体脂历史记录</text></svg>
</div>
</article>
<!-- Training Load & Volume Bar Chart Card -->
<article class="bg-surface-container-lowest rounded-xl p-space-md lg:p-space-lg shadow-xl relative overflow-hidden flex flex-col justify-between group hover:bg-surface-container transition-colors duration-300">
<div>
<div class="flex items-start justify-between gap-space-sm mb-space-sm">
<div class="flex flex-col">

<h2 class="font-headline-sm text-headline-sm text-on-surface font-semibold">训练负荷与容量统计</h2>
</div>
<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-surface-container text-primary font-label-sm text-label-sm font-medium">
<span class="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></span> 已完成组汇总 </span>
</div>
<div class="flex items-baseline gap-space-sm mb-space-md">
<span class="font-metric-display text-metric-display text-on-surface font-semibold tabular-nums">{{fmt(volume)}}</span>
<span class="font-label-md text-label-md text-on-surface-variant">kg 累积容量</span>
<span class="text-primary font-code-inline text-code-inline ml-2 flex items-center">
<span class="material-symbols-outlined text-[14px]">arrow_upward</span> {{periodDays.filter(d=>d.training_records).length}} 天训练 </span>
</div>
</div>
<!-- 7-Day Precision Segmented Bars -->
<div class="w-full flex flex-col justify-end pt-4">
<div class="grid grid-cols-7 gap-2 sm:gap-4 items-end h-44 pb-2 px-1"><div v-for="bucket in buckets" :key="bucket.date" class="flex flex-col items-center gap-2 h-full justify-end group/bar cursor-pointer">
<span class="font-code-inline text-code-inline text-on-surface-variant opacity-0 group-hover/bar:opacity-100 transition-opacity">{{fmt(bucket.value)}}</span>
<div class="w-full max-w-[34px] bg-primary rounded-t-lg transition-all duration-300 group-hover/bar:bg-primary-container" :style="{height:bar(bucket.value,Math.max(1,...buckets.map(b=>b.value)))+'%'}"></div>
<span class="font-label-sm text-label-sm text-on-surface-variant">{{bucket.label}}</span>
</div></div>
</div>
</article>
</section>
<!-- Row 2: Caloric Balance & TDEE Comparison -->
<EnergyBalanceChart :daily="periodDays" />
<!-- Row 3: Core Athletic Index Cards (3 Columns) -->
<section class="grid grid-cols-1 md:grid-cols-3 gap-space-lg">
<!-- 1. Strength Index Card -->
<article class="bg-surface-container-lowest rounded-xl p-space-md lg:p-space-lg shadow-xl relative overflow-hidden flex flex-col justify-between hover:bg-surface-container transition-colors duration-200">
<div>
<div class="flex items-center justify-between mb-space-sm">

<span class="font-code-inline text-code-inline text-primary bg-surface-container px-2 py-0.5 rounded flex items-center gap-0.5">
<span class="material-symbols-outlined text-[13px]">arrow_upward</span> 暂无评分 </span>
</div>
<h3 class="font-headline-sm text-headline-sm text-on-surface font-semibold mb-space-md">力量进步指标</h3>
<div class="flex items-baseline gap-2 mb-space-md">
<span class="font-display-lg text-display-lg font-bold text-on-surface tabular-nums">—</span>
<span class="font-body-md text-body-md text-on-surface-variant">/ 100</span>
</div>
<!-- 1RM Progress Bars -->
<div class="flex flex-col gap-3">
<!-- Bench Press -->
<div class="flex flex-col gap-1">
<div class="flex justify-between font-label-md text-label-md">
<span class="text-on-surface">卧推最高负重</span>
<span class="text-on-surface tabular-nums font-semibold">{{fmt(peak('卧推'))}} kg <span class="text-primary font-normal text-xs"></span></span>
</div>
<div class="h-1.5 w-full bg-surface-container-highest rounded-full overflow-hidden">
<div class="h-full bg-primary rounded-full" style="width:0%"></div>
</div>
</div>
<!-- Squat -->
<div class="flex flex-col gap-1">
<div class="flex justify-between font-label-md text-label-md">
<span class="text-on-surface">深蹲最高负重</span>
<span class="text-on-surface tabular-nums font-semibold">{{fmt(peak('深蹲'))}} kg <span class="text-primary font-normal text-xs"></span></span>
</div>
<div class="h-1.5 w-full bg-surface-container-highest rounded-full overflow-hidden">
<div class="h-full bg-primary rounded-full" style="width:0%"></div>
</div>
</div>
<!-- Deadlift -->
<div class="flex flex-col gap-1">
<div class="flex justify-between font-label-md text-label-md">
<span class="text-on-surface">硬拉最高负重</span>
<span class="text-on-surface tabular-nums font-semibold">{{fmt(peak('硬拉'))}} kg <span class="text-primary font-normal text-xs"></span></span>
</div>
<div class="h-1.5 w-full bg-surface-container-highest rounded-full overflow-hidden">
<div class="h-full bg-primary rounded-full" style="width:0%"></div>
</div>
</div>
</div>
</div>
<div class="mt-space-md pt-space-sm bg-surface-container-lowest/60 -mx-space-md -mb-space-md p-space-md flex items-center justify-between">
<span class="font-label-sm text-label-sm text-on-surface-variant">三项总和系数</span>
<span class="font-code-inline text-code-inline text-on-surface font-semibold">暂无综合评分</span>
</div>
</article>
<!-- 2. VO2 Max & Endurance Card -->
<article class="bg-surface-container-lowest rounded-xl p-space-md lg:p-space-lg shadow-xl relative overflow-hidden flex flex-col justify-between hover:bg-surface-container transition-colors duration-200">
<div>
<div class="flex items-center justify-between mb-space-sm">

<span class="font-label-sm text-label-sm text-info bg-info/10 px-2 py-0.5 rounded">暂无监测数据</span>
</div>
<h3 class="font-headline-sm text-headline-sm text-on-surface font-semibold mb-space-md">耐力与心肺水平</h3>
<div class="flex items-baseline gap-2 mb-space-md">
<span class="font-display-lg text-display-lg font-bold text-on-surface tabular-nums">—</span>
<span class="font-body-md text-body-md text-on-surface-variant">/ 100</span>
</div>
<div class="grid grid-cols-2 gap-space-sm mb-space-md">
<div class="bg-surface-container p-3 rounded-lg flex flex-col">
<span class="font-label-sm text-label-sm text-on-surface-variant mb-1">估算 VO2 Max</span>
<span class="font-headline-sm text-headline-sm text-info font-bold tabular-nums">—</span>
<span class="font-code-inline text-code-inline text-on-surface-variant text-[11px]">ml/kg/min</span>
</div>
<div class="bg-surface-container p-3 rounded-lg flex flex-col">
<span class="font-label-sm text-label-sm text-on-surface-variant mb-1">静息心率 (RHR)</span>
<span class="font-headline-sm text-headline-sm text-on-surface font-bold tabular-nums">—</span>
<span class="font-code-inline text-code-inline text-primary text-[11px]">bpm</span>
</div>
</div>
<!-- Micro Pulse Visualizer -->
<div class="h-10 w-full flex items-center gap-1 px-1 bg-surface-container-lowest/40 rounded-lg"><span class="w-full text-center text-on-surface-variant font-label-sm">尚无心率监测记录</span></div>
</div>
<div class="mt-space-md pt-space-sm bg-surface-container-lowest/60 -mx-space-md -mb-space-md p-space-md flex items-center justify-between">
<span class="font-label-sm text-label-sm text-on-surface-variant">乳酸阈值配速</span>
<span class="font-code-inline text-code-inline text-on-surface font-semibold">— /km</span>
</div>
</article>
<!-- 3. Recovery Score Card -->
<article class="bg-surface-container-lowest rounded-xl p-space-md lg:p-space-lg shadow-xl relative overflow-hidden flex flex-col justify-between hover:bg-surface-container transition-colors duration-200">
<div>
<div class="flex items-center justify-between mb-space-sm">

<span class="flex items-center gap-1 font-label-sm text-label-sm text-primary font-medium">
<span class="w-2 h-2 rounded-full bg-primary animate-ping"></span> 暂无监测数据 </span>
</div>
<h3 class="font-headline-sm text-headline-sm text-on-surface font-semibold mb-space-md">身体恢复与睡眠质量</h3>
<div class="flex items-baseline gap-2 mb-space-md">
<span class="font-display-lg text-display-lg font-bold text-primary tabular-nums drop-shadow-sm">—%</span>
<span class="font-label-md text-label-md text-primary font-semibold">暂无评分</span>
</div>
<div class="flex flex-col gap-2.5 mb-space-sm">
<!-- HRV Metric -->
<div class="flex items-center justify-between p-2.5 bg-surface-container rounded-lg">
<div class="flex items-center gap-2">
<span class="material-symbols-outlined text-[18px] text-primary">ecg_heart</span>
<span class="font-label-md text-label-md text-on-surface">HRV 心率变异性</span>
</div>
<span class="font-code-inline text-code-inline text-on-surface font-semibold tabular-nums">— ms <span class="text-primary font-normal text-xs"></span></span>
</div>
<!-- Deep Sleep Metric -->
<div class="flex items-center justify-between p-2.5 bg-surface-container rounded-lg">
<div class="flex items-center gap-2">
<span class="material-symbols-outlined text-[18px] text-info">bedtime</span>
<span class="font-label-md text-label-md text-on-surface">深度睡眠均值</span>
</div>
<span class="font-code-inline text-code-inline text-on-surface font-semibold tabular-nums">— hrs <span class="text-info font-normal text-xs"></span></span>
</div>
</div>
</div>
<div class="mt-space-md pt-space-sm bg-surface-container-lowest/60 -mx-space-md -mb-space-md p-space-md flex items-center justify-between">
<span class="font-label-sm text-label-sm text-on-surface-variant">中枢神经建议</span>
<span class="font-label-sm text-label-sm text-primary font-semibold">尚无恢复评估</span>
</div>
</article>
</section>
<!-- Bottom Precision Status Bar -->
<div class="mt-space-xl flex flex-col sm:flex-row items-center justify-between gap-space-sm text-on-surface-variant font-label-sm text-label-sm pt-space-md"><div class="flex items-center gap-space-sm"><span class="w-2 h-2 rounded-full bg-primary"></span><span class="">{{ui.online?'数据已同步':'等待连接'}}</span></div><div class="flex items-center gap-space-md font-code-inline text-code-inline"><span class="">{{ui.data?.fetched_at?.slice(0,19).replace('T',' ')||'尚未同步'}}</span><span class="">•</span><button class="text-primary hover:underline flex items-center gap-1"><span class="material-symbols-outlined text-[14px]">file_download</span>导出数据</button></div></div>
</div>
</div>
</div></main></div>





</div>
</template>
<style scoped> html, .screen-stats { margin: 0; padding: 0; } .screen-stats { overscroll-behavior: none; } main > :first-child { margin-top: 0 !important; } main > :last-child { margin-bottom: 0 !important; } ::-webkit-scrollbar { display: none; }
</style>
