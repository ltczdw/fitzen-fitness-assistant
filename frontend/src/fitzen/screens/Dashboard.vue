<script setup lang="ts">
import {computed} from 'vue'
import {useFitzen} from '../state'
import {today,todayKey,week,calendar,streak,records,weekDays,weekVolume,fmt,pct,bar} from '../metrics'
const {ui,navigate}=useFitzen()
const todayStr=computed(()=>todayKey.value.replace(/-/g,'/')),weekdayStr=computed(()=>['星期日','星期一','星期二','星期三','星期四','星期五','星期六'][new Date(todayKey.value+'T12:00:00').getDay()])
const greeting=computed(()=>new Date().getHours()<12?'早上好':new Date().getHours()<18?'下午好':'晚上好')
const userName=computed(()=>ui.profile.display_name||'朋友'),userGoal=computed(()=>ui.profile.goal||'尚未设置'),userWeight=computed(()=>ui.profile.body_weight_kg)
const todaySets=computed(()=>today.value.completed_sets??null),todayWeight=computed(()=>today.value.training_volume??null),todayKcal=computed(()=>today.value.calories??null),targetKcal=computed(()=>ui.targets.calories_target??null),kcalProgress=computed(()=>pct(todayKcal.value,targetKcal.value))
const recentWorkouts=computed(()=>records.value.slice(0,5)),weekWorkoutDays=weekDays
const trainingLine=computed(()=>{const points=week.value;if(!points.length)return 'M0 28 L80 28';const max=Math.max(1,...points.map(d=>d.completed_sets));return points.map((d,i)=>(i?'L':'M')+(i*80/6)+' '+(28-d.completed_sets/max*25)).join(' ')})

</script>
<template>
<div class="screen-dashboard bg-background font-body-md text-on-surface antialiased selection:bg-primary selection:text-on-primary"><div class=" flex flex-col min-h-screen"><main class="w-full pt-0 bg-background pb-20 lg:pb-12"><div class="fz-page-inset max-w-7xl mx-auto"><div class="fz-section-stack flex flex-col w-full">
<!-- Top Greeting & Biometric Live Status Header -->
<section class="flex flex-col md:flex-row md:items-end justify-between gap-space-md pb-space-xs">
<div class="flex flex-col gap-1.5">
<div class="flex items-center gap-space-sm flex-wrap">
<span class="font-headline-lg text-headline-lg text-on-surface tracking-tight">{{ greeting }}，{{ userName }}</span>
<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-primary-container/15 text-primary text-label-sm font-label-sm">
<span class="material-symbols-outlined text-[14px] text-primary" style="font-variation-settings: &quot;FILL&quot; 1;">local_fire_department</span> 连续记录 {{streak}} 天 </span>

</div>
<p class="font-body-sm text-body-sm text-on-surface-variant flex items-center gap-2">
<span class="">{{ todayStr }} · {{ weekdayStr }}</span>
<span class="text-on-surface-variant">•</span>
<span class="">阶段目标: {{ userGoal }}</span>
</p>
</div>
<div class="flex items-center gap-space-sm self-start md:self-auto">
<button class="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface text-label-md font-label-md transition-all">
<span class="material-symbols-outlined text-[16px]">file_download</span>
<span class="">导出报告</span>
</button>
<button class="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary-container text-on-primary-container hover:brightness-110 active:scale-[0.98] text-label-md font-label-md font-semibold transition-all shadow-sm">
<span class="material-symbols-outlined text-[18px]">add</span>
<span class="">快速记录</span>
</button>
</div>
</section>
<!-- 4 Telemetry Metric Tiles -->
<section class="dashboard-metrics grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 fz-module-grid">
<!-- Tile 1: 今日训练 -->
<div class="bg-surface-container-lowest rounded-2xl p-5 flex flex-col justify-between card-lift shadow-sm transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 group">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2">
<div class="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center text-primary">
<span class="material-symbols-outlined text-[18px]">fitness_center</span>
</div>
<span class="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">今日训练</span>
</div>
<span class="px-2 py-0.5 rounded bg-primary-container/15 text-primary text-label-sm font-label-sm">{{todaySets?'已记录':'尚未记录'}}</span>
</div>
<div class="my-4 flex items-baseline justify-between">
<div>
<span class="font-display-lg text-display-lg text-on-surface font-bold tracking-tight">{{ todaySets }}</span>
<span class="font-label-md text-label-md text-on-surface-variant ml-1">组</span>
</div>
<!-- Mini Trend Sparkline SVG -->
<div class="w-20 h-8 flex items-end">
<svg class="w-full h-full text-primary" fill="none" preserveAspectRatio="none" viewBox="0 0 80 30">
<path :d="trainingLine" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></path>
<path :d="trainingLine+' V30 H0 Z'" fill="currentColor" fill-opacity="0.10" class="text-primary"></path>
</svg>
</div>
</div>
<div class="flex items-center justify-between text-label-sm font-label-sm pt-2 bg-surface-container-lowest/40 rounded-lg px-2.5 py-1.5">
<span class="text-on-surface-variant">已完成组数</span>
<span class="text-primary font-medium">今日已练 {{ todaySets }} 组</span>
</div>
</div>
<!-- Tile 2: 热量摄入 -->
<div class="bg-surface-container-lowest rounded-2xl p-5 flex flex-col justify-between card-lift shadow-sm transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 group">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2">
<div class="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center text-secondary">
<span class="material-symbols-outlined text-[18px]">local_fire_department</span>
</div>
<span class="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">热量摄入</span>
</div>
<span class="text-label-sm font-label-sm text-on-surface-variant">剩余 {{fmt(targetKcal!=null&&todayKcal!=null?Math.max(0,targetKcal-todayKcal):null)}} kcal</span>
</div>
<div class="my-4">
<div class="flex items-baseline gap-1">
<span class="font-display-lg text-display-lg text-on-surface font-bold tracking-tight">{{ fmt(todayKcal) }}</span>
<span class="font-label-md text-label-md text-on-surface-variant">kcal</span>
</div>
<!-- Progress Bar -->
<div class="w-full h-1.5 bg-surface-container rounded-full mt-3 overflow-hidden">
<div class="h-full bg-primary rounded-full transition-all duration-500 shadow-sm" :style="{ width: (kcalProgress||0) + '%' }"></div>
</div>
</div>
<div class="flex items-center justify-between text-label-sm font-label-sm pt-2 bg-surface-container-lowest/40 rounded-lg px-2.5 py-1.5">
<span class="text-on-surface-variant">目标 {{ fmt(targetKcal) }} kcal</span>
<span class="text-primary font-medium">{{ fmt(kcalProgress) }}%</span>
</div>
</div>
<!-- Tile 3: 运动消耗 -->
<div class="bg-surface-container-lowest rounded-2xl p-5 flex flex-col justify-between card-lift shadow-sm transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 group">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2">
<div class="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center text-primary">
<span class="material-symbols-outlined text-[18px]">ecg_heart</span>
</div>
<span class="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">训练容量</span>
</div>
<span class="px-2 py-0.5 rounded bg-primary-container/15 text-primary text-label-sm font-label-sm flex items-center gap-0.5">
<span class="material-symbols-outlined text-[13px]">arrow_upward</span>— </span>
</div>
<div class="my-4 flex items-baseline justify-between">
<div>
<span class="font-display-lg text-display-lg text-on-surface font-bold tracking-tight">{{ fmt(todayWeight) }}</span>
<span class="font-label-md text-label-md text-on-surface-variant ml-1">kg</span>
</div>
<!-- Micro Activity Heat Pulse Bars -->
<div class="flex items-end gap-1 h-7"><div v-for="(day, i) in week" :key="day.date" class="w-1.5 rounded-sm" :class="i === week.length - 1 ? 'bg-primary' : 'bg-primary/25'" :style="{height:bar(day.training_volume,Math.max(1,...week.map(d=>d.training_volume)))+'%'}"></div></div>
</div>
<div class="flex items-center justify-between text-label-sm font-label-sm pt-2 bg-surface-container-lowest/40 rounded-lg px-2.5 py-1.5">
<span class="text-on-surface-variant">今日总训练重量</span>
<span class="text-on-surface font-code-inline">{{ todaySets }} 组</span>
</div>
</div>
<!-- Tile 4: 体重趋势 -->
<div class="bg-surface-container-lowest rounded-2xl p-5 flex flex-col justify-between card-lift shadow-sm transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 group">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2">
<div class="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center text-secondary">
<span class="material-symbols-outlined text-[18px]">scale</span>
</div>
<span class="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">体重监控</span>
</div>
<span class="px-2 py-0.5 rounded bg-secondary-container text-secondary text-label-sm font-label-sm flex items-center gap-0.5">
<span class="material-symbols-outlined text-[13px]">arrow_downward</span>— kg </span>
</div>
<div class="my-4 flex items-baseline justify-between">
<div>
<span class="font-display-lg text-display-lg text-on-surface font-bold tracking-tight">{{ userWeight ? userWeight.toFixed(1) : '--' }}</span>
<span class="font-label-md text-label-md text-on-surface-variant ml-1">kg</span>
</div>
<!-- Micro Smooth Weight Curve -->
<div class="w-20 h-7">
<svg class="w-full h-full text-secondary" fill="none" viewBox="0 0 70 24"><text x="35" y="16" text-anchor="middle" fill="currentColor" font-size="10">暂无历史</text></svg>
</div>
</div>
<div class="flex items-center justify-between text-label-sm font-label-sm pt-2 bg-surface-container-lowest/40 rounded-lg px-2.5 py-1.5">
<span class="text-on-surface-variant">当前体重</span>
<span class="text-secondary font-medium">目标 {{ ui.profile?.target_weight_kg ? ui.profile.target_weight_kg + ' kg' : '未设置' }}</span>
</div>
</div>
</section>
<!-- Middle Section: 7-Day Training Schedule (Left) + AI Coach Terminal (Right) -->
<section class="dashboard-overview grid grid-cols-1 lg:grid-cols-12 fz-module-grid">
<!-- Left: Weekly Schedule Calendar (8 Cols) -->
<div class="lg:col-span-8 bg-surface-container-lowest rounded-2xl p-space-md flex flex-col justify-between">
<div>
<div class="flex items-center justify-between mb-space-md">
<div class="flex flex-col gap-0.5">
<div class="flex items-center gap-2">
<span class="material-symbols-outlined text-[20px] text-primary">calendar_month</span>
<h2 class="font-headline-sm text-headline-sm text-on-surface font-semibold">本周训练计划</h2>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant">{{ui.data?.plan?.plan_name||'本周训练记录'}}</p>
</div>
<div class="flex items-center gap-2">
<span class="px-2.5 py-1 rounded bg-surface-container text-on-surface-variant text-label-sm font-label-sm">
              周负荷: <span class="text-on-surface font-medium">—</span>
</span>
<button aria-label="Settings" class="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface transition-colors">
<span class="material-symbols-outlined text-[18px]">tune</span>
</button>
</div>
</div>
<!-- 7 Days Grid -->
<div class="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2.5"><div v-for="day in calendar" :key="day.date" :class="day.today?'ring-2 ring-primary bg-primary/[.08]':'bg-surface-container/60'" class="rounded-lg p-3 flex flex-col justify-between min-h-[148px] transition-colors">
<div class="flex items-center justify-between">
<span class="font-label-sm text-label-sm font-medium text-on-surface-variant">{{day.today?'今天 · ':''}}{{day.label}}</span>
<span class="material-symbols-outlined text-[15px] text-primary" style="font-variation-settings: &quot;FILL&quot; 1;">{{day.has?'check_circle':'schedule'}}</span>
</div>
<div class="my-2">
<div class="text-[13px] font-semibold text-on-surface leading-tight">{{day.title}}</div>
<span class="text-[11px] text-on-surface-variant mt-1 block">{{day.actions}} 动作 · {{day.sets}} 组</span>
</div>
<div class="flex items-center justify-between text-[11px] text-on-surface-variant pt-1.5 bg-surface-container-lowest/30 rounded px-1.5">
<span class="">{{day.date.slice(5)}}</span>
<span class="text-primary font-code-inline">已完成 {{day.sets}} 组</span>
</div>
</div></div>
</div>
<!-- Biometric telemetry footer strip in plan -->
<div class="mt-4 pt-3 flex flex-wrap items-center justify-between gap-2 text-label-sm font-label-sm text-on-surface-variant">
<div class="flex items-center gap-4">
<span class="flex items-center gap-1.5">
<span class="w-2 h-2 rounded-full bg-primary"></span>
            已完成 {{ weekWorkoutDays }} 天 / 目标 {{ ui.profile.available_days || '—' }} 天
          </span>
<span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-secondary"></span>本周完成率 {{ fmt(pct(weekWorkoutDays,ui.profile.available_days)) }}%</span>
</div>
<span class="text-on-surface font-code-inline">总容量: {{ Math.round(weekVolume).toLocaleString() }} kg</span>
</div>
</div>
<!-- Right: FitZen AI Terminal (4 Cols) -->
<div class="lg:col-span-4 bg-surface-container-lowest rounded-2xl p-space-md flex flex-col justify-between relative overflow-hidden">
<!-- Ambient Glow in Background -->
<div class="absolute -top-10 -right-10 w-32 h-32 rounded-full pointer-events-none blur-3xl" style="background: rgb(var(--fz-rgb-primary) / 0.06)" aria-hidden="true"></div>
<div class="flex flex-col gap-3.5 relative z-10">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2">
<div class="w-7 h-7 rounded-lg bg-surface-container flex items-center justify-center text-primary">
<span class="material-symbols-outlined text-[17px]">auto_awesome</span>
</div>
<span class="font-headline-sm text-headline-sm text-on-surface font-semibold">FitZen AI</span>
</div>
<div class="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-surface-container text-primary text-label-sm font-label-sm">
<span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
<span class="">{{ui.online?'服务已连接':'等待连接'}}</span>
</div>
</div>
<!-- AI Message Bubble -->
<div class="bg-surface-container-lowest rounded-xl p-4 flex flex-col gap-2.5">
<div class="flex items-center justify-between text-label-sm font-label-sm text-on-surface-variant">
<span class="flex items-center gap-1 font-medium text-on-surface"><span class="text-primary font-code-inline">智能助手</span> · 训后分析建议</span>
<span class="font-code-inline">{{ui.messages.length?'最近会话':'尚无分析'}}</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface leading-relaxed">{{ui.messages.filter(m=>m.role==='assistant').slice(-1)[0]?.content||'记录训练和饮食后，可以请 AI 根据你的实际记录进行分析。'}}</p>
</div>
<!-- Preset Interactive Prompt Chips -->
<div class="flex flex-col gap-2">
<span class="font-label-sm text-label-sm text-on-surface-variant">快捷提问 / 建议指令:</span>
<div class="flex flex-col gap-1.5">
<button class="w-full text-left px-3 py-2 rounded-lg bg-surface-container border border-transparent hover:bg-primary/[.08] hover:text-primary hover:border-primary/20 text-on-surface text-body-sm font-body-sm transition-colors flex items-center justify-between group">
<span class="truncate">🔥 查看今日热量缺口与宏量营养</span>
<span class="material-symbols-outlined text-[15px] text-on-surface-variant group-hover:text-primary transition-colors">arrow_outward</span>
</button>
<button class="w-full text-left px-3 py-2 rounded-lg bg-surface-container border border-transparent hover:bg-primary/[.08] hover:text-primary hover:border-primary/20 text-on-surface text-body-sm font-body-sm transition-colors flex items-center justify-between group">
<span class="truncate">📊 调整下周推拉腿训练容量</span>
<span class="material-symbols-outlined text-[15px] text-on-surface-variant group-hover:text-primary transition-colors">arrow_outward</span>
</button>
</div>
</div>
</div>
<!-- Action Button -->
<div class="mt-4 pt-3 relative z-10">
<a class="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-primary-container text-on-primary-container font-label-md text-label-md font-semibold hover:brightness-110 active:scale-[0.99] transition-all shadow-sm" data-path="ai-chat" href="#/ai-chat">
<span class="">继续深度对话</span>
<span class="material-symbols-outlined text-[16px]">arrow_forward</span>
</a>
</div>
</div>
</section>
<!-- Bottom Full-Width Section: Recent Training Records -->
<section class="bg-surface-container-lowest rounded-2xl p-space-md flex flex-col gap-space-md">
<!-- Header with Filter Tabs -->
<div class="flex flex-col sm:flex-row sm:items-center justify-between gap-space-sm pb-1">
<div class="flex items-center gap-2">
<span class="material-symbols-outlined text-[20px] text-primary">history</span>
<h2 class="font-headline-sm text-headline-sm text-on-surface font-semibold">最近训练记录</h2>
<span class="font-label-sm text-label-sm text-on-surface-variant ml-1 font-code-inline">({{recentWorkouts.length}} 条最新)</span>
</div>
<!-- Category Filter Tabs -->
<div class="flex items-center gap-1.5 p-1 bg-surface-container rounded-lg self-start sm:self-auto" id="filter-tabs">
<button class="px-3 py-1 rounded bg-primary/10 text-primary text-label-sm font-label-sm font-semibold transition-colors">全部</button>
<button class="px-3 py-1 rounded text-on-surface-variant hover:text-on-surface text-label-sm font-label-sm transition-colors">力量</button>
<button class="px-3 py-1 rounded text-on-surface-variant hover:text-on-surface text-label-sm font-label-sm transition-colors">有氧</button>
<button class="px-3 py-1 rounded text-on-surface-variant hover:text-on-surface text-label-sm font-label-sm transition-colors">HIIT</button>
</div>
</div>
<!-- Workout Rows List -->
<div class="fz-data-body flex flex-col">
<!-- 空状态 -->
<div v-if="!recentWorkouts.length" class="flex flex-col items-center justify-center py-12 text-on-surface-variant">
<span class="material-symbols-outlined text-[48px] mb-3">fitness_center</span>
<p class="font-body-md">还没有训练记录</p>
<p class="font-body-sm text-on-surface-variant mt-1">开始你的第一次训练吧</p>
</div>
<!-- 记录列表 -->
<div v-for="(workout, idx) in recentWorkouts" :key="workout.id || idx" class="fz-data-row fz-recent-row flex items-center justify-between transition-colors gap-4 group">
<div class="flex items-center gap-3 min-w-0">
<div class="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center text-primary shrink-0 group-hover:scale-105 transition-transform">
<span class="material-symbols-outlined text-[20px]">fitness_center</span>
</div>
<div class="flex flex-col min-w-0">
<div class="flex items-center gap-2 min-w-0">
<span class="font-body-lg text-body-lg text-on-surface font-medium truncate">{{ workout.exercise || workout.title || '训练记录' }}</span>
</div>
<div class="flex items-center gap-2 text-body-sm font-body-sm text-on-surface-variant mt-0.5">
<span v-if="workout.sets">{{ workout.sets }} 组 × {{ workout.reps || '?' }} 次</span>
<span v-if="workout.weight_kg" class="text-on-surface font-code-inline font-medium">{{ workout.weight_kg }}kg</span>
</div>
</div>
</div>
<div class="flex items-center justify-between sm:justify-end gap-4 shrink-0 font-label-sm text-label-sm text-on-surface-variant">
<span class="font-code-inline">{{ (workout.session_date || workout.date || '').slice(0, 10) }}</span>
<span class="material-symbols-outlined text-[18px] text-on-surface-variant group-hover:text-on-surface transition-colors">chevron_right</span>
</div>
</div>
</div>
<!-- View All Logs Footer Link -->
<div class="flex items-center justify-between pt-2">
<span class="text-label-sm font-label-sm text-on-surface-variant">共计 {{ records.length }} 条记录</span>
<a class="flex items-center gap-1 text-primary hover:underline text-label-md font-label-md font-medium group" data-path="training" href="#/training">
<span class="">查看完整训练日志</span>
<span class="material-symbols-outlined text-[16px] group-hover:translate-x-0.5 transition-transform">arrow_forward</span>
</a>
</div>
</section>
</div>
</div></main></div>





</div>
</template>
<style scoped> html, .screen-dashboard { margin: 0; padding: 0; } .screen-dashboard { overscroll-behavior: none; } main > :first-child { margin-top: 0 !important; } main > :last-child { margin-bottom: 0 !important; } ::-webkit-scrollbar { display: none; }
@media (min-width: 1024px) and (max-width: 1359px) {
  .dashboard-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .dashboard-overview { grid-template-columns: minmax(0, 1fr); }
  .dashboard-overview > div { grid-column: auto; }
}
@media (prefers-reduced-motion: reduce) {
  .card-lift, .card-lift:hover { transition: none; transform: none; }
}
</style>
