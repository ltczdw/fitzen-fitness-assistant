<script setup lang="ts">
import { useFitzen } from '../state'
import {DeleteButton} from '../../components/ui'
const { ui, sendChat, openRecords } = useFitzen()
</script>
<template>
<div class="screen-nutritionmodal bg-background font-body-md text-on-surface antialiased selection:bg-primary selection:text-on-primary"><aside class="hidden lg:flex fixed left-0 top-0 h-full w-[240px] bg-surface-container-lowest border-r border-outline-variant/40 z-50 flex-col justify-between select-none"><div class="flex flex-col"><div class="h-16 px-space-md flex items-center justify-between border-b border-outline-variant/30"><div class="flex items-center gap-space-sm"><span class="fz-brand-logo h-8 w-auto object-contain" role="img" aria-label="FitZen Logo"><svg class="fz-brand-symbol" viewBox="0 0 32 32" aria-hidden="true"><rect class="fz-brand-symbol-bg" x="1" y="1" width="30" height="30" rx="9" /><path d="M6 17h5l3-8 4 15 3-7h5" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" /></svg><span class="fz-brand-wordmark" aria-hidden="true">FitZen</span></span></div><span class="fz-brand-version font-label-sm text-label-sm px-space-xs py-0.5 rounded bg-surface-container text-primary font-medium border border-outline-variant/40">v2.4 Pro</span></div><div class="px-space-md py-space-sm"><div class="text-on-surface-variant font-label-sm text-label-sm uppercase tracking-wider px-space-xs mb-space-xs">控制台</div><nav class="flex flex-col gap-1" data-active-classes="bg-surface-container text-primary border border-outline-variant/60 shadow-sm font-medium"><a class="flex items-center justify-between px-space-sm py-2 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors" data-path="dashboard" href="#/dashboard"><div class="flex items-center gap-space-sm"><span class="material-symbols-outlined text-[18px]">grid_view</span><span class="font-body-sm text-body-sm">首页</span></div></a><a class="flex items-center justify-between px-space-sm py-2 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors" data-path="training" href="#/training"><div class="flex items-center gap-space-sm"><span class="material-symbols-outlined text-[18px]">fitness_center</span><span class="font-body-sm text-body-sm">训练记录</span></div></a><a aria-current="page" class="flex items-center justify-between px-space-sm py-2 rounded-lg transition-colors bg-surface-container text-primary border border-outline-variant/60 shadow-sm font-medium" data-path="nutrition" href="#/nutrition"><div class="flex items-center gap-space-sm"><span class="material-symbols-outlined text-[18px]">restaurant</span><span class="font-body-sm text-body-sm">饮食记录</span></div></a><a class="flex items-center justify-between px-space-sm py-2 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors" data-path="stats" href="#/stats"><div class="flex items-center gap-space-sm"><span class="material-symbols-outlined text-[18px]">trending_up</span><span class="font-body-sm text-body-sm">数据统计</span></div></a><a class="flex items-center justify-between px-space-sm py-2 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors" data-path="ai-chat" href="#/ai-chat"><div class="flex items-center gap-space-sm"><span class="material-symbols-outlined text-[18px]">auto_awesome</span><span class="font-body-sm text-body-sm">AI 对话</span></div><span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span></a><a class="flex items-center justify-between px-space-sm py-2 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors" data-path="profile" href="#/profile"><div class="flex items-center gap-space-sm"><span class="material-symbols-outlined text-[18px]">person</span><span class="font-body-sm text-body-sm">个人中心</span></div></a></nav></div></div><div class="flex flex-col border-t border-outline-variant/30 p-space-md gap-space-sm"><div class="flex items-center justify-between px-space-xs py-1 text-on-surface-variant font-label-sm text-label-sm"><div class="flex items-center gap-space-xs"><span class="w-2 h-2 rounded-full bg-primary animate-ping"></span><span class="text-on-surface-variant">已同步</span></div><span class="font-code-inline text-code-inline text-on-surface-variant">60Hz</span></div><div class="flex items-center justify-between p-space-xs rounded-lg bg-surface-container border border-outline-variant/30"><div class="flex items-center gap-space-sm overflow-hidden"><img alt="默认头像" class="w-8 h-8 rounded-full object-cover shrink-0" src="/fitzen/assets/default-avatar.svg"><div class="flex flex-col truncate"><span class="font-label-md text-label-md text-on-surface font-medium truncate">Alex Vance</span><span class="font-label-sm text-label-sm text-on-surface-variant truncate">专业运动员</span></div></div><button aria-label="Switch Account" class="p-1 text-on-surface-variant hover:text-on-surface transition-colors"><span class="material-symbols-outlined text-[16px]">unfold_more</span></button></div></div></aside><div class="lg:pl-[240px] flex flex-col min-h-screen"><header class="fixed top-0 right-0 left-0 lg:left-[240px] h-16 bg-surface-container-lowest/90 backdrop-blur-xl border-b border-outline-variant/30 z-40 flex items-center justify-between px-space-md lg:px-space-lg"><div class="flex items-center gap-space-md"><div class="flex lg:hidden items-center gap-space-sm"><span class="fz-brand-logo h-7 w-auto object-contain" role="img" aria-label="FitZen Logo"><svg class="fz-brand-symbol" viewBox="0 0 32 32" aria-hidden="true"><rect class="fz-brand-symbol-bg" x="1" y="1" width="30" height="30" rx="9" /><path d="M6 17h5l3-8 4 15 3-7h5" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" /></svg><span class="fz-brand-wordmark" aria-hidden="true">FitZen</span></span></div><div class="hidden sm:flex items-center gap-space-xs font-body-sm text-body-sm text-on-surface-variant"><span class="text-on-surface-variant">控制台</span><span class="material-symbols-outlined text-[14px]">chevron_right</span><span class="text-on-surface font-medium capitalize">终端</span></div></div><div class="flex items-center gap-space-md"><div class="hidden md:flex items-center gap-space-sm px-space-sm py-1.5 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface-variant font-label-md text-label-md"><span class="material-symbols-outlined text-[16px]">search</span><span class="">搜索指标、食物与日志...</span><kbd class="px-1.5 py-0.5 rounded bg-surface-variant text-on-surface font-code-inline text-code-inline border border-outline-variant/40">⌘K</kbd></div><button aria-label="Notifications" class="relative p-2 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors"><span class="material-symbols-outlined text-[20px]">notifications</span><span class="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-primary ring-2 ring-surface-container-lowest"></span></button><div class="flex items-center gap-space-sm border-l border-outline-variant/30 pl-space-md"><img alt="默认头像" class="w-8 h-8 rounded-full object-cover ring-1 ring-outline-variant/40" src="/fitzen/assets/default-avatar.svg"></div></div></header><main class="w-full pt-16 bg-background pb-20 lg:pb-12"><div class="fz-page-inset max-w-7xl mx-auto"><div class="fz-section-stack flex flex-col w-full">
<div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-space-md">
<div class="flex flex-col gap-1">
<div class="flex items-center gap-space-xs">
<span class="font-headline-lg text-headline-lg font-bold tracking-tight text-on-surface">饮食记录</span>
<span class="inline-flex items-center gap-1 px-space-xs py-0.5 rounded bg-surface-container text-primary font-code-inline text-code-inline"><span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>实时追踪</span>
</div>
<p class="font-body-md text-body-md text-on-surface-variant">精准宏量营养素追踪 · 达成每日增肌代谢目标</p>
</div>
<div class="flex items-center gap-space-sm self-start lg:self-auto">
<button class="flex items-center gap-space-xs px-space-md py-2 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface font-label-md text-label-md transition-colors shadow-sm">
<span class="material-symbols-outlined text-[18px] text-primary">photo_camera</span>
<span class="">拍照识别食物 (AI)</span>
</button>
<button class="flex items-center gap-space-xs px-space-md py-2 rounded-lg bg-primary-container text-on-primary-container hover:bg-primary hover:text-on-primary font-label-md text-label-md font-semibold transition-all shadow-sm">
<span class="material-symbols-outlined text-[18px]">add</span>
<span class="">添加食物记录</span>
</button>
</div>
</div>
<div class="rounded-xl bg-surface-container-lowest p-space-lg shadow-md flex flex-col gap-space-lg relative overflow-hidden">
<div class="absolute -right-20 -top-20 w-72 h-72 rounded-full bg-surface-container blur-3xl pointer-events-none"></div>
<div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 z-10">
<div class="flex items-center gap-space-xs">
<span class="material-symbols-outlined text-primary text-[20px]">donut_large</span>
<h2 class="font-headline-sm text-headline-sm font-semibold text-on-surface">今日营养素摄入汇总</h2>
<span class="font-label-sm text-label-sm px-2 py-0.5 rounded bg-surface-container text-on-surface-variant">周期: 增肌期 · 高蛋白</span>
</div>
<div class="flex items-center gap-space-sm text-on-surface-variant font-body-sm text-body-sm">
<span class="">推荐基准能量:</span>
<span class="font-metric-display text-headline-sm font-semibold text-primary">2,400</span>
<span class="text-on-surface-variant">kcal</span>
</div>
</div>
<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 fz-module-grid z-10">
<div class="flex items-center gap-space-md p-space-md rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors">
<div class="relative w-16 h-16 shrink-0 flex items-center justify-center">
<svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
<circle class="text-surface-variant" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-width="5"></circle>
<circle class="text-primary" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-dasharray="163.36" stroke-dashoffset="37.57" stroke-linecap="round" stroke-width="5"></circle>
</svg>
<span class="absolute font-label-md text-label-md font-bold text-on-surface">77%</span>
</div>
<div class="flex flex-col min-w-0">
<span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">总能量摄入</span>
<div class="flex items-baseline gap-1 mt-0.5">
<span class="font-headline-sm text-headline-sm font-bold text-on-surface">1,850</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">/ 2,400 kcal</span>
</div>
<span class="font-label-sm text-label-sm text-primary mt-1">剩余可用 550 kcal</span>
</div>
</div>
<div class="flex items-center gap-space-md p-space-md rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors">
<div class="relative w-16 h-16 shrink-0 flex items-center justify-center">
<svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
<circle class="text-surface-variant" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-width="5"></circle>
<circle class="text-secondary" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-dasharray="163.36" stroke-dashoffset="17.96" stroke-linecap="round" stroke-width="5"></circle>
</svg>
<span class="absolute font-label-md text-label-md font-bold text-on-surface">89%</span>
</div>
<div class="flex flex-col min-w-0">
<span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">蛋白质</span>
<div class="flex items-baseline gap-1 mt-0.5">
<span class="font-headline-sm text-headline-sm font-bold text-secondary">142</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">/ 160 g</span>
</div>
<span class="font-label-sm text-label-sm text-on-surface-variant mt-1">目标 2.0g/kg 体重</span>
</div>
</div>
<div class="flex items-center gap-space-md p-space-md rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors">
<div class="relative w-16 h-16 shrink-0 flex items-center justify-center">
<svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
<circle class="text-surface-variant" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-width="5"></circle>
<circle class="text-primary-dim" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-dasharray="163.36" stroke-dashoffset="40.84" stroke-linecap="round" stroke-width="5"></circle>
</svg>
<span class="absolute font-label-md text-label-md font-bold text-on-surface">75%</span>
</div>
<div class="flex flex-col min-w-0">
<span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">碳水化合物</span>
<div class="flex items-baseline gap-1 mt-0.5">
<span class="font-headline-sm text-headline-sm font-bold text-on-surface">195</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">/ 260 g</span>
</div>
<span class="font-label-sm text-label-sm text-on-surface-variant mt-1">剩余 65 g</span>
</div>
</div>
<div class="flex items-center gap-space-md p-space-md rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors">
<div class="relative w-16 h-16 shrink-0 flex items-center justify-center">
<svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
<circle class="text-surface-variant" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-width="5"></circle>
<circle class="text-tertiary-fixed-dim" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-dasharray="163.36" stroke-dashoffset="42.47" stroke-linecap="round" stroke-width="5"></circle>
</svg>
<span class="absolute font-label-md text-label-md font-bold text-on-surface">74%</span>
</div>
<div class="flex flex-col min-w-0">
<span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">优质脂肪</span>
<div class="flex items-baseline gap-1 mt-0.5">
<span class="font-headline-sm text-headline-sm font-bold text-on-surface">48</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">/ 65 g</span>
</div>
<span class="font-label-sm text-label-sm text-on-surface-variant mt-1">剩余 17 g</span>
</div>
</div>
</div>
<div class="flex flex-col gap-2 pt-2 z-10">
<div class="flex items-center justify-between text-on-surface-variant font-label-sm text-label-sm">
<span class="uppercase tracking-wider">全天摄入节律分布</span>
<span class="font-code-inline text-code-inline text-primary">摄入节奏分布</span>
</div>
<div class="h-3 w-full rounded-full bg-surface-container-high overflow-hidden flex gap-0.5 p-0.5">
<div class="h-full rounded-l-full bg-primary-container" style="width: 25%" title="早餐: 25%"></div>
<div class="h-full bg-primary" style="width: 40%" title="午餐: 40%"></div>
<div class="h-full bg-secondary-fixed-dim" style="width: 25%" title="晚餐: 25%"></div>
<div class="h-full rounded-r-full bg-surface-variant" style="width: 10%" title="加餐/补剂: 10%"></div>
</div>
<div class="flex items-center justify-between text-on-surface-variant font-label-sm text-label-sm px-1">
<div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-primary-container"></span><span class="">早 25% (480 kcal)</span></div>
<div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-primary"></span><span class="">午 40% (780 kcal)</span></div>
<div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-secondary-fixed-dim"></span><span class="">晚 25% (590 kcal)</span></div>
<div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-surface-variant"></span><span class="">加餐 10%</span></div>
</div>
</div>
</div>
<div class="grid grid-cols-1 lg:grid-cols-3 gap-space-lg items-stretch">
<div class="flex flex-col justify-between rounded-xl bg-surface-container-lowest p-space-md shadow-md">
<div class="flex flex-col gap-space-md">
<div class="flex items-center justify-between pb-2">
<div class="flex flex-col">
<div class="flex items-center gap-space-xs">
<span class="text-[18px]">🍳</span>
<h3 class="font-headline-sm text-headline-sm font-semibold text-on-surface">早餐</h3>
</div>
<div class="flex items-center gap-space-xs text-on-surface-variant font-label-sm text-label-sm mt-1">
<span class="text-primary font-medium">480 kcal</span>
<span class="">·</span>
<span class="">蛋白质 32g</span>
</div>
</div>
<button aria-label="添加早餐" class="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-primary hover:text-on-surface transition-colors">
<span class="material-symbols-outlined text-[20px]">add</span>
</button>
</div>
<div class="relative w-full h-32 rounded-lg overflow-hidden">
<img alt="早餐示意图" class="w-full h-full object-cover"  src="/fitzen/assets/breakfast.svg">
<div class="absolute inset-0 bg-gradient-to-t from-surface-container-lowest via-transparent to-transparent"></div>
</div>
<div class="flex flex-col gap-space-xs">
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">燕麦牛奶糊</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">80g 燕麦 + 250ml 脱脂奶</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">320 kcal</span>
<span class="">P: 16g</span>
<span class="">C: 52g</span>
<span class="">F: 5g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">水煮蛋</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">2 枚 (约 100g)</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">140 kcal</span>
<span class="">P: 13g</span>
<span class="">C: 1g</span>
<span class="">F: 9g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">黑咖啡</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">美式无糖 300ml</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">20 kcal</span>
<span class="">P: 3g</span>
<span class="">C: 0g</span>
<span class="">F: 0g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
</div>
</div>
<div class="pt-3 mt-3 flex items-center justify-between text-on-surface-variant font-label-sm text-label-sm">
<span class="">早餐摄入达标</span>
<span class="text-primary font-medium">98% 符合目标</span>
</div>
</div>
<div class="flex flex-col justify-between rounded-xl bg-surface-container-lowest p-space-md shadow-md">
<div class="flex flex-col gap-space-md">
<div class="flex items-center justify-between pb-2">
<div class="flex flex-col">
<div class="flex items-center gap-space-xs">
<span class="text-[18px]">🥩</span>
<h3 class="font-headline-sm text-headline-sm font-semibold text-on-surface">午餐</h3>
</div>
<div class="flex items-center gap-space-xs text-on-surface-variant font-label-sm text-label-sm mt-1">
<span class="text-primary font-medium">780 kcal</span>
<span class="">·</span>
<span class="">蛋白质 62g</span>
</div>
</div>
<button aria-label="添加午餐" class="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-primary hover:text-on-surface transition-colors">
<span class="material-symbols-outlined text-[20px]">add</span>
</button>
</div>
<div class="relative w-full h-32 rounded-lg overflow-hidden">
<img alt="午餐示意图" class="w-full h-full object-cover"  src="/fitzen/assets/lunch.svg">
<div class="absolute inset-0 bg-gradient-to-t from-surface-container-lowest via-transparent to-transparent"></div>
</div>
<div class="flex flex-col gap-space-xs">
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">香烤鸡胸肉沙拉</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">鸡胸 200g + 混合绿叶菜</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">380 kcal</span>
<span class="">P: 45g</span>
<span class="">C: 8g</span>
<span class="">F: 8g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">糙米藜麦饭</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">180g 蒸煮熟重</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">240 kcal</span>
<span class="">P: 6g</span>
<span class="">C: 50g</span>
<span class="">F: 2g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">鳄梨牛油果</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">半个 (70g) 优质脂肪来源</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">110 kcal</span>
<span class="">P: 2g</span>
<span class="">C: 4g</span>
<span class="">F: 10g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">清炒芦笋</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">120g 橄榄油轻炒</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">50 kcal</span>
<span class="">P: 9g</span>
<span class="">C: 3g</span>
<span class="">F: 1g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
</div>
</div>
<div class="pt-3 mt-3 flex items-center justify-between text-on-surface-variant font-label-sm text-label-sm">
<span class="">宏量黄金配比</span>
<span class="text-primary font-medium">高蛋白优质</span>
</div>
</div>
<div class="flex flex-col justify-between rounded-xl bg-surface-container-lowest p-space-md shadow-md">
<div class="flex flex-col gap-space-md">
<div class="flex items-center justify-between pb-2">
<div class="flex flex-col">
<div class="flex items-center gap-space-xs">
<span class="text-[18px]">🥗</span>
<h3 class="font-headline-sm text-headline-sm font-semibold text-on-surface">晚餐</h3>
</div>
<div class="flex items-center gap-space-xs text-on-surface-variant font-label-sm text-label-sm mt-1">
<span class="text-primary font-medium">590 kcal</span>
<span class="">·</span>
<span class="">蛋白质 48g</span>
</div>
</div>
<button aria-label="添加晚餐" class="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-primary hover:text-on-surface transition-colors">
<span class="material-symbols-outlined text-[20px]">add</span>
</button>
</div>
<div class="relative w-full h-32 rounded-lg overflow-hidden">
<img alt="晚餐示意图" class="w-full h-full object-cover"  src="/fitzen/assets/dinner.svg">
<div class="absolute inset-0 bg-gradient-to-t from-surface-container-lowest via-transparent to-transparent"></div>
</div>
<div class="flex flex-col gap-space-xs">
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">香煎大西洋三文鱼</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">180g 富含 Omega-3</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">360 kcal</span>
<span class="">P: 38g</span>
<span class="">C: 0g</span>
<span class="">F: 22g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">烤红薯</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">150g 慢碳补充</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">165 kcal</span>
<span class="">P: 3g</span>
<span class="">C: 38g</span>
<span class="">F: 0.5g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
<div class="flex items-center justify-between p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors group">
<div class="flex flex-col min-w-0 pr-2">
<span class="font-label-md text-label-md text-on-surface font-medium truncate">蒜蓉水煮西蓝花</span>
<span class="font-body-sm text-body-sm text-on-surface-variant truncate">150g 高纤维抗氧化</span>
<div class="flex items-center gap-2 mt-1 font-label-sm text-label-sm text-on-surface-variant">
<span class="text-on-surface">65 kcal</span>
<span class="">P: 7g</span>
<span class="">C: 8g</span>
<span class="">F: 1g</span>
</div>
</div>
<div class="flex items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
<button aria-label="编辑" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
<DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact />
</div>
</div>
</div>
</div>
<div class="pt-3 mt-3 flex items-center justify-between text-on-surface-variant font-label-sm text-label-sm">
<span class="">慢释修复配方</span>
<span class="text-secondary font-medium">助眠配方</span>
</div>
</div>
</div>
<div class="grid grid-cols-1 lg:grid-cols-3 fz-module-grid">
<div class="lg:col-span-1 rounded-xl bg-surface-container-lowest p-space-md shadow-md flex flex-col justify-between gap-space-sm">
<div class="flex items-center justify-between">
<div class="flex items-center gap-space-xs">
<span class="material-symbols-outlined text-primary text-[20px]">medication</span>
<span class="font-headline-sm text-headline-sm font-semibold text-on-surface">加餐与补剂</span>
</div>
<button class="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-primary hover:text-on-surface transition-colors"><span class="material-symbols-outlined text-[20px]">add</span></button>
</div>
<div class="flex flex-col gap-2 mt-1">
<div class="flex items-center justify-between p-space-xs rounded bg-surface-container text-body-sm font-body-sm">
<div class="flex items-center gap-2">
<span class="w-1.5 h-1.5 rounded-full bg-primary"></span>
<span class="text-on-surface">支链氨基酸 BCAA</span>
</div>
<span class="font-code-inline text-code-inline text-on-surface-variant">5g (训练中)</span>
</div>
<div class="flex items-center justify-between p-space-xs rounded bg-surface-container text-body-sm font-body-sm">
<div class="flex items-center gap-2">
<span class="w-1.5 h-1.5 rounded-full bg-secondary-fixed"></span>
<span class="text-on-surface">分离乳清蛋白粉</span>
</div>
<span class="font-code-inline text-code-inline text-on-surface-variant">1 勺 (25g 蛋白质)</span>
</div>
</div>
<span class="font-label-sm text-label-sm text-on-surface-variant">已同步至每日肌肉合成支持指数 (MPS 94%)</span>
</div>
<div class="lg:col-span-2 rounded-xl bg-surface-container-lowest p-space-md shadow-md flex flex-col sm:flex-row sm:items-center justify-between gap-space-md relative overflow-hidden">
<div class="flex items-start gap-space-md z-10">
<div class="p-3 rounded-xl bg-surface-container text-primary shrink-0 flex items-center justify-center">
<span class="material-symbols-outlined text-[24px]">smart_toy</span>
</div>
<div class="flex flex-col gap-1">
<div class="flex items-center gap-space-xs">
<span class="font-label-md text-label-md text-primary uppercase font-bold tracking-wider">AI 智能营养引擎提示</span>
<span class="font-label-sm text-label-sm px-1.5 py-0.2 rounded bg-surface-container text-on-surface-variant">FitZen Neural v2.4</span>
</div>
<p class="font-body-md text-body-md text-on-surface">
            今日蛋白质已达标 <strong class="text-primary font-bold">89%</strong>，距最佳合成阈值尚差 18g。建议睡前 30 分钟补充 <span class="text-secondary">150g 希腊酸奶</span> 或酪蛋白，提供缓释氨基酸，有助于夜间肌肉深度修复。
          </p>
</div>
</div>
<button class="shrink-0 px-space-md py-2.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface font-label-md text-label-md transition-colors flex items-center justify-center gap-1 z-10">
<span class="">采纳并一键添加</span>
<span class="material-symbols-outlined text-[16px]">arrow_forward</span>
</button>
</div>
</div>
</div></div></main></div><nav class="lg:hidden fixed bottom-0 left-0 right-0 h-14 bg-surface-container-lowest/95 backdrop-blur-xl border-t border-outline-variant/30 z-50 flex items-center justify-around px-space-sm" data-active-classes="text-primary font-medium"><a class="flex flex-col items-center gap-0.5 text-on-surface-variant hover:text-on-surface" data-path="dashboard" href="#/dashboard"><span class="material-symbols-outlined text-[20px]">grid_view</span><span class="font-label-sm text-label-sm">首页</span></a><a class="flex flex-col items-center gap-0.5 text-on-surface-variant hover:text-on-surface" data-path="training" href="#/training"><span class="material-symbols-outlined text-[20px]">fitness_center</span><span class="font-label-sm text-label-sm">训练</span></a><a aria-current="page" class="flex flex-col items-center gap-0.5 text-primary font-medium" data-path="nutrition" href="#/nutrition"><span class="material-symbols-outlined text-[20px]">restaurant</span><span class="font-label-sm text-label-sm">饮食</span></a><a class="flex flex-col items-center gap-0.5 text-on-surface-variant hover:text-on-surface" data-path="stats" href="#/stats"><span class="material-symbols-outlined text-[20px]">trending_up</span><span class="font-label-sm text-label-sm">数据</span></a><a class="flex flex-col items-center gap-0.5 text-on-surface-variant hover:text-on-surface relative" data-path="ai-chat" href="#/ai-chat"><span class="material-symbols-outlined text-[20px]">auto_awesome</span><span class="font-label-sm text-label-sm">AI</span><span class="absolute top-0 right-1 w-1.5 h-1.5 rounded-full bg-primary"></span></a></nav>
<div class="fixed inset-0 z-50 bg-scrim/75 backdrop-blur-md flex items-center justify-center p-4" id="add-food-modal-overlay"><div class="w-full max-w-[580px] rounded-2xl bg-surface-container-lowest border border-outline-variant shadow-2xl p-6 text-on-surface flex flex-col gap-4 max-h-[90vh] overflow-y-auto"><div class="flex items-start justify-between border-b border-outline-variant pb-3.5"><div class="flex items-center gap-3"><div class="w-10 h-10 rounded-xl bg-surface-container border border-primary/20 flex items-center justify-center text-primary shrink-0"><span class="material-symbols-outlined text-[22px]">restaurant</span></div><div><div class="flex items-center gap-2"><h3 class="font-headline-sm text-lg font-semibold text-on-surface tracking-tight">添加饮食摄入</h3><span class="px-2 py-0.5 rounded-full bg-surface-container text-primary text-[11px] font-medium border border-primary/20">组合摄入 / 批量添加</span></div><p class="text-[12px] text-on-surface-variant mt-0.5">多食物组合摄入 · 实时计算总量与宏量营养配比</p></div></div><button aria-label="关闭弹窗" class="p-1.5 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest transition-colors"><span class="material-symbols-outlined text-[20px]">close</span></button></div><div class="flex flex-col gap-3.5"><div class="grid grid-cols-1 sm:grid-cols-2 gap-3"><div class="flex flex-col gap-1.5"><label class="text-[12px] font-medium text-on-surface-variant">餐次划分</label><div class="grid grid-cols-4 gap-1.5"><button class="flex items-center justify-center gap-1 py-1.5 rounded-lg border border-outline-variant bg-surface-container-low text-on-surface-variant text-xs hover:border-outline-variant">🍳 早</button><button class="flex items-center justify-center gap-1 py-1.5 rounded-lg border border-primary/50 bg-surface-container text-primary text-xs font-semibold shadow-sm">🥩 午</button><button class="flex items-center justify-center gap-1 py-1.5 rounded-lg border border-outline-variant bg-surface-container-low text-on-surface-variant text-xs hover:border-outline-variant">🥗 晚</button><button class="flex items-center justify-center gap-1 py-1.5 rounded-lg border border-outline-variant bg-surface-container-low text-on-surface-variant text-xs hover:border-outline-variant">💊 加</button></div></div><div class="flex flex-col gap-1.5"><label class="text-[12px] font-medium text-on-surface-variant">用餐时间</label><div class="flex items-center justify-between px-3 py-1.5 rounded-lg bg-surface-container-low border border-outline-variant text-xs text-on-surface"><span class="flex items-center gap-1.5"><span class="material-symbols-outlined text-[16px] text-primary">schedule</span>今天 12:30</span><span class="material-symbols-outlined text-[16px] text-on-surface-variant">expand_more</span></div></div></div><div class="flex flex-col gap-2"><div class="flex items-center justify-between"><span class="text-[13px] font-medium text-on-surface-variant flex items-center gap-1.5"><span class="material-symbols-outlined text-[16px] text-primary">format_list_bulleted</span>已选食物清单 (3 项)</span><span class="text-[11px] text-on-surface-variant">点击份量可直接编辑</span></div><div class="flex flex-col gap-2"><div class="p-3 rounded-xl bg-surface-container-lowest border border-primary/30 flex items-center justify-between gap-3"><div class="flex flex-col min-w-0 flex-1"><div class="flex items-center gap-2"><span class="text-sm font-semibold text-on-surface truncate">去皮清蒸鸡胸肉</span><span class="text-[10px] px-1.5 py-0.2 rounded bg-primary/20 text-primary font-medium">高蛋白推荐</span></div><div class="flex items-center gap-2 mt-1 text-[11px] text-on-surface-variant"><span class="text-on-surface font-medium">134 kcal</span><span class="">·</span><span class="">蛋白质 3.1g</span><span class="">·</span><span class="">碳水 28.3g</span><span class="">·</span><span class="">脂肪 1.0g</span></div></div><div class="flex items-center gap-2 shrink-0"><div class="flex items-center bg-surface-container-lowest border border-outline-variant rounded-lg px-2 py-1"><input class="w-12 bg-transparent text-sm text-right font-medium text-on-surface focus:outline-none" type="number" value="150"><span class="text-xs text-on-surface-variant ml-1">g</span></div><DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact /></div></div><div class="p-3 rounded-xl bg-surface-container-lowest border border-outline-variant flex items-center justify-between gap-3 hover:border-outline-variant transition-colors"><div class="flex flex-col min-w-0 flex-1"><div class="flex items-center gap-2"><span class="text-sm font-medium text-on-surface truncate">优质糙米藜麦饭</span><span class="text-[10px] px-1.5 py-0.2 rounded bg-surface-variant text-on-surface-variant">低GI慢碳</span></div><div class="flex items-center gap-2 mt-1 text-[11px] text-on-surface-variant"><span class="text-on-surface font-medium">134 kcal</span><span class="">·</span><span class="">蛋白质 3.1g</span><span class="">·</span><span class="">碳水 28.3g</span><span class="">·</span><span class="">脂肪 1.0g</span></div></div><div class="flex items-center gap-2 shrink-0"><div class="flex items-center bg-surface-container-lowest border border-outline-variant rounded-lg px-2 py-1"><input class="w-12 bg-transparent text-sm text-right font-medium text-on-surface focus:outline-none" type="number" value="120"><span class="text-xs text-on-surface-variant ml-1">g</span></div><DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact /></div></div><div class="p-3 rounded-xl bg-surface-container-lowest border border-outline-variant flex items-center justify-between gap-3 hover:border-outline-variant transition-colors"><div class="flex flex-col min-w-0 flex-1"><div class="flex items-center gap-2"><span class="text-sm font-medium text-on-surface truncate">蒜蓉水煮西兰花</span><span class="text-[10px] px-1.5 py-0.2 rounded bg-surface-variant text-on-surface-variant">高纤维抗氧化</span></div><div class="flex items-center gap-2 mt-1 text-[11px] text-on-surface-variant"><span class="text-on-surface font-medium">35 kcal</span><span class="">·</span><span class="">蛋白质 2.8g</span><span class="">·</span><span class="">碳水 4.2g</span><span class="">·</span><span class="">脂肪 0.4g</span></div></div><div class="flex items-center gap-2 shrink-0"><div class="flex items-center bg-surface-container-lowest border border-outline-variant rounded-lg px-2 py-1"><input class="w-12 bg-transparent text-sm text-right font-medium text-on-surface focus:outline-none" type="number" value="100"><span class="text-xs text-on-surface-variant ml-1">g</span></div><DeleteButton :action="()=>openRecords('nutrition')" subject="饮食记录" compact /></div></div></div></div><div class="flex flex-col gap-2"><div class="relative flex items-center"><span class="material-symbols-outlined absolute left-3 text-on-surface-variant text-[18px]">search</span><input class="w-full bg-surface-container-lowest border border-outline-variant rounded-xl pl-9 pr-24 py-2 text-sm text-on-surface focus:outline-none focus:border-primary placeholder:text-on-surface-variant" placeholder="搜索并添加更多食物..." type="text"><button class="absolute right-1.5 px-2.5 py-1 rounded-lg bg-surface-variant text-xs text-primary font-medium hover:bg-surface-container-high transition-colors flex items-center gap-1"><span class="material-symbols-outlined text-[14px]">add</span>继续添加</button></div><div class="flex items-center gap-1.5 overflow-x-hidden"><span class="text-[11px] text-on-surface-variant shrink-0">常用快捷:</span><button class="px-2 py-0.5 rounded-full bg-surface-container-low border border-outline-variant text-[11px] text-on-surface-variant hover:text-on-surface hover:border-primary/40 transition-colors">+ 水煮蛋 1枚</button><button class="px-2 py-0.5 rounded-full bg-surface-container-low border border-outline-variant text-[11px] text-on-surface-variant hover:text-on-surface hover:border-primary/40 transition-colors">+ 全麦面包 1片</button><button class="px-2 py-0.5 rounded-full bg-surface-container-low border border-outline-variant text-[11px] text-on-surface-variant hover:text-on-surface hover:border-primary/40 transition-colors">+ 脱脂牛奶 200ml</button></div></div><div class="rounded-xl bg-surface-container-lowest border border-outline-variant p-3.5 flex flex-col gap-2.5"><div class="flex items-center justify-between border-b border-outline-variant/80 pb-2"><div class="flex items-center gap-1.5 text-xs text-on-surface-variant"><span class="material-symbols-outlined text-[16px] text-primary">bolt</span><span class="font-medium text-on-surface">本次摄入合计 (共 3 项食物)</span></div><span class="text-[11px] text-primary font-medium bg-surface-container px-2 py-0.5 rounded-full">占今日目标 21%</span></div><div class="grid grid-cols-4 gap-2 text-center"><div class="flex flex-col"><span class="text-[11px] text-on-surface-variant">总热量</span><span class="text-base font-bold text-on-surface mt-0.5">416 <span class="text-[10px] font-normal text-on-surface-variant">kcal</span></span></div><div class="flex flex-col"><span class="text-[11px] text-on-surface-variant">蛋白质</span><span class="text-base font-bold text-primary mt-0.5">52.4 <span class="text-[10px] font-normal text-on-surface-variant">g</span></span></div><div class="flex flex-col"><span class="text-[11px] text-on-surface-variant">碳水</span><span class="text-base font-bold text-on-surface mt-0.5">32.5 <span class="text-[10px] font-normal text-on-surface-variant">g</span></span></div><div class="flex flex-col"><span class="text-[11px] text-on-surface-variant">优质脂肪</span><span class="text-base font-bold text-error mt-0.5">6.8 <span class="text-[10px] font-normal text-on-surface-variant">g</span></span></div></div><div class="flex flex-col gap-1 pt-1 border-t border-outline-variant/40"><div class="h-1.5 w-full rounded-full bg-surface-container-lowest overflow-hidden flex gap-0.5"><div class="h-full rounded-l-full bg-primary" style="width: 50%" title="蛋白质 50%"></div><div class="h-full bg-secondary-fixed" style="width: 31%" title="碳水 31%"></div><div class="h-full rounded-r-full bg-error" style="width: 15%" title="脂肪 15%"></div></div><div class="flex items-center justify-between text-[10px] text-on-surface-variant px-0.5"><span class="flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-primary"></span>蛋白 50%</span><span class="flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-secondary-fixed"></span>碳水 31%</span><span class="flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-error"></span>脂肪 15%</span></div></div></div></div><div class="flex items-center justify-between pt-2 border-t border-outline-variant"><DeleteButton :action="()=>{ui.foods=[];ui.mealTime=''}" subject="当前餐次草稿" label="清空全部" compact /><div class="flex items-center gap-2"><button class="px-4 py-2 rounded-[10px] border border-outline-variant text-xs font-medium text-on-surface-variant hover:bg-surface-container-low hover:text-on-surface transition-colors">取消</button><button class="flex items-center gap-1.5 px-5 py-2 rounded-[10px] bg-primary hover:bg-primary text-on-primary font-semibold text-xs shadow-sm transition-all"><span class="material-symbols-outlined text-[16px]">check</span><span class="">保存本次摄入 (3 项食物)</span></button></div></div></div></div>

</div>
</template>
<style scoped> html, .screen-nutritionmodal { margin: 0; padding: 0; } .screen-nutritionmodal { overscroll-behavior: none; } main > :first-child { margin-top: 0 !important; } main > :last-child { margin-bottom: 0 !important; } ::-webkit-scrollbar { display: none; }
</style>
