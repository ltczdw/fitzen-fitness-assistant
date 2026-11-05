<script setup lang="ts">
import { useFitzen } from '../state'
import { records } from '../metrics'
const { ui, sendChat } = useFitzen()
</script>
<template>
<div class="screen-status bg-background font-sans text-on-surface-variant antialiased selection:bg-primary selection:text-on-primary">
<!-- 顶部通栏导航 / 面包屑与状态栏 (无侧边栏设计) -->

<!-- 沉浸式大屏主体内容容器 -->
<main class="max-w-7xl mx-auto fz-page-inset fz-section-stack flex flex-col">
<!-- 头部指标与元信息 -->
<section class="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-2 border-b border-outline-variant/60">
<div class="flex flex-col gap-2">
<div class="flex items-center gap-2 text-primary font-mono text-xs uppercase tracking-wider">
<span class="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></span>
<span>STATUS_MATRIX // V2.4</span>
<span class="text-on-surface-variant">/</span>
<span class="text-on-surface-variant">DIAGNOSTIC_SUITE</span>
</div>
<h1 class="text-3xl sm:text-4xl font-semibold tracking-tight text-on-surface">系统状态矩阵</h1>
<p class="text-sm text-on-surface-variant max-w-2xl leading-relaxed">
          FitZen 运行态及边界状态完整图谱。覆盖数据拓扑缺失、异步推导流水线以及容错降级分支。
        </p>
</div>
<!-- 状态指示徽章 -->
<div class="flex items-center gap-2.5 self-start lg:self-end">
<div class="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-surface-container-lowest border border-outline-variant text-on-surface-variant font-mono text-xs">
<span class="text-primary font-semibold">{{[ui.online,!!ui.data,ui.messages.length>0].filter(Boolean).length}}</span>
<span class="text-on-surface-variant">/</span>
<span>ACTIVE_STATES</span>
</div>
<div class="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-surface-container-lowest border border-outline-variant text-on-surface-variant font-mono text-xs">
<span class="w-2 h-2 rounded-full bg-primary"></span>
<span>{{ui.data?.fetched_at?.slice(11,19)||'尚未同步'}}</span>
</div>
</div>
</section>
<!-- SECTION 1: 空状态 (EMPTY STATES) -->
<section class="flex flex-col gap-4">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2.5">
<span class="font-mono text-xs px-2 py-0.5 rounded bg-surface-container-lowest border border-outline-variant text-primary font-medium">SEC_01</span>
<h2 class="text-lg font-semibold text-on-surface">空状态 (Empty States)</h2>
</div>
<span class="font-mono text-xs text-on-surface-variant hidden sm:inline">NULL_DATASET_HANDLERS</span>
</div>
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 fz-module-grid">
<!-- 1. 训练记录为空 (哑铃) -->
<div class="group relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant hover:border-outline-variant transition-colors min-h-[380px]">
<div class="flex items-center justify-between w-full">
<span class="font-mono text-[11px] text-on-surface-variant">EVT_001 // WORKOUT_LOG</span>
<span class="w-1.5 h-1.5 rounded-full bg-surface-container-highest group-hover:bg-primary transition-colors"></span>
</div>
<div class="flex flex-col items-center text-center my-auto px-2 py-4">
<div class="relative w-16 h-16 rounded-xl bg-surface-container-lowest border border-outline-variant flex items-center justify-center mb-4 text-primary group-hover:text-primary transition-colors">
<!-- 哑铃图标 -->
<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" viewBox="0 0 24 24">
<path d="M6 5v14"></path>
<path d="M18 5v14"></path>
<path d="M2 9v6"></path>
<path d="M22 9v6"></path>
<path d="M6 12h12"></path>
</svg>
<div class="absolute inset-0 rounded-xl bg-surface-container pointer-events-none"></div>
</div>
<h3 class="text-base font-semibold text-on-surface mb-1.5">{{records.length?'已保存 '+records.length+' 条训练记录':'还没有训练记录'}}</h3>
<p class="text-xs text-on-surface-variant max-w-xs leading-relaxed">
              开始你的第一次训练吧，记录力量增长轨迹与复合动作负荷。
            </p>
</div>
<div class="w-full pt-4">
<button class="w-full py-2.5 px-4 rounded-lg bg-primary hover:bg-primary/90 text-on-primary font-medium text-xs sm:text-sm flex items-center justify-center gap-1.5 transition-all shadow-sm active:scale-[0.99]" type="button">
<span class="material-symbols-outlined text-base">add</span>
<span>添加训练</span>
</button>
</div>
</div>
<!-- 2. 饮食记录为空 (餐盘) -->
<div class="group relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant hover:border-outline-variant transition-colors min-h-[380px]">
<div class="flex items-center justify-between w-full">
<span class="font-mono text-[11px] text-on-surface-variant">EVT_002 // DIET_INTAKE</span>
<span class="w-1.5 h-1.5 rounded-full bg-surface-container-highest group-hover:bg-primary transition-colors"></span>
</div>
<div class="flex flex-col items-center text-center my-auto px-2 py-4">
<div class="relative w-16 h-16 rounded-xl bg-surface-container-lowest border border-outline-variant flex items-center justify-center mb-4 text-primary group-hover:text-primary transition-colors">
<!-- 餐盘/餐具图标 -->
<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" viewBox="0 0 24 24">
<path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"></path>
<path d="M7 2v20"></path>
<path d="M21 15V2v0a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"></path>
</svg>
<div class="absolute inset-0 rounded-xl bg-surface-container pointer-events-none"></div>
</div>
<h3 class="text-base font-semibold text-on-surface mb-1.5">{{ui.data?.today?.meal_count?'今天已记录 '+ui.data.today.meal_count+' 餐':'今天还没有饮食记录'}}</h3>
<p class="text-xs text-on-surface-variant max-w-xs leading-relaxed">
              合理规划宏量营养素摄入，平衡碳水、蛋白质与优质脂肪比率。
            </p>
</div>
<div class="w-full pt-4">
<button class="w-full py-2.5 px-4 rounded-lg bg-primary hover:bg-primary/90 text-on-primary font-medium text-xs sm:text-sm flex items-center justify-center gap-1.5 transition-all shadow-sm active:scale-[0.99]" type="button">
<span class="material-symbols-outlined text-base">restaurant</span>
<span>添加饮食</span>
</button>
</div>
</div>
<!-- 3. AI 对话为空 -->
<div class="group relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant hover:border-outline-variant transition-colors min-h-[380px]">
<div class="flex items-center justify-between w-full">
<span class="font-mono text-[11px] text-on-surface-variant">EVT_003 // AI_PROMPT_READY</span>
<span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
</div>
<div class="flex flex-col items-center text-center my-auto px-1 py-3">
<div class="relative w-16 h-16 rounded-xl bg-surface-container-lowest border border-outline-variant flex items-center justify-center mb-4 text-primary">
<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" viewBox="0 0 24 24">
<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
<path d="M9 10h.01"></path>
<path d="M15 10h.01"></path>
<path d="M12 10h.01"></path>
</svg>
<div class="absolute inset-0 rounded-xl bg-surface-container pointer-events-none"></div>
</div>
<h3 class="text-base font-semibold text-on-surface mb-1">{{ui.messages.length?'当前会话 '+ui.messages.length+' 条消息':'和 FitZen AI 开始对话吧'}}</h3>
<p class="text-xs text-on-surface-variant max-w-xs mb-3 leading-relaxed">
              获取针对性的训练计划微调与营养处方解析。
            </p>
<!-- 快捷提示胶囊 -->
<div class="flex flex-wrap justify-center gap-1.5 w-full">
<button class="px-2.5 py-1 rounded-full bg-surface-container-low hover:bg-surface-container-high border border-outline-variant text-on-surface-variant text-xs transition-colors" type="button">
                今天练胸怎么安排？
              </button>
<button class="px-2.5 py-1 rounded-full bg-surface-container-low hover:bg-surface-container-high border border-outline-variant text-on-surface-variant text-xs transition-colors" type="button">
                减脂期碳水循环
              </button>
<button class="px-2.5 py-1 rounded-full bg-surface-container-low hover:bg-surface-container-high border border-outline-variant text-on-surface-variant text-xs transition-colors" type="button">
                深蹲膝盖内扣纠正
              </button>
</div>
</div>
<div class="flex items-center justify-between pt-2 border-t border-outline-variant/40 font-mono text-[11px] text-on-surface-variant">
<span>FitZen AI</span>
<span class="text-primary">{{ui.sending?'回复中':'等待输入'}}</span>
</div>
</div>
</div>
</section>
<!-- SECTION 2: 加载状态 (LOADING STATES) -->
<section class="flex flex-col gap-4">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2.5">
<span class="font-mono text-xs px-2 py-0.5 rounded bg-surface-container-lowest border border-outline-variant text-primary font-medium">SEC_02</span>
<h2 class="text-lg font-semibold text-on-surface">加载状态 (Loading States)</h2>
</div>
<span class="font-mono text-xs text-on-surface-variant hidden sm:inline">PIPELINE_IN_FLIGHT</span>
</div>
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 fz-module-grid">
<!-- 4. 骨架屏闪烁 -->
<div class="relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant min-h-[380px] overflow-hidden">
<div class="flex items-center justify-between w-full mb-4">
<span class="font-mono text-[11px] text-on-surface-variant">SKL_004 // TELEMETRY_MOUNTING</span>
<div class="w-12 h-3 rounded bg-surface-container-high animate-pulse"></div>
</div>
<!-- 模拟组件骨架屏 -->
<div class="space-y-4 w-full animate-pulse my-auto">
<div class="flex items-center justify-between">
<div class="space-y-2">
<div class="w-20 h-2.5 bg-surface-container-high rounded"></div>
<div class="w-32 h-6 bg-surface-container-low rounded"></div>
</div>
<div class="w-10 h-10 rounded-lg bg-surface-container-high"></div>
</div>
<!-- 图表条形骨架 -->
<div class="pt-2 pb-1">
<div class="flex items-end gap-1.5 h-16 w-full">
<div class="flex-1 bg-surface-container-high rounded-t" style="height: 40%;"></div>
<div class="flex-1 bg-surface-container-high rounded-t" style="height: 65%;"></div>
<div class="flex-1 bg-surface-container-high rounded-t" style="height: 30%;"></div>
<div class="flex-1 bg-surface-container-high rounded-t" style="height: 85%;"></div>
<div class="flex-1 bg-surface-container-high rounded-t" style="height: 50%;"></div>
<div class="flex-1 bg-surface-container-high rounded-t" style="height: 95%;"></div>
<div class="flex-1 bg-surface-container-high rounded-t" style="height: 70%;"></div>
<div class="flex-1 bg-surface-container-high rounded-t" style="height: 45%;"></div>
</div>
</div>
<!-- 列表行骨架 -->
<div class="space-y-2">
<div class="flex items-center justify-between p-2 rounded bg-surface-container-low/60">
<div class="flex items-center gap-2">
<div class="w-4 h-4 rounded-full bg-surface-container-high"></div>
<div class="w-24 h-2 bg-surface-container-high rounded"></div>
</div>
<div class="w-10 h-2 bg-surface-container-high rounded"></div>
</div>
<div class="flex items-center justify-between p-2 rounded bg-surface-container-low/40">
<div class="flex items-center gap-2">
<div class="w-4 h-4 rounded-full bg-surface-container-high"></div>
<div class="w-32 h-2 bg-surface-container-high rounded"></div>
</div>
<div class="w-8 h-2 bg-surface-container-high rounded"></div>
</div>
</div>
</div>
<div class="flex items-center justify-between pt-3 border-t border-outline-variant/40 font-mono text-[11px] text-on-surface-variant">
<span>PARSING_FRAMEWORKS</span>
<span class="text-primary font-semibold animate-pulse">{{ui.loading?'同步中…':'同步空闲'}}</span>
</div>
</div>
<!-- 5. FitZen AI 跳动三圆点思考 -->
<div class="relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant min-h-[380px]">
<div class="flex items-center justify-between w-full">
<span class="font-mono text-[11px] text-on-surface-variant">AI_005 // INFERENCE_RUNNING</span>
<span class="font-mono text-[11px] text-primary">{{ui.sending?'回复中':'空闲'}}</span>
</div>
<div class="flex flex-col items-center justify-center my-auto py-6 text-center">
<!-- 脉冲氛围光与三点动画 -->
<div class="relative flex items-center justify-center mb-6">
<div class="w-20 h-20 rounded-full bg-surface-container blur-xl absolute pointer-events-none"></div>
<div class="relative flex items-center gap-2 px-4 py-2.5 rounded-full bg-surface-container-low border border-outline-variant">
<span class="w-2.5 h-2.5 rounded-full bg-primary animate-[bounce_1s_infinite_100ms] shadow-sm"></span>
<span class="w-2.5 h-2.5 rounded-full bg-primary animate-[bounce_1s_infinite_250ms] shadow-sm"></span>
<span class="w-2.5 h-2.5 rounded-full bg-primary animate-[bounce_1s_infinite_400ms] shadow-sm"></span>
</div>
</div>
<p class="text-base font-semibold text-on-surface tracking-tight mb-1.5"> {{ui.sending?'AI 正在生成回复…':'AI 当前空闲'}} </p>
<p class="text-xs text-on-surface-variant max-w-xs leading-relaxed"> 根据当前会话处理请求 </p>
</div>
<!-- 内联终端条 -->
<div class="p-2.5 rounded-lg bg-background border border-outline-variant font-mono text-[11px] text-on-surface-variant flex items-center justify-between">
<div class="flex items-center gap-1.5 truncate">
<span class="text-primary">&gt;</span>
<span class="truncate">{{ui.chatError||'暂无会话错误'}}</span>
</div>
<span class="text-on-surface-variant shrink-0 ml-2">—</span>
</div>
</div>
<!-- 6. 按钮加载状态展示 (Spinner / Disabled) -->
<div class="relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant min-h-[380px]">
<div class="flex items-center justify-between w-full">
<span class="font-mono text-[11px] text-on-surface-variant">CTL_006 // ATOMIC_BUTTONS</span>
<span class="font-mono text-[11px] text-on-surface-variant">STATES</span>
</div>
<div class="flex flex-col gap-4 my-auto py-2">
<!-- 正在同步 / 旋转 Spinner -->
<div class="flex flex-col gap-1.5">
<span class="font-mono text-[11px] text-on-surface-variant">同步进行中 (Loading Spinner)</span>
<button class="w-full py-2.5 px-4 rounded-lg bg-primary text-on-primary font-medium text-xs sm:text-sm flex items-center justify-center gap-2 shadow-sm cursor-wait opacity-95" :disabled="true" type="button">
<svg class="animate-spin h-4 w-4 text-on-surface" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3"></circle>
<path class="opacity-90" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" fill="currentColor"></path>
</svg>
<span>{{ui.loading?'数据同步中…':'同步完成'}}</span>
</button>
</div>
<!-- 只读锁死态 / Disabled -->
<div class="flex flex-col gap-1.5">
<span class="font-mono text-[11px] text-on-surface-variant">只读锁死态 (Disabled State)</span>
<button class="w-full py-2.5 px-4 rounded-lg bg-surface-container-low border border-outline-variant text-on-surface-variant font-medium text-xs sm:text-sm flex items-center justify-center gap-2 cursor-not-allowed" :disabled="true" type="button">
<span class="material-symbols-outlined text-base">lock</span>
<span>同步流水线挂起</span>
</button>
</div>
<!-- 次级常规态 -->
<div class="flex flex-col gap-1.5">
<span class="font-mono text-[11px] text-on-surface-variant">次级操作就绪 (Idle / Secondary)</span>
<button class="w-full py-2.5 px-4 rounded-lg bg-surface-container-low hover:bg-surface-container-high border border-outline-variant text-on-surface-variant font-medium text-xs sm:text-sm flex items-center justify-center gap-2 transition-colors" type="button">
<span class="material-symbols-outlined text-base">refresh</span>
<span>强制刷新管道缓存</span>
</button>
</div>
</div>
<div class="flex items-center justify-between pt-2 border-t border-outline-variant/40 font-mono text-[11px] text-on-surface-variant">
<span>按需刷新</span>
<span>{{ui.loading?'BUSY':'IDLE'}}</span>
</div>
</div>
</div>
</section>
<!-- SECTION 3: 异常与错误状态 (ERROR STATES) -->
<section class="flex flex-col gap-4">
<div class="flex items-center justify-between">
<div class="flex items-center gap-2.5">
<span class="font-mono text-xs px-2 py-0.5 rounded bg-error-container border border-error/20 text-error font-medium">SEC_03</span>
<h2 class="text-lg font-semibold text-on-surface">异常与错误状态 (Error States)</h2>
</div>
<span class="font-mono text-xs text-error hidden sm:inline">FAULT_HANDLING_BRANCHES</span>
</div>
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 fz-module-grid">
<!-- 7. 网络连接错误 -->
<div class="group relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant hover:border-outline-variant transition-colors min-h-[380px]">
<div class="flex items-center justify-between w-full">
<span class="font-mono text-[11px] text-error">ERR_503 // NETWORK_DISCONNECT</span>
<span class="w-2 h-2 rounded-full bg-error animate-ping"></span>
</div>
<div class="flex flex-col items-center text-center my-auto px-2 py-4">
<div class="relative w-16 h-16 rounded-xl bg-surface-container-lowest border border-outline-variant flex items-center justify-center mb-4 text-error shadow-sm">
<!-- 断网图标 -->
<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" viewBox="0 0 24 24">
<line x1="2" x2="22" y1="2" y2="22"></line>
<path d="M8.5 16.5a5 5 0 0 1 7 0"></path>
<path d="M2 8.82a15 15 0 0 1 4.17-2.65"></path>
<path d="M10.66 5c4.01-.36 8.14.9 11.34 3.76"></path>
<path d="M16.85 11.25a10 10 0 0 1 2.22 1.68"></path>
<path d="M5 13a10 10 0 0 1 5.24-2.76"></path>
<line x1="12" x2="12.01" y1="20" y2="20"></line>
</svg>
<div class="absolute inset-0 rounded-xl bg-error/5 pointer-events-none"></div>
</div>
<h3 class="text-base font-semibold text-on-surface mb-1.5">{{ui.online?'数据连接正常':'数据连接失败'}}</h3>
<p class="text-xs text-on-surface-variant max-w-xs leading-relaxed"> {{ui.loadError||'训练、饮食和个人资料已同步。'}} </p>
</div>
<div class="w-full pt-4">
<button class="w-full py-2.5 px-4 rounded-lg bg-surface-container-low hover:bg-surface-container-high border border-outline-variant text-on-surface-variant font-medium text-xs sm:text-sm flex items-center justify-center gap-1.5 transition-colors" type="button">
<span class="material-symbols-outlined text-base">sync_problem</span>
<span>重新连接</span>
</button>
</div>
</div>
<!-- 8. 404 找不到页面 -->
<div class="group relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant hover:border-outline-variant transition-colors min-h-[380px]">
<div class="flex items-center justify-between w-full">
<span class="font-mono text-[11px] text-on-surface-variant">HTTP_404 // ROUTE_NOT_FOUND</span>
<span class="w-1.5 h-1.5 rounded-full bg-surface-container-highest"></span>
</div>
<div class="flex flex-col items-center text-center my-auto px-2 py-2">
<div class="relative mb-2 select-none">
<span class="text-6xl font-black tracking-tighter text-on-surface-variant">404</span>
<div class="absolute inset-0 flex items-center justify-center">
<span class="font-mono text-[10px] text-primary bg-surface-container-lowest border border-outline-variant px-2 py-0.5 rounded shadow">UNRESOLVED_URI</span>
</div>
</div>
<h3 class="text-base font-semibold text-on-surface mb-1.5">页面导航</h3>
<p class="text-xs text-on-surface-variant max-w-xs leading-relaxed"> 从导航进入对应页面，或返回首页。 </p>
</div>
<div class="w-full pt-4">
<button class="w-full py-2.5 px-4 rounded-lg bg-primary hover:bg-primary/90 text-on-primary font-medium text-xs sm:text-sm flex items-center justify-center gap-1.5 transition-all shadow-sm active:scale-[0.99]" type="button">
<span class="material-symbols-outlined text-base">home</span>
<span>返回首页</span>
</button>
</div>
</div>
<!-- 9. 权限拒绝错误 -->
<div class="group relative flex flex-col justify-between p-6 rounded-xl bg-surface-container-lowest border border-outline-variant hover:border-outline-variant transition-colors min-h-[380px]">
<div class="flex items-center justify-between w-full">
<span class="font-mono text-[11px] text-error">ERR_403 // PERMISSION_DENIED</span>
<span class="w-1.5 h-1.5 rounded-full bg-error"></span>
</div>
<div class="flex flex-col items-center text-center my-auto px-2 py-4">
<div class="relative w-16 h-16 rounded-xl bg-surface-container-lowest border border-outline-variant flex items-center justify-center mb-4 text-on-surface-variant group-hover:text-on-surface transition-colors">
<!-- 挂锁图标 -->
<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" viewBox="0 0 24 24">
<rect height="11" rx="2" ry="2" width="18" x="3" y="11"></rect>
<path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
</svg>
<div class="absolute inset-0 rounded-xl bg-surface-container pointer-events-none"></div>
</div>
<h3 class="text-base font-semibold text-on-surface mb-1.5">账户服务</h3>
<p class="text-xs text-on-surface-variant max-w-xs leading-relaxed"> 当前为本地账户，尚未配置权限服务。 </p>
</div>
<div class="w-full pt-4">
<button class="w-full py-2.5 px-4 rounded-lg bg-surface-container-low hover:bg-surface-container-high border border-outline-variant text-on-surface-variant font-medium text-xs sm:text-sm flex items-center justify-center gap-1.5 transition-colors" type="button">
<span class="material-symbols-outlined text-base">lock_open</span>
<span>返回登录</span>
</button>
</div>
</div>
</div>
</section>
<!-- 底部状态与诊断元数据 -->
<footer class="p-4 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col md:flex-row items-center justify-between gap-4 text-xs font-mono text-on-surface-variant">
<div class="flex items-center gap-4 flex-wrap">
<div class="flex items-center gap-1.5">
<span class="text-on-surface-variant">SYSTEM_NODE:</span>
<span class="text-on-surface-variant font-medium">LOCAL</span>
</div>
<div class="w-px h-3 bg-surface-container-highest hidden sm:block"></div>
<div class="flex items-center gap-1.5">
<span class="text-on-surface-variant">DIAGNOSTIC_MODE:</span>
<span class="text-primary font-medium">LIVE_DATA</span>
</div>
<div class="w-px h-3 bg-surface-container-highest hidden sm:block"></div>
<div class="flex items-center gap-1.5">
<span class="text-on-surface-variant">HEALTH:</span>
<span class="text-on-surface-variant font-medium">{{ui.online?'CONNECTED':'DISCONNECTED'}}</span>
</div>
</div>
<div class="flex items-center gap-1.5">
<span class="text-on-surface-variant">TIMESTAMP:</span>
<span class="text-on-surface-variant" id="system-live-clock">{{ui.data?.fetched_at||'尚未同步'}}</span>
</div>
</footer>
</main>

</div>
</template>
<style scoped>
      html, .screen-status { margin: 0; padding: 0; }
    .screen-status { overscroll-behavior: none; }
    ::-webkit-scrollbar { display: none; }

</style>
