<script setup lang="ts">
import {computed,ref,watch} from 'vue'
import {useFitzen} from '../state'
import {DeleteButton} from '../../components/ui'
import {fmt,todayKey,pct,sumMeals,editRecord,removeMeal,clearMealDeleteConfirmation} from '../metrics'
const {ui,openFood}=useFitzen()
const userName=computed(()=>ui.profile.display_name||'朋友'),userGoal=computed(()=>ui.profile.goal||'尚未设置')
const selectedDate=ref(todayKey.value)
watch(todayKey,(date,old)=>{if(selectedDate.value===old)selectedDate.value=date})
const filteredNutrition=computed(()=>ui.nutrition.filter((n:any)=>(n.date_key||n.date?.slice(0,10))===selectedDate.value))
const targetKcal=computed(()=>ui.targets.calories_target??null),targetProtein=computed(()=>ui.targets.protein_target_g??null),targetCarbs=computed(()=>ui.targets.carbs_target_g??null),targetFat=computed(()=>ui.targets.fat_target_g??null)
const todayKcal=computed(()=>sumMeals(filteredNutrition.value,'calories')),todayProtein=computed(()=>sumMeals(filteredNutrition.value,'protein_g')),todayCarbs=computed(()=>sumMeals(filteredNutrition.value,'carbs_g')),todayFat=computed(()=>sumMeals(filteredNutrition.value,'fat_g'))
const kcalPct=computed(()=>pct(todayKcal.value,targetKcal.value)),proteinPct=computed(()=>pct(todayProtein.value,targetProtein.value)),carbsPct=computed(()=>pct(todayCarbs.value,targetCarbs.value)),fatPct=computed(()=>pct(todayFat.value,targetFat.value))
const kcalOffset=computed(()=>(1-Math.min(100,kcalPct.value||0)/100)*163.36),proteinOffset=computed(()=>(1-Math.min(100,proteinPct.value||0)/100)*163.36),carbsOffset=computed(()=>(1-Math.min(100,carbsPct.value||0)/100)*163.36),fatOffset=computed(()=>(1-Math.min(100,fatPct.value||0)/100)*163.36)
const breakfastItems=computed(()=>filteredNutrition.value.filter(n=>['breakfast','早餐','早上加餐'].includes(n.meal_type||''))),lunchItems=computed(()=>filteredNutrition.value.filter(n=>['lunch','午餐'].includes(n.meal_type||''))),dinnerItems=computed(()=>filteredNutrition.value.filter(n=>['dinner','晚餐'].includes(n.meal_type||''))),snackItems=computed(()=>filteredNutrition.value.filter(n=>!['breakfast','早餐','早上加餐','lunch','午餐','dinner','晚餐'].includes(n.meal_type||'')))
const breakfastKcal=computed(()=>sumMeals(breakfastItems.value,'calories')),lunchKcal=computed(()=>sumMeals(lunchItems.value,'calories')),dinnerKcal=computed(()=>sumMeals(dinnerItems.value,'calories')),snackKcal=computed(()=>sumMeals(snackItems.value,'calories'))
const breakfastPct=computed(()=>pct(breakfastKcal.value,todayKcal.value)),lunchPct=computed(()=>pct(lunchKcal.value,todayKcal.value)),dinnerPct=computed(()=>pct(dinnerKcal.value,todayKcal.value)),snackPct=computed(()=>pct(snackKcal.value,todayKcal.value))
const gap=(target:any,current:any)=>target==null||current==null?null:Math.max(0,target-current)
const mainMealCards=computed(()=>[
 {type:'早餐',icon:'🍳',items:breakfastItems.value,kcal:breakfastKcal.value},
 {type:'午餐',icon:'🥩',items:lunchItems.value,kcal:lunchKcal.value},
 {type:'晚餐',icon:'🥗',items:dinnerItems.value,kcal:dinnerKcal.value},
])
function mealFoods(meal:any){return meal.items?.length?meal.items:[{}]}
function mealTimeLabel(value?:string){
 if(!value)return '时间未记录'
 const parsed=new Date(value)
 if(!Number.isFinite(parsed.getTime()))return '时间未记录'
 return parsed.toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false})
}
function addToMeal(items:any[],mealType:string){
 clearMealDeleteConfirmation()
 const existing=items.find(item=>item.source==='ai_food_log')
 if(existing){editRecord(existing,true);return}
 ui.meal=mealType;openFood(selectedDate.value)
}
function openNewSnack(){ui.meal='加餐';openFood(selectedDate.value)}

</script>

<template>
<div class="screen-nutrition bg-background font-body-md text-on-surface antialiased selection:bg-primary selection:text-on-primary">
  <!-- 侧边栏 -->


  <div class=" flex flex-col min-h-screen">
    <!-- 顶部导航 -->


    <main class="w-full pt-0 bg-background pb-20 lg:pb-12">
      <div class="fz-page-inset max-w-7xl mx-auto">
        <div class="fz-section-stack flex flex-col w-full">
          <!-- 标题区 -->
          <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-space-md">
            <div class="flex flex-col gap-1">
              <div class="flex items-center gap-space-xs">
                <span class="font-headline-lg text-headline-lg font-bold tracking-tight text-on-surface">饮食记录</span>
                <span class="inline-flex items-center gap-1 px-space-xs py-0.5 rounded bg-surface-container text-primary font-code-inline text-code-inline">
                  <span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>实时追踪
                </span>
              </div>
              <p class="font-body-md text-body-md text-on-surface-variant">精准宏量营养素追踪 · 依据已保存的饮食记录</p>
            </div>
            <div class="flex items-center gap-space-sm self-start lg:self-auto">
              <!-- 日期选择器 -->
              <div class="flex items-center gap-2 px-space-md py-2 rounded-lg bg-surface-container border border-outline-variant/30">
                <span class="material-symbols-outlined text-[18px] text-primary">calendar_today</span>
                <input :min="ui.data?.start_date" :max="todayKey" type="date" v-model="selectedDate" class="bg-transparent text-on-surface font-label-md text-label-md focus:outline-none">
              </div>
              <button class="flex items-center gap-space-xs px-space-md py-2 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface font-label-md text-label-md transition-colors shadow-sm">
                <span class="material-symbols-outlined text-[18px] text-primary">photo_camera</span>
                <span class="">拍照识别食物</span>
              </button>
              <button class="flex items-center gap-space-xs px-space-md py-2 rounded-lg bg-primary-container text-on-primary-container hover:bg-primary hover:text-on-primary font-label-md text-label-md font-semibold transition-all shadow-sm">
                <span class="material-symbols-outlined text-[18px]">add</span>
                <span class="">添加食物记录</span>
              </button>
            </div>
          </div>

          <!-- 营养素汇总卡片 -->
          <div class="rounded-xl bg-surface-container-lowest p-space-lg shadow-md flex flex-col gap-space-lg relative overflow-hidden">
            <div class="absolute -right-20 -top-20 w-72 h-72 rounded-full bg-surface-container blur-3xl pointer-events-none"></div>
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 z-10">
              <div class="flex items-center gap-space-xs">
                <span class="material-symbols-outlined text-primary text-[20px]">donut_large</span>
                <h2 class="font-headline-sm text-headline-sm font-semibold text-on-surface">今日营养素摄入汇总</h2>
                <span class="font-label-sm text-label-sm px-2 py-0.5 rounded bg-surface-container text-on-surface-variant">目标：{{userGoal}}</span>
              </div>
              <div class="flex items-center gap-space-sm text-on-surface-variant font-body-sm text-body-sm">
                <span class="">后端估算目标:</span>
                <span class="font-metric-display text-headline-sm font-semibold text-primary">{{ fmt(targetKcal) }}</span>
                <span class="text-on-surface-variant">kcal</span>
              </div>
            </div>

            <!-- 四个营养素卡片 -->
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 fz-module-grid z-10">
              <!-- 总能量 -->
              <div class="flex items-center gap-space-md p-space-md rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors">
                <div class="relative w-16 h-16 shrink-0 flex items-center justify-center">
                  <svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                    <circle class="text-surface-variant" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-width="5"></circle>
                    <circle class="text-primary" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-dasharray="163.36" :stroke-dashoffset="kcalOffset" stroke-linecap="round" stroke-width="5"></circle>
                  </svg>
                  <span class="absolute font-label-md text-label-md font-bold text-on-surface">{{ fmt(kcalPct) }}%</span>
                </div>
                <div class="flex flex-col min-w-0">
                  <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">总能量摄入</span>
                  <div class="flex items-baseline gap-1 mt-0.5">
                    <span class="font-headline-sm text-headline-sm font-bold text-on-surface">{{ fmt(todayKcal) }}</span>
                    <span class="font-body-sm text-body-sm text-on-surface-variant">/ {{ fmt(targetKcal) }} kcal</span>
                  </div>
                  <span class="font-label-sm text-label-sm text-primary mt-1">剩余可用 {{ fmt(gap(targetKcal,todayKcal)) }} kcal</span>
                </div>
              </div>

              <!-- 蛋白质 -->
              <div class="flex items-center gap-space-md p-space-md rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors">
                <div class="relative w-16 h-16 shrink-0 flex items-center justify-center">
                  <svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                    <circle class="text-surface-variant" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-width="5"></circle>
                    <circle class="text-secondary" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-dasharray="163.36" :stroke-dashoffset="proteinOffset" stroke-linecap="round" stroke-width="5"></circle>
                  </svg>
                  <span class="absolute font-label-md text-label-md font-bold text-on-surface">{{ fmt(proteinPct) }}%</span>
                </div>
                <div class="flex flex-col min-w-0">
                  <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">蛋白质</span>
                  <div class="flex items-baseline gap-1 mt-0.5">
                    <span class="font-headline-sm text-headline-sm font-bold text-secondary">{{ fmt(todayProtein) }}</span>
                    <span class="font-body-sm text-body-sm text-on-surface-variant">/ {{ fmt(targetProtein) }} g</span>
                  </div>
                  <span class="font-label-sm text-label-sm text-on-surface-variant mt-1">按个人资料估算</span>
                </div>
              </div>

              <!-- 碳水 -->
              <div class="flex items-center gap-space-md p-space-md rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors">
                <div class="relative w-16 h-16 shrink-0 flex items-center justify-center">
                  <svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                    <circle class="text-surface-variant" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-width="5"></circle>
                    <circle class="text-primary" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-dasharray="163.36" :stroke-dashoffset="carbsOffset" stroke-linecap="round" stroke-width="5"></circle>
                  </svg>
                  <span class="absolute font-label-md text-label-md font-bold text-on-surface">{{ fmt(carbsPct) }}%</span>
                </div>
                <div class="flex flex-col min-w-0">
                  <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">碳水化合物</span>
                  <div class="flex items-baseline gap-1 mt-0.5">
                    <span class="font-headline-sm text-headline-sm font-bold text-on-surface">{{ fmt(todayCarbs) }}</span>
                    <span class="font-body-sm text-body-sm text-on-surface-variant">/ {{ fmt(targetCarbs) }} g</span>
                  </div>
                  <span class="font-label-sm text-label-sm text-on-surface-variant mt-1">剩余 {{ fmt(gap(targetCarbs,todayCarbs)) }} g</span>
                </div>
              </div>

              <!-- 脂肪 -->
              <div class="flex items-center gap-space-md p-space-md rounded-lg bg-surface-container hover:bg-surface-container-high transition-colors">
                <div class="relative w-16 h-16 shrink-0 flex items-center justify-center">
                  <svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                    <circle class="text-surface-variant" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-width="5"></circle>
                    <circle class="text-tertiary-fixed-dim" cx="32" cy="32" fill="transparent" r="26" stroke="currentColor" stroke-dasharray="163.36" :stroke-dashoffset="fatOffset" stroke-linecap="round" stroke-width="5"></circle>
                  </svg>
                  <span class="absolute font-label-md text-label-md font-bold text-on-surface">{{ fmt(fatPct) }}%</span>
                </div>
                <div class="flex flex-col min-w-0">
                  <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">优质脂肪</span>
                  <div class="flex items-baseline gap-1 mt-0.5">
                    <span class="font-headline-sm text-headline-sm font-bold text-on-surface">{{ fmt(todayFat) }}</span>
                    <span class="font-body-sm text-body-sm text-on-surface-variant">/ {{ fmt(targetFat) }} g</span>
                  </div>
                  <span class="font-label-sm text-label-sm text-on-surface-variant mt-1">剩余 {{ fmt(gap(targetFat,todayFat)) }} g</span>
                </div>
              </div>
            </div>

            <!-- 能量节奏分布 -->
            <div class="flex flex-col gap-2 pt-2 z-10">
              <div class="flex items-center justify-between text-on-surface-variant font-label-sm text-label-sm">
                <span class="uppercase tracking-wider">全天能量摄入节奏分布</span>
                <span class="font-code-inline text-code-inline text-primary">全天摄入节律</span>
              </div>
              <div class="h-3 w-full rounded-full bg-surface-container-high overflow-hidden flex gap-0.5 p-0.5">
                <div class="h-full rounded-l-full bg-primary-container" :style="{ width: breakfastPct + '%' }" :title="'早餐: ' + breakfastPct + '%'"></div>
                <div class="h-full bg-primary" :style="{ width: lunchPct + '%' }" :title="'午餐: ' + lunchPct + '%'"></div>
                <div class="h-full bg-secondary-fixed-dim" :style="{ width: dinnerPct + '%' }" :title="'晚餐: ' + dinnerPct + '%'"></div>
                <div class="h-full rounded-r-full bg-surface-variant" :style="{ width: snackPct + '%' }" :title="'加餐: ' + snackPct + '%'"></div>
              </div>
              <div class="flex items-center justify-between text-on-surface-variant font-label-sm text-label-sm px-1">
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-primary-container"></span><span>早 {{ fmt(breakfastPct) }}% ({{ fmt(breakfastKcal) }} kcal)</span></div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-primary"></span><span>午 {{ fmt(lunchPct) }}% ({{ fmt(lunchKcal) }} kcal)</span></div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-secondary-fixed-dim"></span><span>晚 {{ fmt(dinnerPct) }}% ({{ fmt(dinnerKcal) }} kcal)</span></div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-surface-variant"></span><span>加餐 {{ fmt(snackPct) }}%</span></div>
              </div>
            </div>
          </div>

          <!-- 三餐记录：按整餐展示，用餐时间只显示一次 -->
          <div class="grid grid-cols-1 lg:grid-cols-3 gap-space-lg items-stretch">
            <div v-for="card in mainMealCards" :key="card.type" class="flex flex-col rounded-xl bg-surface-container-lowest p-space-md shadow-md">
              <div class="flex items-center justify-between pb-2">
                <div class="flex flex-col">
                  <div class="flex items-center gap-space-xs">
                    <span class="text-[18px]">{{card.icon}}</span>
                    <h3 class="font-headline-sm text-headline-sm font-semibold text-on-surface">{{card.type}}</h3>
                  </div>
                  <span class="text-primary font-label-sm text-label-sm font-medium mt-1">{{fmt(card.kcal)}} kcal</span>
                </div>
                <button :aria-label="'向'+card.type+'添加食物'" @click.stop="addToMeal(card.items,card.type)" class="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-primary hover:text-on-surface transition-colors">
                  <span class="material-symbols-outlined text-[20px]">add</span>
                </button>
              </div>

              <div v-if="!card.items.length" class="text-on-surface-variant text-sm py-4 text-center">还没有{{card.type}}记录</div>
              <div v-else class="flex flex-col gap-space-sm mt-space-xs">
                <section v-for="(meal,mIdx) in card.items" :key="meal.id||mIdx" class="rounded-lg border border-outline-variant/30 bg-surface-container overflow-hidden">
                  <header class="flex items-center justify-between gap-2 px-space-sm py-space-xs border-b border-outline-variant/30 bg-surface-container-high/40">
                    <div class="flex items-center gap-2 min-w-0 text-[11px] text-on-surface-variant">
                      <span class="material-symbols-outlined text-[15px] text-primary">schedule</span>
                      <strong class="text-on-surface whitespace-nowrap">{{mealTimeLabel(meal.date)}}</strong>
                      <span>· {{meal.items?.length||1}} 项食物</span>
                    </div>
                    <div class="flex items-center gap-1 shrink-0">
                      <button v-if="meal.source==='ai_food_log'" @click.stop="clearMealDeleteConfirmation();editRecord(meal,true)" aria-label="向本餐添加食物" title="向本餐添加食物" class="p-1 text-primary hover:bg-surface-container rounded transition-colors"><span class="material-symbols-outlined text-[16px]">add_circle</span></button>
                      <button @click.stop="clearMealDeleteConfirmation();editRecord(meal,true)" aria-label="编辑整餐" title="编辑整餐" class="p-1 hover:text-primary transition-colors"><span class="material-symbols-outlined text-[16px]">edit</span></button>
                      <DeleteButton :action="()=>removeMeal(meal)" :subject="meal.meal_type||'整餐'" compact />
                    </div>
                  </header>
                  <div class="divide-y divide-outline-variant/20">
                    <div v-for="(food,fIdx) in mealFoods(meal)" :key="(food.id||fIdx)" class="px-space-sm py-space-xs">
                      <div class="flex items-center justify-between gap-2">
                        <div class="min-w-0">
                          <p class="font-label-md text-label-md text-on-surface font-medium truncate">{{food.food_name||food.raw_food_name||meal.description||'食物记录'}}</p>
                          <p v-if="food.weight_g" class="font-body-sm text-body-sm text-on-surface-variant">{{food.weight_g}}g</p>
                        </div>
                        <span class="text-xs text-on-surface whitespace-nowrap">{{fmt(food.calories??meal.calories)}} kcal</span>
                      </div>
                      <p class="mt-1 text-[11px] text-on-surface-variant truncate">
                        <template v-if="food.protein_g||meal.protein_g">蛋白质 {{fmt(food.protein_g??meal.protein_g,1)}}g</template>
                        <template v-if="food.carbs_g||meal.carbs_g"> · 碳水 {{fmt(food.carbs_g??meal.carbs_g,1)}}g</template>
                        <template v-if="food.fat_g||meal.fat_g"> · 脂肪 {{fmt(food.fat_g??meal.fat_g,1)}}g</template>
                      </p>
                    </div>
                  </div>
                </section>
              </div>
            </div>
          </div>

          <!-- 加餐 + AI 提示 -->
          <div class="grid grid-cols-1 lg:grid-cols-3 fz-module-grid">
            <div class="lg:col-span-1 rounded-xl bg-surface-container-lowest p-space-md shadow-md flex flex-col gap-space-sm">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-space-xs">
                  <span class="material-symbols-outlined text-primary text-[20px]">medication</span>
                  <span class="font-headline-sm text-headline-sm font-semibold text-on-surface">加餐与补剂</span>
                </div>
                <button @click.stop="openNewSnack" aria-label="添加加餐" class="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-primary hover:text-on-surface transition-colors"><span class="material-symbols-outlined text-[20px]">add</span></button>
              </div>
              <div v-if="!snackItems.length" class="text-on-surface-variant text-sm py-2 text-center">暂无加餐记录</div>
              <section v-for="(meal,idx) in snackItems" :key="meal.id||idx" class="rounded-lg border border-outline-variant/30 bg-surface-container overflow-hidden">
                <header class="flex items-center justify-between gap-2 px-space-xs py-1.5 border-b border-outline-variant/30">
                  <span class="flex items-center gap-1 text-[11px] text-on-surface-variant"><span class="material-symbols-outlined text-[14px] text-primary">schedule</span>{{mealTimeLabel(meal.date)}} · {{meal.items?.length||1}} 项</span>
                  <div class="flex items-center gap-1">
                    <button v-if="meal.source==='ai_food_log'" @click.stop="clearMealDeleteConfirmation();editRecord(meal,true)" aria-label="向本餐添加食物" class="p-1 text-primary"><span class="material-symbols-outlined text-[15px]">add_circle</span></button>
                    <button @click.stop="clearMealDeleteConfirmation();editRecord(meal,true)" aria-label="编辑整餐" class="p-1 hover:text-primary"><span class="material-symbols-outlined text-[15px]">edit</span></button>
                    <DeleteButton :action="()=>removeMeal(meal)" :subject="meal.meal_type||'整餐'" compact />
                  </div>
                </header>
                <div v-for="(food,fIdx) in mealFoods(meal)" :key="food.id||fIdx" class="flex items-center justify-between gap-2 px-space-xs py-1.5 border-b last:border-b-0 border-outline-variant/20 text-xs">
                  <span class="truncate text-on-surface">{{food.food_name||food.raw_food_name||meal.description||'加餐'}}</span>
                  <span class="text-on-surface-variant whitespace-nowrap">{{fmt(food.calories??meal.calories)}} kcal</span>
                </div>
              </section>
            </div>

            <div class="lg:col-span-2 rounded-xl bg-surface-container-lowest p-space-md shadow-md flex flex-col sm:flex-row sm:items-center justify-between gap-space-md relative overflow-hidden">
              <div class="flex items-start gap-space-md z-10">
                <div class="p-3 rounded-xl bg-surface-container text-primary shrink-0 flex items-center justify-center">
                  <span class="material-symbols-outlined text-[24px]">smart_toy</span>
                </div>
                <div class="flex flex-col gap-1">
                  <div class="flex items-center gap-space-xs">
                    <span class="font-label-md text-label-md text-primary uppercase font-bold tracking-wider">AI 智能营养引擎提示</span>
                    <span class="font-label-sm text-label-sm px-1.5 py-0.2 rounded bg-surface-container text-on-surface-variant">已记录营养汇总</span>
                  </div>
                  <p class="font-body-md text-body-md text-on-surface">
                    今日蛋白质已摄入 <strong class="text-primary font-bold">{{ fmt(proteinPct) }}%</strong>，
                    <template v-if="targetProtein!=null&&todayProtein!=null&&todayProtein < targetProtein">距目标尚差 <strong>{{ fmt(gap(targetProtein,todayProtein)) }}g</strong>，建议补充优质蛋白。</template>
                    <template v-else-if="targetProtein!=null&&todayProtein!=null">已达标，继续保持！</template>
                  </p>
                </div>
              </div>
              <button class="shrink-0 px-space-md py-2.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface font-label-md text-label-md transition-colors flex items-center justify-center gap-1 z-10">
                <span class="">采纳并一键添加</span>
                <span class="material-symbols-outlined text-[16px]">arrow_forward</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- 手机端底部导航 -->

  </div>
</div>
</template>

<style scoped>
html, .screen-nutrition { margin: 0; padding: 0; }
.screen-nutrition { overscroll-behavior: none; }
main > :first-child { margin-top: 0 !important; }
main > :last-child { margin-bottom: 0 !important; }
::-webkit-scrollbar { display: none; }
</style>
