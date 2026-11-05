<script setup lang="ts">
import {computed,watch,nextTick,ref,onBeforeUnmount} from 'vue'
import DeleteButton from './DeleteButton.vue'
import {deleteConfirmation} from './delete-confirmation'
import {api,findFoodSuggestions,customFoods,ensureCustomFoods,ensureFoodCatalog,notify,saveCustomFood,deleteLibraryFood,updateLibraryFood,useFitzen,foodsCatalog,type FoodDraft} from '../../fitzen/state'

type SourceTab='quick'|'custom'|'ai'
type CatalogFood={name:string;category:string;preparation_state:string;kcal:number;protein:number;carbs:number;fat:number;fiber?:number;id?:number;source?:'local'|'custom'}
type EditableFood={name:string;grams:number;kcal:number;protein:number;carbs:number;fat:number;fiber:number}

const {ui,addFood,addNutritionFood,saveFood}=useFitzen()
const dialog=ref<HTMLElement|null>(null)
const sourceTab=ref<SourceTab>('quick')
const sourceTabs=[{id:'quick' as const,label:'常用推荐',icon:'recommend'},{id:'custom' as const,label:'自定义',icon:'edit_note'},{id:'ai' as const,label:'问 AI',icon:'auto_awesome'}]
const nutrientFields=[{key:'protein' as const,label:'蛋白质 g'},{key:'carbs' as const,label:'碳水 g'},{key:'fat' as const,label:'脂肪 g'},{key:'fiber' as const,label:'膳食纤维 g'}]
const catalog=ref<CatalogFood[]>([])
const catalogSearch=ref('')
const catalogLoading=ref(false)
const catalogOpen=ref(false)
const selectedCategory=ref('全部')
const custom=ref<EditableFood>({name:'',grams:100,kcal:0,protein:0,carbs:0,fat:0,fiber:0})
const aiQuery=ref('')
const aiLoading=ref(false)
const aiResult=ref<EditableFood|null>(null)
const aiNotes=ref('')
const aiConfidence=ref('')
const aiResultPanel=ref<HTMLElement|null>(null)
const aiSaved=ref(false)
let previous:HTMLElement|null=null

const totals=computed(()=>ui.foods.reduce((a,f)=>({kcal:a.kcal+f.kcal*f.grams/100,protein:a.protein+f.protein*f.grams/100,carbs:a.carbs+f.carbs*f.grams/100,fat:a.fat+f.fat*f.grams/100}),{kcal:0,protein:0,carbs:0,fat:0}))
const libraryFoods=computed<CatalogFood[]>(()=>[...catalog.value,...customFoods.map(food=>({...food,category:'自定义',preparation_state:'用户自定义',source:'custom' as const}))])
const catalogCategories=computed(()=>['全部','自定义',...Array.from(new Set(libraryFoods.value.map(food=>food.category).filter(category=>Boolean(category)&&category!=='全部'&&category!=='自定义')))])
const filteredCatalog=computed(()=>{const query=catalogSearch.value.trim().toLowerCase();return libraryFoods.value.filter(food=>(selectedCategory.value==='全部'||food.category===selectedCategory.value)&&(!query||[food.name,food.category,food.preparation_state].some(value=>String(value||'').toLowerCase().includes(query))))})

const quickSearchResults=ref<CatalogFood[]>([]),quickSearchTerm=ref(''),quickSearchLoading=ref(false),quickSearchError=ref('')
let quickSearchTimer:ReturnType<typeof setTimeout>|null=null
let quickSearchController:AbortController|null=null
let quickSearchVersion=0
function cancelQuickSearch(){
 quickSearchVersion++;if(quickSearchTimer){clearTimeout(quickSearchTimer);quickSearchTimer=null}quickSearchController?.abort();quickSearchController=null
 quickSearchResults.value=[];quickSearchTerm.value='';quickSearchLoading.value=false;quickSearchError.value=''
}
function queueQuickSearch(event?:Event){
 if((event as InputEvent|undefined)?.isComposing)return
 cancelQuickSearch();const term=ui.foodSearch.trim();quickSearchTerm.value=term
 if(term){quickSearchLoading.value=true;quickSearchTimer=setTimeout(()=>{quickSearchTimer=null;void searchQuickFoods()},250)}
}
async function searchQuickFoods(){
 if(quickSearchTimer){clearTimeout(quickSearchTimer);quickSearchTimer=null}
 const term=ui.foodSearch.trim();if(!term){cancelQuickSearch();return}
 quickSearchController?.abort();const controller=new AbortController();quickSearchController=controller;const version=++quickSearchVersion
 quickSearchTerm.value=term;quickSearchResults.value=[];quickSearchLoading.value=true;quickSearchError.value=''
 try{const foods=await findFoodSuggestions(term,controller.signal);if(version===quickSearchVersion&&term===ui.foodSearch.trim())quickSearchResults.value=foods}
 catch(e){if(version===quickSearchVersion&&!controller.signal.aborted)quickSearchError.value=e instanceof Error&&e.name==='TimeoutError'?'搜索超时，请重试。':e instanceof Error?e.message:'食物搜索失败，请重试。'}
 finally{if(version===quickSearchVersion)quickSearchLoading.value=false}
}
function searchOnEnter(event:KeyboardEvent){if(event.isComposing||event.keyCode===229)return;event.preventDefault();void searchQuickFoods()}
function selectFoodSuggestion(food:CatalogFood){if(addCatalogFood(food)){ui.foodSearch='';cancelQuickSearch()}}

function clearFoodDeleteConfirmation(){deleteConfirmation.clear()}
function clearFoods(){ui.foods=[]}
function removeFood(index:number){
 clearFoodDeleteConfirmation();ui.foods.splice(index,1);ui.foodError=''
}
function close(){if(!ui.savingFood){clearFoodDeleteConfirmation();ui.foodOpen=false}}
function key(e:KeyboardEvent){if(e.key==='Escape'){e.stopPropagation();if(catalogOpen.value){closeCatalog();return}close()}if(e.key==='Tab'&&!catalogOpen.value){const nodes=dialog.value?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),textarea:not(:disabled),[tabindex="0"]');if(!nodes?.length)return;const first=nodes[0],last=nodes[nodes.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus()}}}
function sourceLabel(source?:string){return source==='ai'?'AI 估算':source==='custom'?'自定义':'本地库'}
function validFood(food:EditableFood){return Boolean(food.name.trim())&&[food.grams,food.kcal,food.protein,food.carbs,food.fat,food.fiber].every(Number.isFinite)&&food.grams>0&&food.kcal>=0&&food.protein>=0&&food.carbs>=0&&food.fat>=0&&food.fiber>=0}
function append(food:EditableFood,source:FoodDraft['source']){if(!validFood(food)){ui.foodError='请填写食物名称、有效克重和非负的每 100g 营养数据。';return false}addNutritionFood({...food,name:food.name.trim(),source});ui.foodError='';return true}
function addCatalogFood(food:CatalogFood){return append({name:food.name,grams:100,kcal:Number(food.kcal),protein:Number(food.protein),carbs:Number(food.carbs),fat:Number(food.fat),fiber:Number(food.fiber||0)},food.source==='custom'?'custom':'local')}
function customValues(food:EditableFood|{name:string;grams:number;kcal:number;protein:number;carbs:number;fat:number;fiber?:number}){
 return {name:food.name,grams:Number(food.grams||100),kcal:Number(food.kcal||0),protein:Number(food.protein||0),carbs:Number(food.carbs||0),fat:Number(food.fat||0),fiber:Number(food.fiber||0)}
}
async function saveCustomToLibrary(food:EditableFood){
 if(!validFood(food)){ui.foodError='请先填写完整的自定义食物营养数据。';return false}
 try{await saveCustomFood(customValues(food));ui.foodError='';notify('已保存到食物库');return true}
 catch(error){ui.foodError=error instanceof Error?error.message:'保存自定义食物失败';return false}
}
function addCustom(){if(append(custom.value,'custom')){void saveCustomToLibrary(custom.value);custom.value={name:'',grams:100,kcal:0,protein:0,carbs:0,fat:0,fiber:0}}}
function addCustomLibraryFood(food:any){append(customValues(food),'custom')}
async function askAi(){
 const description=aiQuery.value.trim();if(description.length<2){ui.foodError='请描述食物名称、品牌或烹饪方式。';return}
 aiLoading.value=true;ui.foodError='';aiResult.value=null;aiSaved.value=false
 try{
  const data=await api('/nutrition/food/estimate',{method:'POST',signal:AbortSignal.timeout(45000),body:JSON.stringify({description})}),p=data?.per_100g
  if(!p)throw new Error('AI 未返回可用营养数据，请稍后重试。')
  aiResult.value={name:data.food_name,grams:100,kcal:Number(p.kcal),protein:Number(p.protein_g),carbs:Number(p.carbs_g),fat:Number(p.fat_g),fiber:Number(p.fiber_g||0)}
  aiNotes.value=data.notes||'';aiConfidence.value=data.confidence||'medium';await nextTick();aiResultPanel.value?.scrollIntoView({behavior:'smooth',block:'nearest'})
 }catch(error){ui.foodError=error instanceof Error&&error.name==='TimeoutError'?'AI 查询超时，请稍后重试。':error instanceof Error?error.message:'AI 营养查询失败'}finally{aiLoading.value=false}
}
async function saveAiToCustom(){
 if(!aiResult.value)return
 if(await saveCustomToLibrary(aiResult.value))aiSaved.value=true
}
function addAi(){if(aiResult.value&&append(aiResult.value,'ai')){aiResult.value=null;aiQuery.value='';aiNotes.value='';aiConfidence.value='';aiSaved.value=false}}


watch(()=>ui.foodOpen,async open=>{if(open){cancelQuickSearch();ui.foodSearch='';clearFoodDeleteConfirmation();sourceTab.value='quick';void ensureFoodCatalog();void ensureCustomFoods();previous=document.activeElement as HTMLElement;await nextTick();dialog.value?.focus();document.body.style.overflow='hidden'}else{cancelQuickSearch();catalogOpen.value=false;document.body.style.overflow='';previous?.focus()}})
onBeforeUnmount(()=>{cancelQuickSearch();clearFoodDeleteConfirmation();document.body.style.overflow=''})

const catalogDialog=ref<HTMLElement|null>(null),libraryNameInput=ref<HTMLInputElement|null>(null)
const managingLibrary=ref(false),addingLibrary=ref(false),libraryBusy=ref(false),libraryError=ref('')
const editingLibraryFood=ref<CatalogFood|null>(null)
const newLibraryFood=ref<EditableFood>({name:'',grams:100,kcal:0,protein:0,carbs:0,fat:0,fiber:0})
const libraryFields=[{key:'kcal' as const,label:'热量 kcal',max:1200},{key:'protein' as const,label:'蛋白质 g',max:100},{key:'carbs' as const,label:'碳水 g',max:100},{key:'fat' as const,label:'脂肪 g',max:100},{key:'fiber' as const,label:'膳食纤维 g',max:100}]
let catalogPrevious:HTMLElement|null=null
function closeCatalog(){if(libraryBusy.value)return;catalogOpen.value=false;deleteConfirmation.clear();catalogPrevious?.focus()}
function toggleLibraryManagement(){if(libraryBusy.value)return;managingLibrary.value=!managingLibrary.value;addingLibrary.value=false;libraryError.value='';deleteConfirmation.clear()}
async function startLibraryAdd(){if(libraryBusy.value)return;editingLibraryFood.value=null;newLibraryFood.value={name:'',grams:100,kcal:0,protein:0,carbs:0,fat:0,fiber:0};managingLibrary.value=true;addingLibrary.value=true;libraryError.value='';deleteConfirmation.clear();await nextTick();libraryNameInput.value?.focus()}
async function startLibraryEdit(food:CatalogFood){
 if(libraryBusy.value||!food.id)return
 editingLibraryFood.value={...food};newLibraryFood.value={name:food.name,grams:100,kcal:Number(food.kcal),protein:Number(food.protein),carbs:Number(food.carbs),fat:Number(food.fat),fiber:Number(food.fiber||0)}
 managingLibrary.value=true;addingLibrary.value=true;libraryError.value='';deleteConfirmation.clear();await nextTick();libraryNameInput.value?.focus()
}
function cancelLibraryEdit(){if(libraryBusy.value)return;addingLibrary.value=false;editingLibraryFood.value=null;libraryError.value='';catalogDialog.value?.focus()}
function catalogKey(event:KeyboardEvent){
 if(event.key==='Escape'){event.preventDefault();event.stopPropagation();if(addingLibrary.value&&!libraryBusy.value){cancelLibraryEdit()}else closeCatalog()}
 if(event.key==='Tab'){
  const nodes=Array.from(catalogDialog.value?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),[tabindex="0"]')||[]).filter(el=>el.getClientRects().length)
  const first=nodes[0],last=nodes[nodes.length-1];if(!first){event.preventDefault();return}
  if(event.shiftKey&&(document.activeElement===first||document.activeElement===catalogDialog.value)){event.preventDefault();last.focus()}
  else if(!event.shiftKey&&(document.activeElement===last||document.activeElement===catalogDialog.value)){event.preventDefault();first.focus()}
 }
}
async function createLibraryFood(){
 if(libraryBusy.value)return
 const food={...newLibraryFood.value,name:newLibraryFood.value.name.trim()}
 if(!food.name||food.name.length>200||!libraryFields.every(field=>Number.isFinite(food[field.key])&&food[field.key]>=0&&food[field.key]<=field.max)){libraryError.value='请填写名称及有效的每 100g 营养数据：热量 0–1200，其他营养 0–100。';return}
 libraryBusy.value=true;libraryError.value='';await nextTick();catalogDialog.value?.focus()
 try{
  const original=editingLibraryFood.value
  if(original){const saved=await updateLibraryFood(original,food);if(original.source!=='custom'){const index=catalog.value.findIndex(item=>item.id===original.id);if(index>=0)catalog.value.splice(index,1,{...original,...saved})}cancelQuickSearch()}
  else await saveCustomFood(food)
  catalogSearch.value=food.name;selectedCategory.value='全部';newLibraryFood.value={name:'',grams:100,kcal:0,protein:0,carbs:0,fat:0,fiber:0};addingLibrary.value=false;editingLibraryFood.value=null;notify(original?'食物已更新':'已新增到食物库');await nextTick();catalogDialog.value?.focus()
 }
 catch(e){libraryError.value=e instanceof Error?e.message:'保存失败，请重试'}finally{libraryBusy.value=false}
}
async function removeLibraryFood(food:CatalogFood){
 if(libraryBusy.value)return
 libraryBusy.value=true;libraryError.value='';await nextTick();catalogDialog.value?.focus()
 try{await deleteLibraryFood(food);if(food.source!=='custom')catalog.value=catalog.value.filter(item=>item.id!==food.id);cancelQuickSearch();if(!catalogCategories.value.includes(selectedCategory.value))selectedCategory.value='全部';notify('已从食物库删除');await nextTick();catalogDialog.value?.focus()}
 catch(e){libraryError.value=e instanceof Error?e.message:'删除失败，请重试';throw e}finally{libraryBusy.value=false}
}
async function loadCatalog(){
 if(catalogLoading.value||libraryBusy.value)return
 catalogLoading.value=true;libraryError.value=''
 try{const [data,personal]=await Promise.all([api('/nutrition/food/catalog?limit=500&user_id=demo-user'),api('/nutrition/custom-foods?limit=200&user_id=demo-user')]);if(!Array.isArray(data.foods)||!Array.isArray(personal.foods))throw new Error('食物库返回格式无效');catalog.value=data.foods;customFoods.splice(0,customFoods.length,...personal.foods)}
 catch(e){libraryError.value=e instanceof Error?e.message:'食物库加载失败，请重试'}finally{catalogLoading.value=false}
}
async function openCatalog(){catalogPrevious=document.activeElement as HTMLElement;selectedCategory.value='全部';catalogSearch.value='';managingLibrary.value=false;addingLibrary.value=false;libraryError.value='';catalogOpen.value=true;void loadCatalog();await nextTick();catalogDialog.value?.focus()}

</script>

<template>
<Teleport to="body"><div v-if="ui.foodOpen" class="fixed inset-0 z-[100] bg-scrim/75 backdrop-blur-md flex items-center justify-center p-4" @click.self="close" @keydown="key">
<section ref="dialog" tabindex="-1" role="dialog" aria-modal="true" aria-labelledby="food-title" class="food-entry-dialog w-full max-w-[760px] rounded-2xl bg-surface-container-lowest border border-outline-variant shadow-2xl p-4 sm:p-6 text-on-surface flex flex-col gap-4 overflow-hidden outline-none">
 <header class="shrink-0 flex items-center justify-between border-b border-outline-variant pb-3.5">
  <div class="flex items-center gap-3"><div class="w-10 h-10 rounded-xl bg-surface-container border border-primary/20 flex items-center justify-center text-primary"><span class="material-symbols-outlined">restaurant</span></div><div><div class="flex flex-wrap gap-2 items-center"><h3 id="food-title" class="text-lg font-semibold">添加饮食摄入</h3></div></div></div>
  <button aria-label="关闭弹窗" @click="close" class="text-on-surface-variant hover:text-on-surface p-1"><span class="material-symbols-outlined">close</span></button>
 </header>

 <div class="food-entry-body flex flex-col gap-4">
 <div class="meal-details-grid grid grid-cols-1 sm:grid-cols-2 gap-3 items-start">
  <div class="min-w-0 space-y-1.5"><div id="meal-type-label" class="text-xs leading-5 text-on-surface-variant">餐次划分</div><div role="group" aria-labelledby="meal-type-label" class="grid grid-cols-4 gap-1.5"><button v-for="(meal,i) in ['早餐','午餐','晚餐','加餐']" :key="meal" @click="ui.meal=meal" :aria-pressed="ui.meal===meal" :class="ui.meal===meal?'border-primary/50 bg-surface-container text-primary':'border-outline-variant bg-surface-container-low text-on-surface-variant'" class="h-10 rounded-lg border text-xs">{{['🍳 早','🥩 午','🥗 晚','💊 加'][i]}}</button></div></div>
  <div class="min-w-0 space-y-1.5"><label for="meal-time" class="block text-xs leading-5 text-on-surface-variant">本餐用餐时间</label><div class="relative"><span aria-hidden="true" class="material-symbols-outlined pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-base text-primary">schedule</span><input id="meal-time" v-model="ui.mealTime" type="datetime-local" step="60" :disabled="ui.savingFood" class="w-full min-w-0 h-10 rounded-lg border border-outline-variant bg-surface-container-low pl-9 pr-2 text-xs text-on-surface outline-none focus:border-primary/60 disabled:cursor-not-allowed disabled:text-on-surface-variant"></div></div>
 </div>

 <div class="text-xs text-on-surface-variant">☷ 已选食物清单 ({{ui.foods.length}} 项)</div>
 <div class="flex flex-col gap-2">
  <article v-for="(food,index) in ui.foods" :key="index" class="fz-food-item p-3 rounded-xl bg-surface-container-lowest border border-outline-variant flex justify-between gap-3">
   <div class="min-w-0"><div class="flex items-center gap-2"><span class="text-sm font-semibold">{{food.name}}</span><small class="rounded bg-surface-container text-primary px-1.5 py-0.5 text-[9px]">{{sourceLabel(food.source)}}</small></div><p class="text-[11px] text-on-surface-variant mt-1">{{(food.kcal*food.grams/100).toFixed(0)}} kcal · 蛋白质 {{(food.protein*food.grams/100).toFixed(1)}}g · 碳水 {{(food.carbs*food.grams/100).toFixed(1)}}g · 脂肪 {{(food.fat*food.grams/100).toFixed(1)}}g</p></div>
   <div class="flex items-center gap-2 shrink-0"><div class="flex bg-surface-container-lowest border border-outline-variant rounded-lg px-2 py-1"><input v-model.number="food.grams" :aria-label="food.name+'克数'" type="number" min="1" max="100000" class="w-14 bg-transparent text-sm text-right outline-none"><span class="text-xs text-on-surface-variant ml-1">g</span></div><DeleteButton :action="()=>removeFood(index)" :subject="food.name" :disabled="ui.savingFood" compact /></div>
  </article>
  <p v-if="!ui.foods.length" class="text-sm text-on-surface-variant text-center py-4 border border-dashed border-outline-variant rounded-xl">暂无已选食物。</p>
 </div>

 <nav aria-label="食物录入方式" class="food-source-tabs grid grid-cols-3 rounded-t-xl border border-outline-variant bg-background">
   <button v-for="tab in sourceTabs" :key="tab.id" type="button" :aria-pressed="sourceTab===tab.id" @click="sourceTab=tab.id" :class="sourceTab===tab.id?'bg-surface-container text-primary border-b-2 border-primary':'text-on-surface-variant hover:text-on-surface'" class="py-2.5 text-xs flex items-center justify-center gap-1.5"><span aria-hidden="true" class="material-symbols-outlined text-base">{{tab.icon}}</span>{{tab.label}}</button>
 </nav>
 <div class="-mt-4 rounded-b-xl border border-t-0 border-outline-variant bg-background">
  <div v-if="sourceTab==='quick'" class="p-3 space-y-3">
   <form class="relative" @submit.prevent="searchQuickFoods"><span class="material-symbols-outlined absolute left-3 top-2 text-on-surface-variant text-lg">search</span><input v-model="ui.foodSearch" aria-label="搜索常用食物" placeholder="搜索食物，选择后加入..." @input="queueQuickSearch" @compositionend="queueQuickSearch" @keydown.enter="searchOnEnter" class="w-full bg-surface-container-lowest border border-outline-variant rounded-xl pl-9 pr-24 py-2 text-sm outline-none focus:border-primary"><button type="submit" class="absolute right-1.5 top-1.5 px-2.5 py-1 rounded-lg bg-surface-variant text-xs text-primary">搜索</button></form>
   <div class="flex items-center justify-between gap-3"><div><h4 class="text-xs font-semibold text-on-surface-variant">{{quickSearchTerm?'相关食物':'常用食物推荐'}}</h4></div><button @click="openCatalog" class="fz-button food-library-button border-primary/40 text-primary shrink-0"><span aria-hidden="true" class="material-symbols-outlined">inventory_2</span><span>食物库</span></button></div>
   <div v-if="!quickSearchTerm" class="flex flex-wrap gap-2">
    <button v-for="food in foodsCatalog" :key="food.name" @click="addFood(food.name)" class="px-3 py-1.5 rounded-lg bg-surface-container-low border border-outline-variant text-xs text-on-surface-variant hover:border-primary/50 hover:text-primary transition-colors"><span class="text-primary">+</span> {{food.name}} <small class="text-on-surface-variant">{{food.kcal}} kcal</small></button>
    <p v-if="!foodsCatalog.length" class="w-full text-xs text-on-surface-variant text-center py-3">暂无常用食物。</p>
   </div>
   <section v-if="quickSearchTerm" aria-label="相关食物" class="space-y-2">
    <p v-if="quickSearchLoading" role="status" class="text-xs text-on-surface-variant">正在搜索相关食物…</p>
    <div v-else-if="quickSearchError" role="alert" class="text-xs text-error">{{quickSearchError}} <button type="button" @click="searchQuickFoods" class="underline">重试</button></div>
    <p v-else-if="!quickSearchResults.length" role="status" class="text-xs text-on-surface-variant">没有找到相关食物。</p>
    <div v-else class="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-[240px] overflow-y-auto">
     <button v-for="food in quickSearchResults" :key="(food.source||'local')+'-'+(food.id??food.name+'-'+food.preparation_state)" type="button" data-food-suggestion @click="selectFoodSuggestion(food)" :aria-label="'添加'+food.name+'（'+(food.source==='custom'?'自定义':'内置')+'）'" class="fz-food-item rounded-xl p-3 text-left flex items-center justify-between gap-2">
      <span class="min-w-0"><strong class="text-xs break-words">{{food.name}}</strong><small class="block text-[10px] text-on-surface-variant mt-1">{{food.category}} · {{food.preparation_state||'内置食物'}} · {{food.kcal}} kcal/100g</small></span><span class="shrink-0 text-xs text-primary">+ 加入</span>
     </button>
    </div>
   </section>
  </div>

  <div v-else-if="sourceTab==='custom'" class="p-3 space-y-3">
   
   <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
    <label class="col-span-2 text-[10px] text-on-surface-variant">食物名称<input v-model="custom.name" maxlength="200" class="fz-input w-full mt-1" placeholder="例如：自制鸡肉卷"></label>
    <label class="text-[10px] text-on-surface-variant">实际克重<input v-model.number="custom.grams" type="number" min="1" class="fz-input w-full mt-1"></label>
    <label class="text-[10px] text-on-surface-variant">热量 kcal/100g<input v-model.number="custom.kcal" type="number" min="0" class="fz-input w-full mt-1"></label>
    <label v-for="field in nutrientFields" :key="field.key" class="text-[10px] text-on-surface-variant">{{field.label}}/100g<input v-model.number="custom[field.key]" type="number" min="0" step="0.1" class="fz-input w-full mt-1"></label>
   </div>
   <div class="grid grid-cols-2 gap-2">
    <button type="button" @click="addCustom" class="fz-button fz-primary flex-1">+ 加入本餐清单</button>
    <button type="button" @click="saveCustomToLibrary(custom)" class="fz-button border-primary/40 text-primary flex-1">保存到食物库</button>
   </div>
   <div class="flex items-center justify-between gap-2 pt-1"><div><h4 class="text-xs font-semibold text-on-surface-variant">我的自定义食物</h4></div><button type="button" @click="openCatalog" class="fz-button food-library-button border-primary/40 text-primary shrink-0"><span aria-hidden="true" class="material-symbols-outlined">inventory_2</span><span>食物库</span></button></div>
   <div class="flex flex-wrap gap-2">
    <button v-for="food in customFoods.slice(0,8)" :key="food.id" type="button" @click="addCustomLibraryFood(food)" class="px-3 py-1.5 rounded-lg bg-surface-container-low border border-outline-variant text-xs text-on-surface-variant hover:border-primary/50 hover:text-primary transition-colors"><span class="text-primary">+</span> {{food.name}} <small class="text-on-surface-variant">{{food.kcal}} kcal</small></button>
    <p v-if="!customFoods.length" class="w-full text-xs text-on-surface-variant text-center py-2">暂无自定义食物。</p>
   </div>
  </div>

  <div v-else class="p-3 space-y-3">
   <div><label class="text-[11px] text-on-surface-variant">描述食物、品牌和烹饪方式</label><textarea v-model="aiQuery" rows="2" maxlength="500" class="fz-input w-full mt-1 resize-none" placeholder="例如：某品牌低糖蛋白棒，包装 60g；或清炒牛肉，少油熟制"></textarea></div>
   <button type="button" @click="askAi" :disabled="aiLoading" class="fz-button w-full border-primary/40 text-primary">{{aiLoading?'AI 查询中…':'询问 AI 营养成分'}}</button>
   
   <p v-if="ui.foodError" role="alert" class="text-error text-xs">{{ui.foodError}}</p>
   <div v-if="aiResult" ref="aiResultPanel" data-ai-result class="rounded-xl border border-primary/30 bg-surface-container p-3 space-y-3">
    <div class="flex items-start justify-between gap-3"><div><strong class="text-sm text-primary">AI 估算结果</strong><p class="text-[10px] text-on-surface-variant mt-1">{{aiNotes}}</p></div><span class="text-[10px] rounded bg-surface-container text-primary px-2 py-1">置信度 {{aiConfidence}}</span></div>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
     <label class="col-span-2 text-[10px] text-on-surface-variant">食物名称<input v-model="aiResult.name" class="fz-input w-full mt-1"></label>
     <label class="text-[10px] text-on-surface-variant">实际克重<input v-model.number="aiResult.grams" type="number" min="1" class="fz-input w-full mt-1"></label>
     <label class="text-[10px] text-on-surface-variant">热量 kcal/100g<input v-model.number="aiResult.kcal" type="number" min="0" class="fz-input w-full mt-1"></label>
     <label v-for="field in nutrientFields" :key="field.key" class="text-[10px] text-on-surface-variant">{{field.label}}/100g<input v-model.number="aiResult[field.key]" type="number" min="0" step="0.1" class="fz-input w-full mt-1"></label>
    </div>
    
    <div class="grid grid-cols-2 gap-2">
     <button type="button" @click="addAi" class="fz-button fz-primary flex-1">确认并加入本餐清单</button>
     <button type="button" @click="saveAiToCustom" class="fz-button border-primary/40 text-primary flex-1">{{aiSaved?'已保存到食物库':'保存到食物库'}}</button>
    </div>
   </div>
  </div>
 </div>


 <Teleport to="body"><div v-if="catalogOpen" class="fixed inset-0 z-[160] bg-scrim/80 flex items-center justify-center p-4" @click.self="closeCatalog">
  <section ref="catalogDialog" tabindex="-1" role="dialog" aria-modal="true" :aria-busy="libraryBusy" aria-labelledby="catalog-title" @keydown="catalogKey" class="catalog-dialog w-full max-w-[760px] rounded-2xl border border-outline-variant bg-surface-container-lowest shadow-2xl flex flex-col overflow-hidden">
   <header class="shrink-0 flex items-center justify-between gap-3 px-5 py-4 border-b border-outline-variant"><div><h3 id="catalog-title" class="text-base font-semibold">食物库</h3></div><button type="button" @click="closeCatalog" :disabled="libraryBusy" aria-label="关闭食物库" class="fz-button"><span aria-hidden="true" class="material-symbols-outlined">close</span></button></header>
   <div class="shrink-0 p-4 border-b border-outline-variant space-y-3">
    <div class="flex items-center justify-between gap-2"><button type="button" @click="toggleLibraryManagement" :disabled="libraryBusy||catalogLoading" :aria-pressed="managingLibrary" class="fz-button">{{managingLibrary?'完成管理':'管理食物'}}</button><button type="button" @click="startLibraryAdd" :disabled="libraryBusy||catalogLoading" class="fz-button fz-primary inline-flex items-center justify-center gap-1"><span aria-hidden="true" class="material-symbols-outlined">add</span>新增食物</button></div>
    <input v-model="catalogSearch" aria-label="搜索食物库" placeholder="搜索食物名称、分类或状态" class="w-full bg-surface-container-lowest border border-outline-variant rounded-xl px-3 py-2 text-sm">
    <div class="flex gap-2 overflow-x-auto pb-1"><button v-for="category in catalogCategories" :key="category" type="button" @click="selectedCategory=category" :aria-pressed="selectedCategory===category" :class="selectedCategory===category?'bg-primary text-on-primary border-primary':'bg-surface-container-low text-on-surface-variant border-outline-variant'" class="shrink-0 px-3 py-1.5 rounded-full border text-xs">{{category}}</button></div>
   </div>
   <div class="catalog-results p-4 overflow-y-auto">
    <p v-if="libraryError&&!addingLibrary" role="alert" class="text-error text-xs mb-3">{{libraryError}} <button type="button" @click="loadCatalog" :disabled="libraryBusy||catalogLoading" class="underline">重新加载</button></p>
    <form v-if="addingLibrary" :aria-label="editingLibraryFood?'编辑食物':'新增食物'" @submit.prevent.stop="createLibraryFood" class="space-y-3 mb-4 pb-4 border-b border-outline-variant">
     <h4 class="font-semibold text-sm">{{editingLibraryFood?'编辑食物':'新增食物'}} · 每 100g 营养</h4>
     <label class="block text-xs">食物名称<input ref="libraryNameInput" v-model="newLibraryFood.name" required maxlength="200" :disabled="libraryBusy" class="mt-1 w-full bg-surface-container-low border border-outline-variant rounded-xl px-3 py-2" placeholder="例如：自制鸡胸肉沙拉"></label>
     <div class="grid grid-cols-2 sm:grid-cols-3 gap-3"><label v-for="field in libraryFields" :key="field.key" class="text-xs">{{field.label}}<input v-model.number="newLibraryFood[field.key]" type="number" required min="0" :max="field.max" step="any" :disabled="libraryBusy" class="mt-1 w-full min-w-0 bg-surface-container-low border border-outline-variant rounded-xl px-3 py-2"></label></div>
     <p v-if="libraryError" role="alert" class="text-error text-xs">{{libraryError}}</p>
     <div class="flex flex-wrap justify-end gap-2"><button type="button" @click="cancelLibraryEdit" :disabled="libraryBusy" class="fz-button">{{editingLibraryFood?'取消编辑':'取消新增'}}</button><button type="submit" :disabled="libraryBusy" class="fz-button fz-primary">{{libraryBusy?'保存中…':editingLibraryFood?'保存修改':'保存到食物库'}}</button></div>
    </form>
    <p v-if="catalogLoading" role="status" class="text-xs text-on-surface-variant text-center py-8">正在读取食物库…</p>
    <div v-else class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
     <article v-for="food in filteredCatalog" :key="(food.source||'local')+'-'+(food.id??food.name+'-'+food.preparation_state)" class="fz-food-item p-3 rounded-xl bg-surface-container-lowest border border-outline-variant">
      <button type="button" @click="addCatalogFood(food)" :disabled="managingLibrary||libraryBusy" :aria-label="'添加'+food.name+'到本餐'" class="catalog-food-select w-full text-left rounded-lg focus-visible:outline-primary">
       <div class="flex justify-between gap-2"><strong class="text-xs break-words">{{food.name}}</strong><span class="text-[10px] text-primary shrink-0">{{food.kcal}} kcal</span></div><p class="text-[10px] text-on-surface-variant mt-1">{{food.category}} · {{food.preparation_state}}</p><p class="text-[10px] text-on-surface-variant mt-1">蛋白 {{food.protein}}g · 碳水 {{food.carbs}}g · 脂肪 {{food.fat}}g / 100g</p>
      </button>
      <div v-if="managingLibrary" class="mt-3 flex items-center justify-between gap-2"><span class="text-[10px] text-on-surface-variant">{{food.source==='custom'?'自定义食物':'内置食物'}}</span><div class="flex items-center gap-2"><button type="button" @click="startLibraryEdit(food)" :disabled="libraryBusy||!food.id" :aria-label="'编辑'+food.name+'（食物库）'" class="fz-button">编辑</button><DeleteButton :action="()=>removeLibraryFood(food)" :subject="food.name+'（食物库）'" :disabled="libraryBusy||!food.id" /></div></div>
      <p v-if="managingLibrary&&!food.id" class="text-xs text-on-surface-variant mt-2">此条目暂不可管理，请重新加载。</p>
     </article>
     <p v-if="!filteredCatalog.length" class="sm:col-span-2 lg:col-span-3 text-center py-8 text-xs text-on-surface-variant">没有匹配食物。</p>
    </div>
   </div>
   <footer class="shrink-0 px-5 py-3 border-t border-outline-variant flex items-center justify-between gap-2"><span role="status" class="text-xs text-on-surface-variant">{{libraryBusy?'正在更新食物库…':managingLibrary?libraryFoods.length+' 项食物':'已选 '+ui.foods.length+' 项食物'}}</span><button type="button" @click="managingLibrary?toggleLibraryManagement():closeCatalog()" :disabled="libraryBusy" class="fz-button fz-primary">{{managingLibrary?'完成管理':'完成选择'}}</button></footer>
  </section>
 </div></Teleport>


 <div class="rounded-xl bg-surface-container-lowest border border-outline-variant p-3.5"><div class="text-xs pb-2 mb-2 border-b border-outline-variant">ϟ 本次摄入合计 (共 {{ui.foods.length}} 项食物)</div><div class="grid grid-cols-4 text-center gap-2"><div v-for="(value,key) in totals" :key="key"><span class="text-[11px] text-on-surface-variant">{{({kcal:'总热量',protein:'蛋白质',carbs:'碳水',fat:'优质脂肪'})[key]}}</span><p class="font-bold mt-0.5" :class="key==='protein'?'text-primary':key==='fat'?'text-error':''">{{Number.isFinite(value)?value.toFixed(1):'—'}} <small class="text-[10px] text-on-surface-variant font-normal">{{key==='kcal'?'kcal':'g'}}</small></p></div></div><div class="h-1.5 rounded-full bg-surface-container-lowest mt-3 flex overflow-hidden"><span class="bg-primary" :style="{width:totals.protein*4/(totals.kcal||1)*100+'%'}"></span><span class="bg-secondary-fixed" :style="{width:totals.carbs*4/(totals.kcal||1)*100+'%'}"></span><span class="bg-error" :style="{width:totals.fat*9/(totals.kcal||1)*100+'%'}"></span></div></div>
 <p v-if="ui.foodError&&sourceTab!=='ai'" role="alert" class="text-error text-xs">{{ui.foodError}}</p>
 </div>
 <footer class="shrink-0 flex flex-wrap justify-between pt-3 border-t border-outline-variant gap-2"><DeleteButton :action="clearFoods" subject="当前餐次全部草稿食物" label="清空全部" :disabled="ui.savingFood||!ui.foods.length" compact /><div class="flex flex-wrap flex-1 justify-end gap-2"><button @click="close" class="fz-button">取消</button><button @click="saveFood" :disabled="ui.savingFood||!ui.foods.length" class="fz-button fz-primary">{{ui.savingFood?'保存中…':'✓ 保存本次摄入 ('+ui.foods.length+' 项食物)'}}</button></div></footer>
</section></div></Teleport>
</template>
<style scoped>
.catalog-food-select:disabled{opacity:1;cursor:default}

.food-library-button{display:inline-flex;align-items:center;justify-content:center;gap:6px;white-space:nowrap}
.food-library-button .material-symbols-outlined{display:block;flex-shrink:0;font-size:16px;line-height:1}

.food-entry-dialog{height:min(860px,92vh);max-height:calc(100vh - 2rem);height:min(860px,92dvh);max-height:calc(100dvh - 2rem);min-height:0}
.food-entry-body{flex:1;min-height:0;overflow-y:auto;overscroll-behavior-y:contain;scrollbar-gutter:stable;scroll-padding-block:56px 12px}
.food-entry-body>*{flex-shrink:0}
.food-source-tabs{position:sticky;top:0;z-index:1}
.catalog-dialog{height:min(760px,90dvh);max-height:calc(100dvh - 2rem);min-height:0}
.catalog-results{flex:1;min-height:0}
</style>
