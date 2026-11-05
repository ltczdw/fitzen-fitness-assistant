<script setup lang="ts">
import {computed,nextTick,onBeforeUnmount,onMounted,ref} from 'vue'
import DeleteButton from './DeleteButton.vue'
import {deleteConfirmation} from './delete-confirmation'
import {api,findFoodSuggestions,customFoods,ensureCustomFoods,ensureFoodCatalog,foodsCatalog,notify,openRecords,refresh,saveCustomFood,deleteLibraryFood,updateLibraryFood,ui,type RecordItem} from '../../fitzen/state'
type Nutrients={calories:number;protein_g:number;carbs_g:number;fat_g:number}
type Food={key:string;id?:number;name:string;grams:number;originalName:string;originalGrams:number;original:Nutrients|null;basis:Nutrients|null;pending:boolean;error:string;version:number;source?:'local'|'custom'|'ai';fiber?:number;rawName?:string;requestedName?:string;nameDirty?:boolean}
type SourceTab='quick'|'custom'|'ai'
type CatalogFood={name:string;category:string;preparation_state:string;kcal:number;protein:number;carbs:number;fat:number;fiber?:number;id?:number;source?:'local'|'custom'}
type EditableFood={name:string;grams:number;kcal:number;protein:number;carbs:number;fat:number;fiber:number}
const record=ref<RecordItem|null>(null),meal=ref('午餐'),date=ref(''),foods=ref<Food[]>([])
const manual=ref({description:'',calories:0,protein_g:0,carbs_g:0,fat_g:0})
const busy=ref(false),error=ref('')
const sourceTab=ref<SourceTab>('quick'),catalogOpen=ref(false),catalogLoading=ref(false),catalogSearch=ref(''),selectedCategory=ref('全部'),localSearch=ref('')
const sourceTabs=[{id:'quick' as const,label:'常用推荐',icon:'recommend'},{id:'custom' as const,label:'自定义',icon:'edit_note'},{id:'ai' as const,label:'问 AI',icon:'auto_awesome'}]
const nutrientFields=[{key:'protein' as const,label:'蛋白质'},{key:'carbs' as const,label:'碳水'},{key:'fat' as const,label:'脂肪'},{key:'fiber' as const,label:'纤维'}]
const catalog=ref<CatalogFood[]>([])
const custom=ref<EditableFood>({name:'',grams:100,kcal:0,protein:0,carbs:0,fat:0,fiber:0})
const aiQuery=ref(''),aiLoading=ref(false),aiResult=ref<EditableFood|null>(null),aiNotes=ref(''),aiConfidence=ref(''),aiSaved=ref(false),aiResultPanel=ref<HTMLElement|null>(null)
const panel=ref<HTMLElement|null>(null)
let previous:HTMLElement|null=null,overflow=''
const fields=['calories','protein_g','carbs_g','fat_g'] as const
const labels={calories:'热量',protein_g:'蛋白质',carbs_g:'碳水',fat_g:'脂肪'}
const structured=computed(()=>record.value?.source==='ai_food_log')
const mealOptions=computed(()=>['早餐','午餐','晚餐','加餐'].includes(meal.value)?['早餐','午餐','晚餐','加餐']:['早餐','午餐','晚餐',meal.value])
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
 cancelQuickSearch();const term=localSearch.value.trim();quickSearchTerm.value=term
 if(term){quickSearchLoading.value=true;quickSearchTimer=setTimeout(()=>{quickSearchTimer=null;void searchQuickFoods()},250)}
}
async function searchQuickFoods(){
 if(quickSearchTimer){clearTimeout(quickSearchTimer);quickSearchTimer=null}
 const term=localSearch.value.trim();if(!term){cancelQuickSearch();return}
 quickSearchController?.abort();const controller=new AbortController();quickSearchController=controller;const version=++quickSearchVersion
 quickSearchTerm.value=term;quickSearchResults.value=[];quickSearchLoading.value=true;quickSearchError.value=''
 try{const foods=await findFoodSuggestions(term,controller.signal);if(version===quickSearchVersion&&term===localSearch.value.trim())quickSearchResults.value=foods}
 catch(e){if(version===quickSearchVersion&&!controller.signal.aborted)quickSearchError.value=e instanceof Error&&e.name==='TimeoutError'?'搜索超时，请重试。':e instanceof Error?e.message:'食物搜索失败，请重试。'}
 finally{if(version===quickSearchVersion)quickSearchLoading.value=false}
}
function searchOnEnter(event:KeyboardEvent){if(event.isComposing||event.keyCode===229)return;event.preventDefault();void searchQuickFoods()}
function selectFoodSuggestion(food:CatalogFood){if(addCatalogFood(food)){localSearch.value='';cancelQuickSearch()}}

const round=(n:number)=>Math.round(n*10)/10
const num=(n:unknown)=>n==null||!Number.isFinite(Number(n))?'—':String(round(Number(n)))
const signed=(n:number)=>(n>0?'+':'')+num(n)
function nutrients(source:any):Nutrients{return {calories:Number(source?.calories||0),protein_g:Number(source?.protein_g||0),carbs_g:Number(source?.carbs_g||0),fat_g:Number(source?.fat_g||0)}}
function portion(food:Food):Nutrients|null{
 if(!food.basis||!Number.isFinite(food.grams)||food.grams<=0||food.grams>100000)return null
 return Object.fromEntries(fields.map(key=>[key,round(food.basis![key]*food.grams/100)])) as Nutrients
}
const original=computed(()=>nutrients(record.value))
const ready=computed(()=>!structured.value||foods.value.length>0&&foods.value.every(food=>food.name.trim()&&portion(food)&&!food.pending&&!food.error))
const total=computed<Nutrients>(()=>structured.value?foods.value.reduce((sum,food)=>{const p=portion(food);fields.forEach(key=>sum[key]+=p?.[key]||0);return sum},{calories:0,protein_g:0,carbs_g:0,fat_g:0}):nutrients(manual.value))
const delta=computed(()=>round(total.value.calories-original.value.calories))
const proteinProgress=computed(()=>{
 if(!record.value||!ui.targets.protein_target_g||localDate(record.value.date||'').slice(0,10)!==ui.data?.end_date)return null
 return Math.max(0,Math.round((Number(ui.summary.protein_g||0)-original.value.protein_g+total.value.protein_g)/ui.targets.protein_target_g*100))
})
const targetHint=computed(()=>ui.targets.calories_target?'每日目标 '+num(ui.targets.calories_target)+' kcal':'按实际摄入调整分量')
function localDate(value:string){
 if(!value)return ''
 if(!value.includes('T'))return value.slice(0,10)
 const d=new Date(value);if(!Number.isFinite(d.getTime()))return ''
 return [d.getFullYear(),String(d.getMonth()+1).padStart(2,'0'),String(d.getDate()).padStart(2,'0')].join('-')+'T'+String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0')
}
function clearConfirm(){deleteConfirmation.clear()}
function close(){
 if(busy.value)return
 cancelQuickSearch();record.value=null;catalogOpen.value=false;clearConfirm();document.body.style.overflow=overflow;previous?.focus()
}
async function open(event:Event){
 if(busy.value)return
 if(record.value)close()
 const data=(event as CustomEvent<RecordItem>).detail
 if(!data)return
 previous=document.activeElement as HTMLElement;overflow=document.body.style.overflow;document.body.style.overflow='hidden'
 record.value=JSON.parse(JSON.stringify(data));meal.value=data.meal_type||'其他';date.value=localDate(data.date||'')
 manual.value={description:data.description||'',...nutrients(data)}
 foods.value=(data.items||[]).map(item=>{
  const n=nutrients(item),grams=Number(item.weight_g)
  return {key:crypto.randomUUID(),id:item.id,name:item.food_name,grams,originalName:item.food_name,rawName:String(item.raw_food_name||item.food_name||''),originalGrams:grams,nameDirty:false,original:n,basis:grams>0?Object.fromEntries(fields.map(key=>[key,n[key]*100/grams])) as Nutrients:null,pending:false,error:'',version:0,source:item.source==='custom'?'custom':item.source==='ai'?'ai':'local',fiber:Number(item.fiber_g||0)}
 })
 cancelQuickSearch();error.value='';sourceTab.value='quick';catalogOpen.value=false;localSearch.value='';clearConfirm();void ensureFoodCatalog();void ensureCustomFoods();await nextTick();panel.value?.focus()
}
function add(){if(foods.value.length>=30)return;clearConfirm();foods.value.push({key:crypto.randomUUID(),name:'',grams:100,originalName:'',originalGrams:0,original:null,basis:null,pending:false,error:'',version:0,nameDirty:false})}
function appendFood(food:EditableFood,source:'local'|'custom'|'ai',requestedName?:string){
 const intent=(requestedName??food.name).trim()
 if(foods.value.length>=30){error.value='每餐最多添加 30 项食物。';return false}
 if(!food.name.trim()||![food.grams,food.kcal,food.protein,food.carbs,food.fat,food.fiber].every(Number.isFinite)||food.grams<=0||food.kcal<0||food.protein<0||food.carbs<0||food.fat<0||food.fiber<0){error.value='请填写食物名称、有效克重和非负的每 100g 营养数据。';return false}
 foods.value.push({key:crypto.randomUUID(),name:food.name.trim(),grams:food.grams,originalName:'',originalGrams:0,original:null,basis:{calories:food.kcal,protein_g:food.protein,carbs_g:food.carbs,fat_g:food.fat},pending:false,error:'',version:0,source,fiber:food.fiber,rawName:intent,requestedName:intent,nameDirty:false});error.value='';return true
}
async function addLocalByName(name=localSearch.value){
 const query=name.trim();if(!query)return
 try{const data=await api('/nutrition/food/lookup?name='+encodeURIComponent(query)),p=data.per_100g;if(appendFood({name:data.food_name,grams:100,kcal:Number(p.kcal),protein:Number(p.protein_g),carbs:Number(p.carb_g),fat:Number(p.fat_g),fiber:Number(p.fiber_g||0)},'local',query))localSearch.value=''}catch(e){error.value=e instanceof Error?e.message:'食物匹配失败'}
}
function addCatalogFood(food:CatalogFood){return appendFood({name:food.name,grams:100,kcal:Number(food.kcal),protein:Number(food.protein),carbs:Number(food.carbs),fat:Number(food.fat),fiber:Number(food.fiber||0)},food.source==='custom'?'custom':'local')}
function customValues(food:EditableFood|{name:string;grams:number;kcal:number;protein:number;carbs:number;fat:number;fiber?:number}){return {name:food.name,grams:Number(food.grams||100),kcal:Number(food.kcal||0),protein:Number(food.protein||0),carbs:Number(food.carbs||0),fat:Number(food.fat||0),fiber:Number(food.fiber||0)}}
async function saveCustomToLibrary(food:EditableFood){
 if(!food.name.trim()||![food.grams,food.kcal,food.protein,food.carbs,food.fat,food.fiber].every(Number.isFinite)||food.grams<=0||food.kcal<0||food.protein<0||food.carbs<0||food.fat<0||food.fiber<0){error.value='请先填写完整的自定义食物营养数据。';return false}
 try{await saveCustomFood(customValues(food));error.value='';notify('已保存到食物库');return true}catch(e){error.value=e instanceof Error?e.message:'保存自定义食物失败';return false}
}
function addCustom(){if(appendFood(custom.value,'custom')){void saveCustomToLibrary(custom.value);custom.value={name:'',grams:100,kcal:0,protein:0,carbs:0,fat:0,fiber:0}}}
async function askAi(){
 const description=aiQuery.value.trim();if(description.length<2){error.value='请描述食物名称、品牌或烹饪方式。';return}
 aiLoading.value=true;error.value='';aiResult.value=null;aiSaved.value=false
 try{
  const data=await api('/nutrition/food/estimate',{method:'POST',signal:AbortSignal.timeout(45000),body:JSON.stringify({description})}),p=data?.per_100g
  if(!p)throw new Error('AI 未返回可用营养数据，请稍后重试。')
  aiResult.value={name:data.food_name,grams:100,kcal:Number(p.kcal),protein:Number(p.protein_g),carbs:Number(p.carbs_g),fat:Number(p.fat_g),fiber:Number(p.fiber_g||0)}
  aiNotes.value=data.notes||'';aiConfidence.value=data.confidence||'medium';await nextTick();aiResultPanel.value?.scrollIntoView({behavior:'smooth',block:'nearest'})
 }catch(e){error.value=e instanceof Error&&e.name==='TimeoutError'?'AI 查询超时，请稍后重试。':e instanceof Error?e.message:'AI 营养查询失败'}finally{aiLoading.value=false}
}async function saveAiToCustom(){if(aiResult.value&&await saveCustomToLibrary(aiResult.value))aiSaved.value=true}
function addAi(){if(aiResult.value&&appendFood(aiResult.value,'ai')){aiResult.value=null;aiQuery.value='';aiNotes.value='';aiConfidence.value='';aiSaved.value=false}}
function remove(food:Food){if(foods.value.length<=1)return;foods.value=foods.value.filter(f=>f.key!==food.key)}
function changeName(food:Food){food.version++;food.basis=null;food.error='';food.pending=false;food.nameDirty=true;food.requestedName=food.name.trim()}
async function lookup(food:Food){
 const name=food.name.trim(),version=++food.version;if(!name){food.error='请填写食物名称';return}
 const customMatch=customFoods.find(item=>item.name.trim().toLowerCase()===name.toLowerCase())
 if(customMatch){
  if(version!==food.version)return
  food.source='custom';food.fiber=customMatch.fiber;food.basis={calories:customMatch.kcal,protein_g:customMatch.protein,carbs_g:customMatch.carbs,fat_g:customMatch.fat};food.pending=false;food.error='';return
 }
 food.pending=true;food.error='';try{
  const data=await api('/nutrition/food/lookup?name='+encodeURIComponent(name))
  if(version!==food.version)return
  const p=data.per_100g
  food.source='local';food.fiber=Number(p.fiber_g||0);food.basis={calories:p.kcal,protein_g:p.protein_g,carbs_g:p.carb_g,fat_g:p.fat_g}
 }catch(e){if(version===food.version){food.basis=null;food.error=e instanceof Error?e.message:'食物匹配失败'}}
 finally{if(version===food.version)food.pending=false}
}
function changed(food:Food){return !food.original||food.name!==food.originalName||Number(food.grams)!==food.originalGrams}
function changeLabel(food:Food){return !food.original?'新增食物':food.name!==food.originalName?'已替换':'已修改 ('+signed(food.grams-food.originalGrams)+'g)'}
function tag(food:Food){const p=portion(food);if(!p)return '待匹配';const energies=[p.protein_g*4,p.carbs_g*4,p.fat_g*9];return ['蛋白质为主','碳水为主','脂肪为主'][energies.indexOf(Math.max(...energies))]}
async function save(){
 if(!record.value||busy.value)return
 if(!ready.value){error.value='请完成食物匹配，并填写有效分量。';return}
 if(!date.value){error.value='请选择有效的本餐用餐时间。';return}
 busy.value=true;error.value=''
 try{
 const r=record.value
 const body=structured.value?{meal_type:meal.value,items:foods.value.map(f=>{
  const keepId=Boolean(f.id)
   const sendName=!f.id||f.nameDirty||f.name.trim()!==f.originalName
   const submitName=(f.nameDirty?f.name:(f.requestedName??f.name)).trim()
  return {...(keepId?{id:f.id}:{}),...(sendName?{food_name:submitName}:{}),weight_g:Number(f.grams),...(f.source?{source:f.source}:{}),...((f.source==='custom'||f.source==='ai')&&f.basis?{nutrition_per_100g:{kcal:f.basis.calories,protein_g:f.basis.protein_g,carbs_g:f.basis.carbs_g,fat_g:f.basis.fat_g,fiber_g:f.fiber||0}}:{})}
 }),...(date.value!==localDate(r.date||'')?{record_time:new Date(date.value).toISOString()}: {})}:{meal_type:meal.value,...manual.value,...(date.value!==localDate(r.date||'')?{log_date:date.value.slice(0,10)}:{})};await api((structured.value?'/nutrition/food/':'/nutrition/')+encodeURIComponent(r.id)+'?user_id=demo-user',{method:'PUT',body:JSON.stringify(body)})
 busy.value=false;close();notify('饮食记录已保存');await refresh()
 }catch(e){error.value=e instanceof Error?e.message:'保存失败，请重试'}finally{busy.value=false}
}
async function deleteMeal(){
 if(!record.value||busy.value)return
 busy.value=true;error.value=''
 try{await api((structured.value?'/nutrition/food/':'/nutrition/')+encodeURIComponent(record.value.id)+'?user_id=demo-user',{method:'DELETE'});busy.value=false;close();notify('饮食记录已删除');await refresh()}
 catch(e){error.value=e instanceof Error?e.message:'删除失败'}finally{busy.value=false}
}
function key(e:KeyboardEvent){
 if(e.key==='Escape'){e.stopPropagation();if(catalogOpen.value){closeCatalog();return}close()}
 if(e.key==='Tab'){const nodes=panel.value?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled)');if(!nodes?.length)return;const first=nodes[0],last=nodes[nodes.length-1];if(e.shiftKey&&(document.activeElement===first||document.activeElement===panel.value)){e.preventDefault();last.focus()}else if(!e.shiftKey&&(document.activeElement===last||document.activeElement===panel.value)){e.preventDefault();first.focus()}}
}
function openList(){openRecords('nutrition')}
onMounted(()=>{window.addEventListener('fitness-edit-nutrition-record',open);window.addEventListener('fitness-open-nutrition-records',openList)})
onBeforeUnmount(()=>{cancelQuickSearch();window.removeEventListener('fitness-edit-nutrition-record',open);window.removeEventListener('fitness-open-nutrition-records',openList);clearConfirm();if(record.value)document.body.style.overflow=overflow})

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
<Teleport to="body">
<div v-if="record" class="meal-edit-overlay" @click.self="close" @keydown="key">
<form ref="panel" tabindex="-1" class="meal-editor" role="dialog" aria-modal="true" aria-labelledby="meal-edit-title" @submit.prevent="save">
<header class="edit-head">
 <div class="head-line"><span class="edit-emblem material-symbols-outlined">restaurant_menu</span><h2 id="meal-edit-title">编辑本餐食物</h2><span class="edit-status" :title="String(record.id)"><i></i>编辑模式 · 记录编号 #{{String(record.id)}}</span><button type="button" class="close-edit" aria-label="关闭" :disabled="busy" @click="close"><span class="material-symbols-outlined">close</span></button></div>
 
</header>
<div class="edit-basics">
 <div class="section-line"><span>餐次分类设定</span><span class="green"><span class="material-symbols-outlined">flag</span>{{targetHint}}</span></div>
 <div class="meal-tabs"><button v-for="option in mealOptions" :key="option" type="button" :class="{selected:meal===option}" :disabled="busy" @click="meal=option"><span v-if="meal===option" class="material-symbols-outlined">check</span>{{option}}</button></div>
 <div class="section-line time-line"><span><span class="material-symbols-outlined">schedule</span>{{structured?'本餐用餐时间（全部食物共用）':'摄入日期'}}</span><span class="green">{{structured?'新增食物沿用此时间 · 可修改':'可修改'}}</span></div>
 <label class="time-input">{{structured?'用餐时间':'摄入日期'}}<input v-model="date" :type="structured?'datetime-local':'date'" step="60" required :disabled="busy"></label>
</div>
<div class="edit-foods">
 <div class="section-line"><strong>{{structured?'包含食物清单':'饮食营养记录'}} <small v-if="structured">{{foods.length}} 项食物</small></strong></div>
 <template v-if="structured">
 <article v-for="food in foods" :key="food.key" class="edit-food fz-food-item" :class="{modified:changed(food)}">
  <div class="food-top"><div class="food-name-group"><input v-model="food.name" aria-label="食物名称" required maxlength="200" :disabled="busy" placeholder="输入食物名称" @input="changeName(food)" @change="lookup(food)"><small v-if="food.rawName&&food.rawName!==food.name" title="用户原始输入">原输入 {{food.rawName}}</small><small>{{tag(food)}}</small><small v-if="changed(food)" class="change-badge">{{changeLabel(food)}}</small></div><div class="food-controls"><label><input v-model.number="food.grams" aria-label="食物重量" type="number" min="1" max="100000" step="0.1" required :disabled="busy"><span>g</span></label><DeleteButton :action="()=>remove(food)" :subject="food.name||'此项食物'" :disabled="busy||foods.length<=1" compact /></div></div>
  <div class="food-macros"><strong>{{num(portion(food)?.calories)}} <span>kcal</span></strong><span>蛋白 {{num(portion(food)?.protein_g)}}g</span><span>碳水 {{num(portion(food)?.carbs_g)}}g</span><span>脂肪 {{num(portion(food)?.fat_g)}}g</span><em>{{changed(food)&&food.original?'原 '+num(food.originalGrams)+'g ('+num(food.original.calories)+' kcal)':food.original?'未变动':'新增'}}</em></div>
  <p v-if="food.pending" class="lookup-status">正在匹配食物…</p><p v-else-if="food.error" class="edit-error" role="alert">{{food.error}} <button type="button" @click="lookup(food)">重试</button></p>
 </article>

 </template>
 <div v-else class="manual-grid"><label class="description">饮食描述<input v-model="manual.description" required maxlength="1000" :disabled="busy"></label><label v-for="field in fields" :key="field">{{labels[field]}} ({{field==='calories'?'kcal':'g'}})<input v-model.number="manual[field]" type="number" min="0" :max="field==='calories'?10000:field==='carbs_g'?1000:500" step="0.1" required :disabled="busy"></label></div>
</div>
<div v-if="structured" class="edit-source-panel">
 <div class="rounded-xl border border-surface-container-high bg-background overflow-hidden mt-1">
  <div class="grid grid-cols-3 border-b border-surface-container-high">
   <button v-for="tab in sourceTabs" :key="tab.id" type="button" @click="sourceTab=tab.id" :class="sourceTab===tab.id?'bg-surface-container text-primary border-b-2 border-primary':'text-on-surface-variant hover:text-on-surface'" class="py-2 text-[11px] flex items-center justify-center gap-1"><span class="material-symbols-outlined text-[15px]">{{tab.icon}}</span>{{tab.label}}</button>
  </div>
  <div v-if="sourceTab==='quick'" class="p-3 space-y-3">
   <div class="relative"><span class="material-symbols-outlined absolute left-3 top-2 text-on-surface-variant text-[16px]">search</span><input v-model="localSearch" aria-label="搜索常用食物" placeholder="搜索食物，选择后加入..." @input="queueQuickSearch" @compositionend="queueQuickSearch" class="w-full bg-surface-container-low border border-surface-container-high rounded-lg pl-9 pr-20 py-2 text-xs outline-none focus:border-primary" @keydown.enter="searchOnEnter"><button type="button" @click="searchQuickFoods" class="absolute right-1.5 top-1.5 px-2 py-1 rounded bg-surface-container-high text-[10px] text-primary">搜索</button></div>
   <div class="flex items-center justify-between gap-2"><div><strong class="text-xs">{{quickSearchTerm?'相关食物':'常用食物推荐'}}</strong></div><button type="button" @click="openCatalog" class="food-library-button px-3 py-1.5 rounded-lg border border-primary/40 text-primary text-[11px]"><span aria-hidden="true" class="material-symbols-outlined">inventory_2</span><span>食物库</span></button></div>
   <div v-if="!quickSearchTerm" class="flex flex-wrap gap-1.5"><button v-for="food in foodsCatalog" :key="food.name" type="button" @click="addLocalByName(food.name)" class="px-2.5 py-1.5 rounded-lg bg-surface-container-low border border-surface-container-high text-[10px] text-on-surface-variant hover:border-primary/50 hover:text-primary"><span class="text-primary">+</span> {{food.name}}</button><span v-if="!foodsCatalog.length" class="text-[10px] text-on-surface-variant">常用推荐加载中，可打开食物库。</span></div>
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
    <label class="col-span-2 text-[9px] text-on-surface-variant">食物名称<input v-model="custom.name" maxlength="200" class="w-full mt-1 bg-surface-container-low rounded-md p-2 text-xs"></label>
    <label class="text-[9px] text-on-surface-variant">实际克重<input v-model.number="custom.grams" type="number" min="1" class="w-full mt-1 bg-surface-container-low rounded-md p-2 text-xs"></label>
    <label class="text-[9px] text-on-surface-variant">kcal/100g<input v-model.number="custom.kcal" type="number" min="0" class="w-full mt-1 bg-surface-container-low rounded-md p-2 text-xs"></label>
    <label v-for="field in nutrientFields" :key="field.key" class="text-[9px] text-on-surface-variant">{{field.label}}/100g<input v-model.number="custom[field.key]" type="number" min="0" step="0.1" class="w-full mt-1 bg-surface-container-low rounded-md p-2 text-xs"></label>
   </div>
   <div class="grid grid-cols-2 gap-2">
    <button type="button" @click="addCustom" class="add-edit-food flex-1">+ 加入本餐</button>
    <button type="button" @click="saveCustomToLibrary(custom)" :disabled="busy" class="fz-button border-primary/40 text-primary flex-1">保存到食物库</button>
   </div>
  </div>
  <div v-else class="p-3 space-y-3">
   <textarea v-model="aiQuery" rows="2" maxlength="500" class="w-full bg-surface-container-low border border-surface-container-high rounded-lg p-2 text-xs resize-none" placeholder="描述食物、品牌和烹饪方式"></textarea>
   <button type="button" @click="askAi" :disabled="aiLoading" class="add-edit-food">{{aiLoading?'AI 查询中…':'询问 AI 营养成分'}}</button>
   <div v-if="aiResult" class="rounded-lg border border-primary/30 bg-surface-container p-3 space-y-2">
    <div class="flex justify-between gap-2"><div><strong class="text-xs text-primary">AI 估算结果</strong><p class="text-[9px] text-on-surface-variant">{{aiNotes}}</p></div><span class="text-[9px] text-primary">置信度 {{aiConfidence}}</span></div>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2"><label class="col-span-2 text-[9px] text-on-surface-variant">名称<input v-model="aiResult.name" class="w-full mt-1 bg-surface-container-low rounded-md p-2 text-xs"></label><label class="text-[9px] text-on-surface-variant">克重<input v-model.number="aiResult.grams" type="number" min="1" class="w-full mt-1 bg-surface-container-low rounded-md p-2 text-xs"></label><label class="text-[9px] text-on-surface-variant">kcal/100g<input v-model.number="aiResult.kcal" type="number" min="0" class="w-full mt-1 bg-surface-container-low rounded-md p-2 text-xs"></label><label v-for="field in nutrientFields" :key="field.key" class="text-[9px] text-on-surface-variant">{{field.label}}/100g<input v-model.number="aiResult[field.key]" type="number" min="0" step="0.1" class="w-full mt-1 bg-surface-container-low rounded-md p-2 text-xs"></label></div>
    
    <div class="grid grid-cols-2 gap-2">
     <button type="button" @click="addAi" class="add-edit-food flex-1">确认并加入本餐</button>
     <button type="button" @click="saveAiToCustom" :disabled="busy" class="fz-button border-primary/40 text-primary flex-1">{{aiSaved?'已保存到食物库':'保存到食物库'}}</button>
    </div>
   </div>
  </div>
 </div>


 <Teleport to="body"><div v-if="catalogOpen" class="fixed inset-0 z-[160] bg-scrim/80 flex items-center justify-center p-4" @click.self="closeCatalog">
  <section ref="catalogDialog" tabindex="-1" role="dialog" aria-modal="true" :aria-busy="libraryBusy" aria-labelledby="edit-catalog-title" @keydown="catalogKey" class="catalog-dialog w-full max-w-[760px] rounded-2xl border border-outline-variant bg-surface-container-lowest shadow-2xl flex flex-col overflow-hidden">
   <header class="shrink-0 flex items-center justify-between gap-3 px-5 py-4 border-b border-outline-variant"><div><h3 id="edit-catalog-title" class="text-base font-semibold">食物库</h3></div><button type="button" @click="closeCatalog" :disabled="libraryBusy" aria-label="关闭食物库" class="fz-button"><span aria-hidden="true" class="material-symbols-outlined">close</span></button></header>
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
   <footer class="shrink-0 px-5 py-3 border-t border-outline-variant flex items-center justify-between gap-2"><span role="status" class="text-xs text-on-surface-variant">{{libraryBusy?'正在更新食物库…':managingLibrary?libraryFoods.length+' 项食物':'已选 '+foods.length+' 项食物'}}</span><button type="button" @click="managingLibrary?toggleLibraryManagement():closeCatalog()" :disabled="libraryBusy" class="fz-button fz-primary">{{managingLibrary?'完成管理':'完成选择'}}</button></footer>
  </section>
 </div></Teleport>

</div>
<div class="edit-diff">
 <div class="section-line"><strong><i class="live-dot"></i>本次修改变动（实时重算）</strong><span>原 {{num(original.calories)}} kcal → <b>{{ready?num(total.calories):'—'}}</b> kcal <small>{{ready?signed(delta)+' kcal':'待匹配'}}</small></span></div>
 <div class="diff-macros"><span>蛋白质 <b class="green">{{ready?num(total.protein_g):'—'}}g</b> <em v-if="ready" class="green">({{signed(total.protein_g-original.protein_g)}}g)</em></span><span>碳水 <b>{{ready?num(total.carbs_g):'—'}}g</b></span><span>脂肪 <b>{{ready?num(total.fat_g):'—'}}g</b></span><span v-if="proteinProgress!==null&&ready" class="green progress-caption"><span class="material-symbols-outlined">auto_graph</span>今日蛋白达成率 {{proteinProgress}}%</span></div>
 <div class="diff-progress" role="meter" aria-label="今日蛋白达成率" :aria-valuenow="Math.min(100,proteinProgress??0)" aria-valuemin="0" aria-valuemax="100"><i :style="{width:Math.min(100,proteinProgress??0)+'%'}"></i></div>
</div>
<p v-if="error" class="edit-error form-error" role="alert">{{error}}</p>
<footer class="edit-footer"><DeleteButton :action="deleteMeal" subject="此条饮食记录" label="删除此条记录" :disabled="busy" compact /><div><button type="button" class="cancel-edit" :disabled="busy" @click="close">放弃修改</button><button class="save-edit" type="submit" :disabled="busy||!ready"><span class="material-symbols-outlined">{{busy?'hourglass_top':'check'}}</span>{{busy?'保存中…':'保存修改记录'}}</button></div></footer>
</form>
</div>
</Teleport>
</template>
<style scoped>
.catalog-food-select:disabled{opacity:1;cursor:default}

.food-library-button{display:inline-flex;align-items:center;justify-content:center;gap:6px;white-space:nowrap}
.food-library-button .material-symbols-outlined{display:block;flex-shrink:0;font-size:16px;line-height:1}

.meal-edit-overlay{position:fixed;inset:0;z-index:120;background:rgb(var(--fz-rgb-inverse-surface) / .8);backdrop-filter:blur(12px);display:flex;align-items:center;justify-content:center;padding:16px;color:var(--fz-on-surface,var(--fz-on-surface))}
.meal-editor{position:relative;width:100%;max-width:620px;max-height:calc(100dvh - 32px);display:flex;flex-direction:column;overflow:hidden;border-radius:12px;background:var(--fz-surface-container-low,var(--fz-surface-container-lowest));box-shadow:0 24px 64px -12px rgb(var(--fz-rgb-shadow) / .12);outline:none;font-size:12px;line-height:1.5}
.meal-editor:before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--fz-primary),transparent);opacity:.8}
.meal-editor button:not(.fz-delete-button){cursor:pointer;transition:background .15s,color .15s}.meal-editor button:not(.fz-delete-button):disabled{opacity:.45;cursor:not-allowed}.meal-editor button:focus-visible,.meal-editor input:focus-visible{outline:2px solid var(--fz-primary);outline-offset:2px}
.meal-editor .material-symbols-outlined{font-size:16px;vertical-align:middle}.green{color:var(--fz-primary)}.edit-head{padding:24px 24px 16px;flex-shrink:0}.head-line{display:flex;align-items:center;gap:8px}.head-line h2{margin:0;white-space:nowrap;font-size:18px;font-weight:600;letter-spacing:-.02em}
.edit-emblem{width:36px;height:36px;flex-shrink:0;display:grid;place-items:center;border-radius:8px;background:rgb(var(--fz-rgb-primary) / 0.102);color:var(--fz-primary);box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06);font-size:20px!important}
.edit-status{min-width:0;display:flex;align-items:center;gap:6px;border-radius:99px;background:rgb(var(--fz-rgb-primary) / 0.102);color:var(--fz-primary);padding:2px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:10px}.edit-status i,.live-dot{display:inline-block;flex-shrink:0;width:6px;height:6px;background:var(--fz-primary);border-radius:50%}.close-edit{margin-left:auto;flex-shrink:0;width:32px;height:32px;background:var(--fz-surface-container-low);border-radius:8px}.close-edit:hover{background:var(--fz-surface-container-highest)}
.edit-head p{margin:4px 0 0;padding-left:44px;color:var(--fz-on-surface-variant);font-size:12px}.edit-basics{padding:16px 24px;background:rgb(var(--fz-rgb-surface-container-low) / 0.600);display:grid;gap:8px;flex-shrink:0}.section-line{display:flex;justify-content:space-between;align-items:center;gap:8px}.section-line>span{font-size:10px;color:var(--fz-on-surface-variant)}.section-line>span.green{color:var(--fz-primary)}.section-line strong{font-size:12px;font-weight:600}.section-line small{border-radius:99px;background:var(--fz-surface-container-high);color:var(--fz-on-surface-variant);font-size:10px;font-weight:400;padding:2px 6px;margin-left:4px}
.meal-tabs{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;background:var(--fz-background);padding:4px;border-radius:8px}.meal-tabs button{padding:6px;border-radius:4px;color:var(--fz-on-surface-variant)}.meal-tabs button.selected{background:var(--fz-primary);color:var(--fz-on-primary);font-weight:600;box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06)}.meal-tabs .material-symbols-outlined{font-size:14px}
.time-line{padding-top:4px}.time-line>span{display:flex;align-items:center;gap:4px;font-size:12px}.time-line b{font-size:11px;font-weight:600;background:var(--fz-surface-container-high);padding:2px 4px;border-radius:4px}.time-line button{font-size:10px;white-space:nowrap}.time-input{display:flex;align-items:center;gap:8px}.time-input input{color-scheme:light;background:var(--fz-background);border-radius:6px;padding:6px}
.edit-foods{padding:16px 24px;display:flex;flex-direction:column;gap:8px;max-height:380px;min-height:80px;overflow-y:auto;overscroll-behavior:contain;scrollbar-width:thin;scrollbar-color:var(--fz-outline-variant) transparent}.edit-source-panel{padding:0 24px 16px;flex-shrink:0;max-height:280px;overflow-y:auto;overscroll-behavior:contain;scrollbar-width:thin;scrollbar-color:var(--fz-outline-variant) transparent}.catalog-dialog{height:min(760px,90dvh);max-height:calc(100dvh - 2rem);min-height:0}.catalog-results{flex:1;min-height:0}
.edit-food{padding:16px;border-radius:12px;--fz-food-item-fill:var(--fz-surface-container-low);display:flex;flex-direction:column;gap:4px}.edit-food.modified{box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06)}.food-top,.food-controls,.food-name-group{display:flex;align-items:center;gap:8px}.food-name-group{min-width:0;flex:1;flex-wrap:wrap;gap:4px 8px}.food-name-group input{background:transparent;min-width:70px;width:135px;max-width:100%;font-size:14px;font-weight:600;color:var(--fz-on-surface);border:none;outline:none}.food-name-group small{padding:2px 4px;border-radius:4px;background:var(--fz-surface-container-high);color:var(--fz-on-surface-variant);font-size:10px}.food-name-group .change-badge{background:rgb(var(--fz-rgb-primary) / 0.102);color:var(--fz-primary)}
.food-controls{flex-shrink:0}.food-controls label{display:flex;align-items:center;gap:4px;background:var(--fz-background);border-radius:8px;padding:4px 10px}.food-controls input{width:48px;background:transparent;text-align:right;font-size:11px;font-weight:500;outline:none;appearance:textfield}.food-controls input::-webkit-inner-spin-button{appearance:none}.food-controls label span{font-size:11px;color:var(--fz-on-surface-variant)}.food-controls button:not(.fz-delete-button){width:32px;height:32px;border-radius:8px;color:var(--fz-on-surface-variant)}.food-controls button:not(.fz-delete-button):hover{color:var(--fz-error);background:var(--fz-surface-container-high)}
.food-macros{display:flex;align-items:center;gap:16px;padding-top:4px;color:var(--fz-on-surface-variant);font-size:11px;flex-wrap:wrap}.food-macros strong{color:var(--fz-on-surface);font-weight:500}.food-macros strong span{color:var(--fz-on-surface-variant);font-weight:400}.food-macros em{margin-left:auto;color:var(--fz-on-surface-variant);font-size:10px;font-style:normal}.add-edit-food{width:100%;display:flex;align-items:center;justify-content:center;gap:4px;background:var(--fz-background);border-radius:12px;padding:10px 16px;color:var(--fz-on-surface-variant)}.add-edit-food:hover{color:var(--fz-primary);background:var(--fz-surface-container-low)}
.edit-diff{padding:16px 24px;background:var(--fz-surface-container-low);display:grid;gap:8px;flex-shrink:0}.edit-diff .live-dot{margin-right:4px}.edit-diff .section-line>b,.edit-diff .section-line span>b{color:var(--fz-primary);font-size:14px}.edit-diff .section-line small{background:rgb(var(--fz-rgb-primary) / 0.200);color:var(--fz-primary);font-weight:600;border-radius:4px}.diff-macros{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;padding:8px;background:var(--fz-background);border-radius:8px;font-size:11px;color:var(--fz-on-surface-variant)}.diff-macros b{color:var(--fz-on-surface)}.diff-macros b.green{color:var(--fz-primary)}.diff-macros em{font-style:normal;font-size:10px}.progress-caption{font-size:10px}.diff-progress{height:6px;border-radius:99px;background:var(--fz-background);overflow:hidden}.diff-progress i{display:block;height:100%;background:var(--fz-primary);transition:width .15s}
.edit-footer{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:16px 24px;flex-shrink:0}.edit-footer>div{display:flex;gap:8px}.edit-footer button:not(.fz-delete-button){display:flex;align-items:center;justify-content:center;gap:6px;padding:8px 16px;border-radius:8px;white-space:nowrap}.cancel-edit{background:var(--fz-surface-container-low)}.save-edit{background:var(--fz-primary);color:var(--fz-on-primary);font-size:14px;font-weight:600;box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06)}
.edit-error{color:var(--fz-error);font-size:11px}.edit-error button{text-decoration:underline}.form-error{padding:8px 24px}.lookup-status{font-size:11px;color:var(--fz-on-surface-variant)}.manual-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.manual-grid label{display:grid;gap:4px;color:var(--fz-on-surface-variant)}.manual-grid .description{grid-column:1/-1}.manual-grid input{padding:8px;background:var(--fz-background);border-radius:6px;color:var(--fz-on-surface)}
@media(max-width:600px){.edit-head,.edit-basics,.edit-foods,.edit-source-panel,.edit-diff,.edit-footer{padding-left:16px;padding-right:16px}.head-line{flex-wrap:wrap}.edit-status{max-width:100%;order:3;margin-left:44px}.edit-head p{padding-left:0;margin-top:8px}.food-top{align-items:flex-start}.food-name-group{flex-direction:column;align-items:flex-start}.food-macros{gap:8px}.edit-footer{flex-wrap:wrap}.edit-footer>div{margin-left:auto}.time-line{flex-wrap:wrap}.edit-diff .section-line{flex-wrap:wrap}.edit-head{padding-top:16px}}
</style>
