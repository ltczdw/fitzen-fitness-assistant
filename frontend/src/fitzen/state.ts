import { reactive, nextTick } from 'vue'
function currentLocalMinute(){
 const now=new Date();now.setSeconds(0,0)
 return new Date(now.getTime()-now.getTimezoneOffset()*60000).toISOString().slice(0,16)
}
export type Citation = {title:string;source:string;chunk_id?:string|null;page?:string|number|null;url?:string|null}
export type ChatMode = 'legacy'|'react_readonly'|'react_write'
export type ConfirmationSelector = {pending_request_id:string;pending_op_id:string;pending_digest:string}
export type PendingConfirmation = ConfirmationSelector & {expires_at:string;unavailable?:string}
type ChatRequest = {message:string;conversationId:string;requestId:string;index:number;mode:ChatMode;confirmation?:ConfirmationSelector}
export type ChatWrite = {status:string;receipts:string[];pending:PendingConfirmation[];settled:boolean;reason:string;blocked:boolean}
export type ChatMessage = {role:string;content:string;citations?:Citation[];run_id?:string;message_id?:string;feedback?:string;feedbackPending?:boolean;request?:ChatRequest;write?:ChatWrite}
export type Profile = { user_id?:string; display_name?:string; goal?:string; experience?:string; available_days?:number; body_weight_kg?:number|null; target_weight_kg?:number|null; height_cm?:number|null; body_fat_pct?:number|null }
export type FoodDraft = {name:string;grams:number;kcal:number;protein:number;carbs:number;fat:number;fiber?:number;source?:'local'|'custom'|'ai'}
export type RecordItem = {id:string|number; date?:string; session_date?:string; source_record_id?:number|null; title?:string; exercise?:string; description?:string; meal_type?:string; calories?:number; protein_g?:number; carbs_g?:number; fat_g?:number; source?:string; sets?:number; reps?:number; weight_kg?:number; exercises?:any[]; items?:any[]}
export const ui = reactive({
 route: location.hash.slice(2) || 'dashboard', preview:false,
 loading:false, online:false, loadError:'', notice:'', menu:false, search:false, searchText:'',
 records:false, recordsTab:'training', recordSearch:'', recordPage:1, recordRange:'全部',
 foodOpen:false, foodError:'', savingFood:false, meal:'午餐', mealTime:'', mealDate:'', foodSearch:'', foodKey:crypto.randomUUID(),
 foods:[] as FoodDraft[],
 targets:{} as Record<string,any>, data:null as any, profile:{} as Profile, workouts:[] as RecordItem[], sessions:[] as RecordItem[], nutrition:[] as RecordItem[], summary:{} as Record<string,number>,
 message:'', chatMode:'react_readonly' as ChatMode, sending:false, chatError:'', canRetryChat:false, messages:[] as ChatMessage[], conversationId:'',
 chatRecordCapability:{checked:false,available:false,reason:''},
 conversations:[] as any[], convPanelOpen:false,
})
let noticeTimer: ReturnType<typeof setTimeout>
// One logical message owns its ID until completion or replacement. Transport retries
// reuse this object; a new/edited send always creates a new ID.
let pendingChat:ChatRequest|null=null
const confirmationStates=new Map<string,string>()
function confirmationKey(target:ConfirmationSelector){return [target.pending_request_id,target.pending_op_id,target.pending_digest].join(':')}
function clearChatRetry(){pendingChat=null;ui.canRetryChat=false}
let chatAbort:AbortController|null=null
let cancelChatRender:(()=>void)|null=null
export function notify(message:string){ui.notice=message;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>ui.notice='',4500)}
export function navigate(route:string){ui.menu=false;ui.search=false;location.hash='/'+route}
export async function api(path:string,init:RequestInit={}){
 const response=await fetch('/api'+path,{...init,headers:{'Content-Type':'application/json',...init.headers},signal:init.signal||AbortSignal.timeout(12000)})
 const data=await response.json().catch(()=>null)
 if(!response.ok)throw new Error(typeof data?.detail==='string'?data.detail:'请求失败，请检查服务与输入。')
 return data
}
let chatCapabilityFlight:Promise<boolean>|null=null
export function chatRecordStatusText(){
 const capability=ui.chatRecordCapability
 if(!capability.checked)return '正在检查聊天记录功能…'
 if(capability.available)return '记录模式 · 保存结果以实际回执为准'
 if(capability.reason==='react_write_disabled')return '聊天记录尚未开放，可用加号手动添加'
 if(capability.reason==='llm_not_configured')return 'AI记录服务尚未配置，可用加号手动添加'
 return '聊天记录暂不可用，可稍后重试或用加号手动添加'
}
export async function loadChatCapabilities():Promise<boolean>{
 if(chatCapabilityFlight)return chatCapabilityFlight
 const flight=(async()=>{
  try{
   const data=await api('/health')
   const capability=data?.capabilities?.chat_record
   const available=data?.ready===true&&capability?.available===true
   ui.chatRecordCapability={checked:true,available,reason:available?'available':typeof capability?.reason==='string'?capability.reason:'capability_unavailable'}
   return available
  }catch{
   ui.chatRecordCapability={checked:true,available:false,reason:'capability_unavailable'}
   return false
  }
 })()
 chatCapabilityFlight=flight
 try{return await flight}finally{if(chatCapabilityFlight===flight)chatCapabilityFlight=null}
}
let refreshFlight:Promise<void>|null=null
let refreshAgain=false
let foodCatalogFlight:Promise<void>|null=null
export async function refresh():Promise<void>{
 if(refreshFlight){refreshAgain=true;return refreshFlight}
 refreshFlight=(async()=>{
 do{
 refreshAgain=false;ui.loading=true;ui.loadError=''
 try{
  const [data,targets]=await Promise.all([api('/dashboard?user_id=demo-user&days=365'),api('/nutrition/summary?user_id=demo-user&days=1').catch(()=>null)])
  ui.targets=data.profile.body_weight_kg&&targets?.target_ready?targets:{}
  ui.data=data;ui.profile=data.profile;ui.workouts=data.workouts;ui.sessions=data.sessions;ui.nutrition=data.nutrition
  ui.summary=data.today;ui.online=true
  const dashboardCatalog=Array.isArray(data.food_catalog)?data.food_catalog:[]
  foodsCatalog.splice(0,foodsCatalog.length,...dashboardCatalog.map((f:any)=>({...f,grams:100})))
  if(!foodsCatalog.length)await ensureFoodCatalog()
 }catch(error){ui.online=false;ui.loadError=error instanceof Error?error.message:'数据库读取失败'}
 finally{ui.loading=false}
 }while(refreshAgain)
 })()
 try{await refreshFlight}finally{refreshFlight=null}
}
export function openRecords(tab='training'){ui.recordsTab=tab;ui.recordPage=1;ui.recordSearch='';ui.records=true;void refresh()}
export function openWorkout(){window.dispatchEvent(new Event('fitness-open-workout-sessions'))}
export function openFood(mealDate=''){
 ui.foodError='';ui.mealDate=mealDate
 if((!ui.foodOpen&&!ui.foods.length)||!ui.mealTime){const now=currentLocalMinute();ui.mealTime=mealDate?mealDate+now.slice(10):now}
 ui.foodOpen=true
}
export const foodsCatalog=reactive<{id?:number;name:string;grams:number;kcal:number;protein:number;carbs:number;fat:number;fiber?:number}[]>([])
export type CustomFood = {id:number;name:string;grams:number;kcal:number;protein:number;carbs:number;fat:number;fiber:number;usage_count?:number;last_used_at?:string|null}
export const customFoods=reactive<CustomFood[]>([])
export type ExerciseMeta={name:string;category:string;target_muscles:string;source?:string}
export const exerciseLibrary=reactive<ExerciseMeta[]>([])
let exerciseLibraryFlight:Promise<void>|null=null
let exerciseLibraryLoaded=false
let exerciseLibraryRevision=0
const exerciseLibraryWrites=new Map<string,{revision:number;item:ExerciseMeta}>()
export function upsertExerciseMeta(value:any){
 const name=String(value?.name||'').trim()
 if(!name)return
 const item:ExerciseMeta={name,category:String(value?.category||'').trim(),target_muscles:String(value?.target_muscles||'').trim(),source:String(value?.source||'')}
 const key=name.toLowerCase()
 exerciseLibraryRevision+=1
 exerciseLibraryWrites.set(key,{revision:exerciseLibraryRevision,item})
 const index=exerciseLibrary.findIndex(entry=>entry.name.trim().toLowerCase()===key)
 if(index>=0)exerciseLibrary.splice(index,1,item)
 else exerciseLibrary.push(item)
}
export async function ensureExerciseLibrary(force=false){
 if(!force&&exerciseLibraryLoaded)return
 if(exerciseLibraryFlight){await exerciseLibraryFlight;return}
 const revisionAtStart=exerciseLibraryRevision
 const flight=(async()=>{
  try{
   const data=await api('/workout-exercises?user_id=demo-user&limit=300')
   const rows=Array.isArray(data?.exercises)?data.exercises:[]
   const mapped=rows.map((item:any)=>({name:String(item?.name||'').trim(),category:String(item?.category||'').trim(),target_muscles:String(item?.target_muscles||'').trim(),source:String(item?.source||'')})).filter((item:ExerciseMeta)=>Boolean(item.name))
   const merged=new Map<string,ExerciseMeta>(mapped.map((item:ExerciseMeta)=>[item.name.toLowerCase(),item]))
   for(const [key,entry] of exerciseLibraryWrites){
    if(entry.revision>revisionAtStart||!merged.has(key))merged.set(key,entry.item)
   }
   exerciseLibrary.splice(0,exerciseLibrary.length,...merged.values())
   exerciseLibraryLoaded=true
  }catch{}
 })()
 exerciseLibraryFlight=flight
 try{await flight}finally{if(exerciseLibraryFlight===flight)exerciseLibraryFlight=null}
}
let customFoodCatalogFlight:Promise<void>|null=null
let customFoodCatalogLoaded=false
export async function ensureFoodCatalog(limit=12){
 if(foodsCatalog.length)return
 if(foodCatalogFlight){await foodCatalogFlight;return}
 const flight=(async()=>{
  try{
   const data=await api('/nutrition/food/catalog?limit='+encodeURIComponent(String(limit))+'&user_id=demo-user')
   const rows=Array.isArray(data?.foods)?data.foods:[]
   const mapped=rows.map((f:any)=>({id:f.id,name:String(f.name||''),grams:100,kcal:Number(f.kcal||0),protein:Number(f.protein??f.protein_g??0),carbs:Number(f.carbs??f.carb_g??0),fat:Number(f.fat??f.fat_g??0),fiber:Number(f.fiber??f.fiber_g??0)})).filter((f:any)=>f.name)
   if(mapped.length)foodsCatalog.splice(0,foodsCatalog.length,...mapped)
  }catch{}
 })()
 foodCatalogFlight=flight
 try{await flight}finally{if(foodCatalogFlight===flight)foodCatalogFlight=null}
}
export async function ensureCustomFoods(force=false, limit=200){
 if(!force&&customFoodCatalogLoaded)return
 if(customFoodCatalogFlight){await customFoodCatalogFlight;return}
 const flight=(async()=>{
  try{
   const data=await api('/nutrition/custom-foods?limit='+encodeURIComponent(String(limit))+'&user_id=demo-user')
   const rows=Array.isArray(data?.foods)?data.foods:[]
   const mapped=rows.map((f:any)=>({id:Number(f.id),name:String(f.name||''),grams:100,kcal:Number(f.kcal||0),protein:Number(f.protein??f.protein_g??0),carbs:Number(f.carbs??f.carb_g??0),fat:Number(f.fat??f.fat_g??0),fiber:Number(f.fiber??f.fiber_g??0),usage_count:Number(f.usage_count||0),last_used_at:f.last_used_at||null})).filter((f:any)=>f.name&&Number.isFinite(f.id))
   customFoods.splice(0,customFoods.length,...mapped)
   customFoodCatalogLoaded=true
  }catch{customFoodCatalogLoaded=false}
 })()
 customFoodCatalogFlight=flight
 try{await flight}finally{if(customFoodCatalogFlight===flight)customFoodCatalogFlight=null}
}
export type FoodSuggestion = {id?:number;name:string;category:string;preparation_state:string;kcal:number;protein:number;carbs:number;fat:number;fiber:number;source:'local'|'custom'}
export async function findFoodSuggestions(query:string,signal?:AbortSignal):Promise<FoodSuggestion[]>{
 const term=query.trim();if(!term)return []
 const suffix='?search='+encodeURIComponent(term)+'&limit=30&user_id=demo-user'
 const requestSignal=signal?AbortSignal.any([signal,AbortSignal.timeout(12000)]):AbortSignal.timeout(12000)
 const [local,personal]=await Promise.all([api('/nutrition/food/catalog'+suffix,{signal:requestSignal}),api('/nutrition/custom-foods'+suffix,{signal:requestSignal})])
 const map=(food:any,source:'local'|'custom'):FoodSuggestion=>({id:food.id,name:String(food.name||''),category:source==='custom'?'自定义':String(food.category||''),preparation_state:source==='custom'?'用户自定义':String(food.preparation_state||''),kcal:Number(food.kcal),protein:Number(food.protein??food.protein_g),carbs:Number(food.carbs??food.carb_g),fat:Number(food.fat??food.fat_g),fiber:Number(food.fiber??food.fiber_g??0),source})
 const rows=[...(Array.isArray(local?.foods)?local.foods:[]).map((food:any)=>map(food,'local')),...(Array.isArray(personal?.foods)?personal.foods:[]).map((food:any)=>map(food,'custom'))]
 const exact=term.replace(/\s+/g,'').toLowerCase()
 return rows.filter(food=>food.name.trim()&&[food.kcal,food.protein,food.carbs,food.fat,food.fiber].every(value=>Number.isFinite(value)&&value>=0)).sort((a,b)=>Number(b.name.replace(/\s+/g,'').toLowerCase()===exact)-Number(a.name.replace(/\s+/g,'').toLowerCase()===exact)).slice(0,20)
}
export async function saveCustomFood(food:Pick<FoodDraft,'name'|'kcal'|'protein'|'carbs'|'fat'|'fiber'>){
 const data=await api('/nutrition/custom-foods',{method:'POST',body:JSON.stringify({user_id:'demo-user',food_name:food.name.trim(),nutrition_per_100g:{kcal:Number(food.kcal||0),protein_g:Number(food.protein||0),carbs_g:Number(food.carbs||0),fat_g:Number(food.fat||0),fiber_g:Number(food.fiber||0)}})})
 const saved=data?.food as CustomFood|undefined
 if(data?.saved!==true||!saved||!Number.isInteger(saved.id)||saved.id<=0)throw new Error('保存结果无效，请重新加载食物库后重试')
 await ensureCustomFoods(true)
 const index=customFoods.findIndex(item=>item.id===saved.id)
 if(index>=0)customFoods.splice(index,1,saved);else customFoods.unshift(saved)
 return saved
}
export async function updateLibraryFood(original:{id?:number;name:string;source?:'local'|'custom'},food:Pick<FoodDraft,'name'|'kcal'|'protein'|'carbs'|'fat'|'fiber'>){
 if(!Number.isInteger(original.id)||Number(original.id)<=0)throw new Error('此条目缺少编号，请重新加载食物库后重试')
 if(customFoodCatalogFlight)await customFoodCatalogFlight
 if(foodCatalogFlight)await foodCatalogFlight
 const path=original.source==='custom'?'/nutrition/custom-foods/':'/nutrition/food/catalog/'
 const result=await api(path+original.id,{method:'PUT',body:JSON.stringify({user_id:'demo-user',food_name:food.name.trim(),nutrition_per_100g:{kcal:food.kcal,protein_g:food.protein,carbs_g:food.carbs,fat_g:food.fat,fiber_g:food.fiber??0}})})
 const saved=result?.food as CustomFood|undefined
 if(result?.saved!==true||!saved||saved.id!==original.id||!saved.name||![saved.kcal,saved.protein,saved.carbs,saved.fat,saved.fiber].every(Number.isFinite))throw new Error('保存结果无效，请重新加载食物库后重试')
 if(original.source==='custom'){const i=customFoods.findIndex(item=>item.id===original.id);if(i>=0)customFoods.splice(i,1,{...saved});else customFoods.unshift({...saved})}
 else{const i=foodsCatalog.findIndex(item=>item.id===original.id||(!item.id&&item.name===original.name));if(i>=0)foodsCatalog.splice(i,1,{...saved})}
 return saved
}
export async function deleteLibraryFood(food:{id?:number;name:string;source?:'local'|'custom'}){
 if(!Number.isInteger(food.id)||Number(food.id)<=0)throw new Error('此条目缺少编号，请重新加载食物库后重试')
 // Wait for outstanding catalog reads before pruning their shared cache.
 if(customFoodCatalogFlight)await customFoodCatalogFlight
 if(foodCatalogFlight)await foodCatalogFlight
 const path=food.source==='custom'?'/nutrition/custom-foods/':'/nutrition/food/catalog/'
 const result=await api(path+food.id+'?user_id=demo-user',{method:'DELETE'})
 if(result?.deleted!==true)throw new Error('删除结果无效，请重新加载食物库后重试')
 if(food.source==='custom'){const i=customFoods.findIndex(item=>item.id===food.id);if(i>=0)customFoods.splice(i,1)}
 else for(let i=foodsCatalog.length-1;i>=0;i--)if(foodsCatalog[i].id===food.id||foodsCatalog[i].name===food.name)foodsCatalog.splice(i,1)
}
export function addNutritionFood(food:FoodDraft){
 if(!ui.foods.length&&!ui.mealTime){const now=currentLocalMinute();ui.mealTime=ui.mealDate?ui.mealDate+now.slice(10):now}
 ui.foods.push(food)
}
export async function addFood(name:string){
 if(!name.trim())return
 try{const data=await api('/nutrition/food/lookup?name='+encodeURIComponent(name.trim()));const p=data.per_100g;addNutritionFood({name:data.food_name,grams:100,kcal:p.kcal||0,protein:p.protein_g||0,carbs:p.carb_g||0,fat:p.fat_g||0,fiber:p.fiber_g||0,source:'local'});ui.foodSearch='';ui.foodError=''}
 catch(error){ui.foodError=error instanceof Error?error.message:'无法匹配食物'}
}
export async function saveFood(){
 if(ui.savingFood)return
 if(!ui.foods.length||ui.foods.some(f=>!Number.isFinite(f.grams)||f.grams<=0||f.grams>100000)){ui.foodError='请添加食物，并填写大于 0 的有效克数。';return}
 const recordTime=new Date(ui.mealTime)
 if(!ui.mealTime||Number.isNaN(recordTime.getTime())){ui.foodError='请选择有效的用餐日期和时间。';return}
 ui.savingFood=true;ui.foodError=''
 try{
 await api('/nutrition/food',{method:'POST',headers:{'Idempotency-Key':ui.foodKey},body:JSON.stringify({user_id:'demo-user',meal_type:ui.meal,record_time:recordTime.toISOString(),foods:ui.foods.map(f=>({food_name:f.name,weight_g:f.grams,source:f.source||'local',...((f.source==='custom'||f.source==='ai')?{nutrition_per_100g:{kcal:f.kcal,protein_g:f.protein,carbs_g:f.carbs,fat_g:f.fat,fiber_g:f.fiber||0}}:{})}))})})
 ui.foodOpen=false;ui.foods=[];ui.mealTime='';ui.mealDate='';ui.foodKey=crypto.randomUUID();void ensureCustomFoods(true);notify('本次摄入已保存');await refresh()
 }catch(error){ui.foodError=error instanceof Error?error.message:'保存失败'}finally{ui.savingFood=false}
}
export async function saveProfileData(patch:Profile){
 try{ui.profile=await api('/profile',{method:'PUT',body:JSON.stringify({...ui.profile,...patch,user_id:'demo-user'})});await refresh();if(!ui.online){notify('已提交，但暂时无法确认数据库内容，请恢复连接后刷新。');return false}notify('个人资料已保存');return true}
 catch(error){notify(error instanceof Error?error.message:'保存失败');return false}
}
function getConversationId(){try{return localStorage.getItem('fitness-assistant-chat-conversation-id-v1')||crypto.randomUUID()}catch{return crypto.randomUUID()}}
ui.conversationId=getConversationId()
try{localStorage.setItem('fitness-assistant-chat-conversation-id-v1',ui.conversationId)}catch{}
export async function loadHistory(){
 const id=ui.conversationId,before=ui.messages
 try{ui.chatError='';const data=await api('/chat/history?user_id=demo-user&conversation_id='+encodeURIComponent(id));if(id!==ui.conversationId||ui.sending||before!==ui.messages||ui.messages.some(m=>m.request))return;ui.messages=Array.isArray(data)?data:(data.messages||[])}catch{if(id===ui.conversationId&&!ui.sending&&before===ui.messages&&!ui.messages.some(m=>m.request))ui.chatError='历史会话暂时无法加载。'}
}
export async function newChat(){if(ui.sending)return;clearChatRetry();ui.chatError='';ui.conversationId=crypto.randomUUID();ui.messages=[];ui.preview=false;try{localStorage.setItem('fitness-assistant-chat-conversation-id-v1',ui.conversationId)}catch{};notify('已开始新会话');void loadConversations()}
let conversationLoadVersion=0
export async function loadConversations(){
 const version=++conversationLoadVersion
 try{
  const data=await api('/chat/conversations?user_id=demo-user')
  if(version!==conversationLoadVersion)return
  if(Array.isArray(data?.conversations))ui.conversations=data.conversations
 }catch{ /* A temporary list failure must not erase already displayed sessions. */ }
}
export async function switchConversation(id:string){
 if(!id||id===ui.conversationId||ui.sending)return
 clearChatRetry()
 ui.conversationId=id;ui.messages=[];ui.chatError='';ui.convPanelOpen=false
 try{localStorage.setItem('fitness-assistant-chat-conversation-id-v1',id)}catch{}
 await loadHistory()
}
export async function deleteConversation(id:string){
 if(!id||ui.sending)return
 try{await api('/chat/conversations/'+encodeURIComponent(id)+'?user_id=demo-user',{method:'DELETE'})}catch(e){notify(e instanceof Error?e.message:'删除失败');return}
 ui.conversations=ui.conversations.filter(item=>item.conversation_id!==id)
 if(id===ui.conversationId){clearChatRetry();ui.chatError='';ui.conversationId=crypto.randomUUID();ui.messages=[];try{localStorage.setItem('fitness-assistant-chat-conversation-id-v1',ui.conversationId)}catch{}}
 await loadConversations()
}
export function stopChat(){
 if(ui.sending)ui.chatError='已停止接收回复；保存可能已执行，请核对回执或用原请求恢复。'
 cancelChatRender?.();chatAbort?.abort()
}
function appendChat(message:string,mode:ChatMode,confirmation?:ConfirmationSelector){
 ui.messages.push({role:'user',content:message},{role:'assistant',content:''})
 const request:ChatRequest={message,mode,conversationId:ui.conversationId,requestId:crypto.randomUUID(),index:ui.messages.length-1,...(confirmation?{confirmation}:{})}
 ui.messages[request.index].request=request
 return request
}
export async function sendChat(){
 const message=ui.message.trim();if(!message||ui.sending)return
 if(message.length>4000){notify('消息不能超过 4000 字');return}
 const mode=ui.chatMode
 if(mode==='react_write'&&!await loadChatCapabilities()){notify(chatRecordStatusText());return}
 // The draft/mode may have changed while readiness was being checked.
 if(ui.sending||ui.message.trim()!==message||ui.chatMode!==mode)return
 ui.message='';pendingChat=appendChat(message,mode)
 await transmitChat(pendingChat)
}
export async function retryChat(){
 if(ui.sending||!pendingChat||pendingChat.conversationId!==ui.conversationId)return
 // A changed draft is a new request; a transport retry owns its original payload.
 if(ui.message.trim()){await sendChat();return}
 await transmitChat(pendingChat)
}
export function pendingUnavailable(item:ChatMessage,target:PendingConfirmation,now=Date.now()){
 if(target.unavailable)return target.unavailable
 if(confirmationStates.has(confirmationKey(target)))return confirmationStates.get(confirmationKey(target))!
 if(!item.request||item.request.conversationId!==ui.conversationId)return '会话已变更'
 if(item.write?.blocked)return '该请求已失效'
 if(!item.write?.settled)return '结果尚未完整接收，请先恢复原请求'
 if(!Number.isFinite(Date.parse(target.expires_at))||Date.parse(target.expires_at)<=now)return '确认已过期，请重新描述记录'
 return ''
}
export async function confirmPending(index:number,targetIndex:number){
 const item=ui.messages[index],target=item?.write?.pending[targetIndex]
 if(ui.sending||!target||pendingUnavailable(item,target))return
 // Never send expiry, display state, or any other fields inside the strict selector.
 const {pending_request_id,pending_op_id,pending_digest}=target
 confirmationStates.set(confirmationKey(target),'确认结果待恢复')
 pendingChat=appendChat('确认保存','react_write',{pending_request_id,pending_op_id,pending_digest})
 await transmitChat(pendingChat)
}
export async function resumeChat(index:number){
 const item=ui.messages[index],request=item?.request
 if(ui.sending||!request||request.conversationId!==ui.conversationId||item.write?.blocked)return
 if(item.write?.settled&&item.write.status==='completed')return
 pendingChat=request;await transmitChat(request)
}
function invalidateChatWrites(reason:string){
 for(const item of ui.messages){if(item.write){item.write.blocked=true;item.write.pending.forEach(target=>target.unavailable=reason)}}
 clearChatRetry()
}
function applyChatResult(request:ChatRequest,data:any,settled:boolean){
 const item=ui.messages[request.index]
 if(!item||request.conversationId!==ui.conversationId)return
 if(data.request_id&&data.request_id!==request.requestId)throw new Error('响应标识不匹配，未使用其中的回执或确认。')
 if(Array.isArray(data.citations))item.citations=data.citations
 if(data.diagnostics_status==='ready'){item.run_id=data.run_id;item.message_id=data.message_id}
 if(request.mode!=='react_write')return
 const metrics=data.metrics
 if(metrics&&typeof metrics==='object'){
  const write=item.write!
  write.status=typeof metrics.react_write==='string'&&metrics.react_write?metrics.react_write:metrics.execution_mode==='limited'?'limited':'unknown'
  write.reason=typeof metrics.mode_reason==='string'?metrics.mode_reason:''
  if(write.reason==='react_write_disabled'||write.reason==='react_write_disabled_frozen')ui.chatRecordCapability={checked:true,available:false,reason:'react_write_disabled'}
  if(Array.isArray(metrics.receipt_refs)&&metrics.receipt_refs.every((r:any)=>typeof r==='string'&&r.trim()))write.receipts=[...new Set<string>([...write.receipts,...metrics.receipt_refs])]
  // A selector is only actionable when explicitly supplied for this original request.
  write.pending=write.status==='pending'&&metrics.execution_mode==='react_write'&&Array.isArray(metrics.pending_confirmations)?metrics.pending_confirmations.filter((p:any)=>p&&p.pending_request_id===request.requestId&&typeof p.pending_op_id==='string'&&/^op2_[a-f0-9]{32}$/.test(p.pending_op_id)&&typeof p.pending_digest==='string'&&/^[a-f0-9]{64}$/.test(p.pending_digest)&&typeof p.expires_at==='string'&&Number.isFinite(Date.parse(p.expires_at))).map((p:any)=>({pending_request_id:p.pending_request_id,pending_op_id:p.pending_op_id,pending_digest:p.pending_digest,expires_at:p.expires_at})):[]
  if(write.reason==='privacy_invalidated'){
   // Do not keep replayed private content or old receipts after privacy invalidation.
   for(const m of ui.messages){if(m.write){m.content='该请求的恢复权限已失效。';m.write.receipts=[];m.citations=[];m.run_id=undefined;m.message_id=undefined}}
   invalidateChatWrites('隐私设置已变更')
  }
 }
 item.write!.settled=settled
 if(settled&&request.confirmation&&item.write!.status==='completed'&&item.write!.receipts.length){
  const selector=request.confirmation
  confirmationStates.set(confirmationKey(selector),'已确认保存')
  for(const message of ui.messages)for(const target of message.write?.pending||[]){
   if(target.pending_request_id===selector.pending_request_id&&target.pending_op_id===selector.pending_op_id&&target.pending_digest===selector.pending_digest)target.unavailable='已确认保存'
  }
 }
}
async function transmitChat(request:ChatRequest){
 const {message,conversationId,requestId,index,mode,confirmation}=request
 const item=ui.messages[index];if(!item)return
 ui.preview=false;ui.sending=true;ui.chatError='';ui.canRetryChat=false
 item.content='';item.citations=[];item.run_id=undefined;item.message_id=undefined
 if(mode==='react_write')item.write={status:'unknown',receipts:item.write?.receipts||[],pending:[],settled:false,reason:'',blocked:false}
 const controller=new AbortController();chatAbort=controller
 let receivedDone=false,receivedError=false,retryAllowed=true
 const renderQueue:string[]=[];let rendering=false,cancelRendering=false
 let renderFinished:(()=>void)|null=null
 const pause=(ms:number)=>new Promise<void>(resolve=>window.setTimeout(resolve,ms))
 const finishRender=()=>{const resolve=renderFinished;renderFinished=null;resolve?.()}
 const drainRenderQueue=async()=>{
  if(rendering)return
  rendering=true
  while(renderQueue.length&&!cancelRendering){
   const chunk=renderQueue.shift()||''
   for(let offset=0;offset<chunk.length&&!cancelRendering;offset+=2){
    item.content+=chunk.slice(offset,offset+2);await pause(chunk[offset]==='\n'?24:12)
   }
  }
  rendering=false;finishRender()
 }
 const enqueueRender=(content:string)=>{if(content&&!item.write?.blocked){renderQueue.push(content);void drainRenderQueue()}}
 const waitForRender=()=>rendering||renderQueue.length?new Promise<void>(resolve=>{renderFinished=resolve}):Promise.resolve()
 cancelChatRender=()=>{cancelRendering=true;renderQueue.length=0;finishRender()}
 function event(block:string){
  const kind=block.match(/^event:\s*(.+)$/m)?.[1]?.trim()
  const line=block.split(/\r?\n/).filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trim()).join('\n')
  if(!line)return
  // Invalid protocol is an interruption, not a fabricated successful completion.
  const data=JSON.parse(line)
  if(kind==='token')enqueueRender(typeof data.content==='string'?data.content:'')
  if(kind==='meta'||kind==='done')applyChatResult(request,data,kind==='done')
  if(kind==='done'){receivedDone=true;if(data.status==='failed'){receivedError=true;ui.chatError||='执行未完成，请核对已有回执后恢复原请求。'}}
  if(kind==='error'){
   receivedError=true;ui.chatError=typeof data.message==='string'?data.message:'AI 返回错误'
   if(data.code==='request_id_conflict'){retryAllowed=false;invalidateChatWrites('请求标识冲突')}
  }
  if(item.write?.blocked){cancelChatRender?.();retryAllowed=false}
 }
 try{
  const response=await fetch('/api/chat/stream',{method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json',Accept:'text/event-stream, application/json'},body:JSON.stringify({message,user_id:'demo-user',conversation_id:conversationId,request_id:requestId,mode,...(confirmation?{confirmation}:{})})})
  if(!response.ok){
   const body=await response.json().catch(()=>null)
   if([409,422].includes(response.status)){retryAllowed=false;invalidateChatWrites(response.status===409?'请求标识冲突':'确认或输入不再有效')}
   throw new Error(typeof body?.detail==='string'?body.detail:response.status===422?'输入或确认标识无效，请重新描述记录。':response.status===409?'请求标识冲突，已停止恢复和确认。':'AI 服务暂时不可用，请稍后重试。')
  }
  if(response.headers.get('Content-Type')?.includes('application/json')){
   const data=await response.json();applyChatResult(request,data,true);receivedDone=true
   enqueueRender(typeof data.answer==='string'?data.answer:'')
  }else{
   if(!response.body)throw new Error('未收到回复数据。')
   const reader=response.body.getReader(),decoder=new TextDecoder();let buffer=''
   try{
    while(true){const part=await reader.read();buffer+=decoder.decode(part.value||new Uint8Array(),{stream:!part.done});const chunks=buffer.split(/\r?\n\r?\n/);buffer=chunks.pop()||'';chunks.forEach(event);await nextTick();if(part.done)break}
    if(buffer.trim())event(buffer)
   }finally{await reader.cancel().catch(()=>{});reader.releaseLock()}
  }
  if(controller.signal.aborted)throw new DOMException('Stopped','AbortError')
  if(!receivedDone)throw new Error(ui.chatError||'连接中断，请重试（复用原请求，不重复记录）。')
  await waitForRender()
  if(receivedError&&item.write)item.write.settled=false
  if(controller.signal.aborted)throw new DOMException('Stopped','AbortError')
  if(!item.content&&!ui.chatError&&!item.write?.blocked)ui.chatError='服务未返回文本，请恢复原请求。'
  if(item.write?.blocked)retryAllowed=false
  if(!receivedError&&!ui.chatError){clearChatRetry()}
  void refresh()
 }catch(error){
  cancelChatRender?.()
  if(item.write)item.write.settled=false
  ui.chatError=error instanceof Error&&error.name==='AbortError'?'已停止接收回复；保存可能已执行，请核对回执或用原请求恢复。':error instanceof Error?error.message:'连接中断，请恢复原请求。'
 }finally{
  if(!retryAllowed)clearChatRetry()
  ui.sending=false;ui.canRetryChat=pendingChat===request;chatAbort=null;cancelChatRender=null
  void loadConversations()
 }
}

export function download(name:string,text:string,type='text/plain;charset=utf-8'){
 const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)
}
export function exportRecords(){
 const mirroredWorkoutIds=new Set(ui.sessions.map((item:any)=>Number(item.source_record_id)).filter((id:number)=>Number.isFinite(id)))
 const training=[...ui.sessions,...ui.workouts.filter((item:any)=>!mirroredWorkoutIds.has(Number(item.id)))]
 download('FitZen-'+ui.recordsTab+'-'+new Date().toISOString().slice(0,10)+'.json',JSON.stringify(ui.recordsTab==='nutrition'?ui.nutrition:training,null,2),'application/json');notify('已导出当前真实记录')
}
export function exportChat(){download('FitZen-会话.md',ui.messages.map(m=>'## '+(m.role==='user'?'我':'FitZen AI')+'\n\n'+m.content).join('\n\n'))}
export async function rateAnswer(index:number,rating:'up'|'down'){
 const item=ui.messages[index];if(!item?.run_id||!item.message_id||item.feedback||item.feedbackPending||item.write?.blocked)return
 item.feedbackPending=true
 try{await api('/feedback',{method:'POST',body:JSON.stringify({run_id:item.run_id,message_id:item.message_id,rating})});item.feedback=rating;notify('反馈已保存')}
 catch(error){notify(error instanceof Error?error.message:'反馈保存失败')}
 finally{item.feedbackPending=false}
}
export function useFitzen(){return {rateAnswer,ui,notify,navigate,refresh,openRecords,openWorkout,openFood,addFood,addNutritionFood,saveFood,saveProfileData,sendChat,retryChat,stopChat,confirmPending,resumeChat,pendingUnavailable,newChat,loadConversations,switchConversation,deleteConversation,exportRecords,exportChat}}
