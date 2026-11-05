import {computed} from 'vue'
import {deleteConfirmation} from '../components/ui/delete-confirmation'
import {ui,api,notify,refresh} from './state'
export const fmt=(value:any,digits=0)=>value==null||!Number.isFinite(Number(value))?'—':Number(value).toLocaleString('zh-CN',{maximumFractionDigits:digits})
export const pct=(value:any,target:any)=>value==null||!target?null:Math.round(value/target*100)
export const bar=(value:any,target:any)=>Math.min(100,Math.max(0,pct(value,target)||0))
export const todayKey=computed(()=>ui.data?.end_date||new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date()))
export const today=computed(()=>ui.data?.today||{})
export const daily=computed<any[]>(()=>ui.data?.daily||[])
export const week=computed(()=>daily.value.slice(-7))
export const currentWeek=computed(()=>{const offset=(new Date(todayKey.value+'T12:00:00').getDay()+6)%7;return daily.value.slice(-(offset+1))})
export const weekVolume=computed(()=>currentWeek.value.reduce((sum,d)=>sum+d.training_volume,0))
export const weekSets=computed(()=>currentWeek.value.reduce((sum,d)=>sum+d.completed_sets,0))
export const weekDays=computed(()=>currentWeek.value.filter(d=>d.training_records>0).length)
export const records=computed<any[]>(()=>{
  const mirroredWorkoutIds=new Set(ui.sessions.map((item:any)=>Number(item.source_record_id)).filter((id:number)=>Number.isFinite(id)))
  return [...ui.sessions,...ui.workouts.filter((item:any)=>!mirroredWorkoutIds.has(Number(item.id)))].sort((a,b)=>(b.session_date||b.date||'').localeCompare(a.session_date||a.date||''))
})
export function setsFor(record:any):any[]{return record.exercises?record.exercises.flatMap((e:any)=>e.sets||[]):record.set_details??Array.from({length:record.sets||0},()=>({weight_kg:record.weight_kg,reps:record.reps,completed:record.completed,difficulty:record.difficulty,note:record.note}))}
export function editRecord(record:any,nutrition=false){window.dispatchEvent(new CustomEvent(nutrition?'fitness-edit-nutrition-record':record.exercises?'fitness-edit-workout-session':'fitness-edit-workout-record',{detail:record}))}
export function clearMealDeleteConfirmation(){deleteConfirmation.clear()}
export async function removeMeal(record:any){
 clearMealDeleteConfirmation()
 try{await api((record.source==='ai_food_log'?'/nutrition/food/':'/nutrition/')+encodeURIComponent(record.id)+'?user_id=demo-user',{method:'DELETE'});await refresh();notify('记录已删除')}catch(e){notify(e instanceof Error?e.message:'删除失败')}
}
export function sumMeals(rows:any[],key:string){return rows.length&&rows.every(row=>row[key]!=null)?rows.reduce((n,row)=>n+Number(row[key]),0):null}
export const calendar=computed(()=>{const date=new Date(todayKey.value+'T12:00:00');date.setDate(date.getDate()-(date.getDay()+6)%7);return Array.from({length:7},(_,i)=>{const d=new Date(date);d.setDate(d.getDate()+i);const key=[d.getFullYear(),String(d.getMonth()+1).padStart(2,'0'),String(d.getDate()).padStart(2,'0')].join('-');const items=records.value.filter(r=>(r.session_date||r.date||'').slice(0,10)===key);return {date:key,label:['周一','周二','周三','周四','周五','周六','周日'][i],today:key===todayKey.value,title:items.map(r=>r.title||r.exercise).join('、')||'暂无训练记录',sets:items.flatMap(setsFor).filter(s=>s.completed!==false).length,actions:items.reduce((n,r)=>n+(r.exercises?.length??1),0),has:items.length>0}})})
export const streak=computed(()=>{let count=0;const days=[...daily.value].reverse();if(days[0]&&!days[0].training_records&&!days[0].meal_count)days.shift();for(const d of days){if(!d.training_records&&!d.meal_count)break;count++}return count})

