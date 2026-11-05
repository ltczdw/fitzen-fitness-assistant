import {reactive} from 'vue'
export type RecordItem=any
export const evidence=reactive({requests:[] as any[],notifications:[] as string[]})
export const ui=reactive({targets:{},summary:{}})
export const customFoods:any[]=[];export const foodsCatalog:any[]=[]
export async function ensureFoodCatalog(){};export async function ensureCustomFoods(){}
export function notify(s:string){evidence.notifications.push(s)}
export function openRecords(){};export async function refresh(){};export async function saveCustomFood(){}
export async function api(url:string,init?:any){
 const method=init?.method||'GET'
 evidence.requests.push({url,method,body:init?.body?JSON.parse(init.body):null})
 if(url.startsWith('/nutrition/food/lookup'))return {food_name:'熟大米',per_100g:{kcal:116,protein_g:2.6,carb_g:25.9,carbs_g:25.9,fat_g:.3,fiber_g:.3}}
 if(method==='PUT')return {updated:true}
 if(url.startsWith('/nutrition/food/catalog'))return {foods:[]}
 throw new Error('Fixture forbids unsupported API '+url)
}
export const meal={id:'round2-synthetic-meal',source:'ai_food_log',date:'2026-10-03T03:00:00Z',meal_type:'午餐',items:[{id:11,food_name:'熟大米',raw_food_name:'米饭',weight_g:100,calories:116,protein_g:2.6,carbs_g:25.9,fat_g:.3,source:'local'}]}
export const session={id:100,user_id:'demo-user',session_date:'2026-10-03',title:'合成异质训练',note:'fixture',exercises:[{id:101,exercise_name:'卧推',difficulty:'未记录',note:'source',sets:[{id:201,weight_kg:60,reps:8,difficulty:'未记录',completed:true,note:'first'},{id:202,weight_kg:62.5,reps:6,difficulty:'未记录',completed:true,note:'second'}]}]}
