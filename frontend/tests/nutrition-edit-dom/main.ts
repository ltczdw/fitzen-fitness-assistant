import {createApp,h} from 'vue'
import Nutrition from '../../src/components/ui/NutritionEditModal.vue'
import Workout from '../../src/components/ui/WorkoutSessionManager.vue'
import {evidence,meal,session} from './state'
// Only fixture routes are returned. There is no API proxy or real network fetch.
window.fetch=async(input:any,init?:any)=>{
 const url=String(input),method=init?.method||'GET'
 evidence.requests.push({url,method,body:init?.body?JSON.parse(init.body):null})
 let data:any
 if(method==='PUT'&&url.startsWith('/api/workout-sessions/100'))data=session
 else if(method==='GET'&&url.startsWith('/api/workout-sessions?'))data=[session]
 else if(method==='GET'&&url.startsWith('/api/workout-exercises?'))data={exercises:[]}
 else throw new Error('Fixture forbids unsupported fetch '+url)
 return new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}})
}
createApp({setup(){return()=>h('main',[
 h('h1','第二轮：实际编辑组件，合成数据'),
 h('button',{onClick:()=>window.dispatchEvent(new CustomEvent('fitness-edit-nutrition-record',{detail:meal}))},'打开饮食'),
 h('button',{onClick:()=>window.dispatchEvent(new CustomEvent('fitness-edit-workout-session',{detail:session}))},'打开训练'),
 h('button',{onClick:()=>{evidence.requests.splice(0);evidence.notifications.splice(0)}},'清除证据'),
 h('pre',{id:'evidence'},JSON.stringify(evidence,null,2)),h(Nutrition),h(Workout)])}}).mount('#app')
