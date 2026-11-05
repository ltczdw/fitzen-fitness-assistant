import {createApp,h} from 'vue'
import Nutrition from '../../src/components/ui/NutritionEditModal.vue'
import Workout from '../../src/components/ui/WorkoutSessionManager.vue'
import {evidence,meal,session,holdNext,releaseHeld} from './state'
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
 h('h1','第四轮：实际编辑组件，合成数据'),
 h('button',{onClick:()=>window.dispatchEvent(new CustomEvent('fitness-edit-nutrition-record',{detail:meal}))},'打开饮食'),
 h('button',{onClick:()=>window.dispatchEvent(new CustomEvent('fitness-edit-workout-session',{detail:session}))},'打开训练'),
 h('button',{onClick:()=>{evidence.requests.splice(0);evidence.notifications.splice(0)}},'清除证据'),
 // IAB text input sends input but does not always emit native change on blur.
 // Dispatch the DOM event through a visible fixture control; production Vue's
 // actual @change listener performs lookup and produces the actual save body.
 h('button',{style:'position:fixed;right:8px;top:8px;z-index:1000',onClick:()=>Array.from(document.querySelectorAll('input[aria-label="食物名称"]')).at(-1)?.dispatchEvent(new Event('change',{bubbles:true}))},'匹配编辑名称（夹具 change 事件）'),
 h('button',{style:'position:fixed;right:8px;top:45px;z-index:1000',onClick:holdNext},'延迟下一匹配（夹具）'),
 h('button',{style:'position:fixed;right:8px;top:80px;z-index:1000',onClick:releaseHeld},'释放旧匹配（夹具）'),
 h('pre',{id:'evidence'},JSON.stringify(evidence,null,2)),h(Nutrition),h(Workout)])}}).mount('#app')
