import test from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import vm from 'node:vm'
import ts from 'typescript'
import {ref,computed,nextTick} from 'vue'
import {parse} from 'vue/compiler-sfc'

const exercises=[
 {name:'杠铃卧推',category:'胸部',target_muscles:'胸大肌',source:'local'},
 {name:'划船机',category:'有氧',target_muscles:'全身',source:'local'},
 {name:'坐姿划船',category:'背部',target_muscles:'背阔肌',source:'local'},
 {name:'跑步',category:'有氧',target_muscles:'下肢',source:'local'},
 {name:'自定义推举',category:'自定义',target_muscles:'肩部',source:'custom'},
 {name:'自定义单车',category:'有氧',target_muscles:'下肢',source:'custom'},
]
const names=value=>Array.from(value,item=>item.name)
function fixture(){
 const source=parse(readFileSync(new URL('../src/components/ui/WorkoutSessionManager.vue',import.meta.url),'utf8')).descriptor.scriptSetup.content
 const js=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText.replace(/^import .*$/gm,'')
 const requests=[],notifications=[];let library=structuredClone(exercises),failSave=false
 const context={ref,computed,nextTick,onMounted(){},onBeforeUnmount(){},watch(){},
  notify:message=>notifications.push(message),upsertExerciseMeta:()=>{},deleteConfirmation:{clear(){}},
  document:{getElementById:()=>null,body:{style:{}}},window:{dispatchEvent(){}},Event,
  setTimeout,clearTimeout,AbortSignal,console,
  fetch:async(url,init={})=>{
   const method=init.method||'GET',body=init.body?JSON.parse(init.body):null
   requests.push({url,method,body})
   if(url.startsWith('/api/workout-exercises/custom')&&method==='POST'){
    const item={name:body.exercise_name,category:body.category,target_muscles:body.target_muscles,source:'custom'}
    library.push(item);return Response.json({exercise:item})
   }
   if(url.startsWith('/api/workout-exercises?'))return Response.json({exercises:library})
   if(url.startsWith('/api/workout-sessions')&&method==='GET')return Response.json([])
   if(url.startsWith('/api/workout-sessions')&&method==='POST')return Response.json(failSave?{detail:'保存失败，请重试'}:{id:7},{status:failSave?503:200})
   throw Error('Unmocked request '+url)
  }}
 vm.runInNewContext(js+`\nglobalThis.f={cardioMode,cardioDistance,cardioDuration,addSet,validateDraft,exerciseType,exerciseLibrary,exerciseCategories,exerciseLibraryCategory,exerciseLibrarySearch,exerciseLibraryTab,customExerciseCategory,customExerciseCategories,customExerciseName,customExerciseMuscles,quickExerciseQuery,quickExerciseTerm,quickExerciseMatches,quickExerciseRecommendations,matchingExercises,recommendedExercises,customExercises,draft,firstExercise,editingId,editorOpen,saving,savingCustomExercise,notice,createNew,editSession,switchExerciseType,openExerciseLibrary,selectExercise,loadExerciseLibrary,saveCustomExercise,saveSession};`,context)
 context.f.exerciseLibrary.value=structuredClone(exercises)
 return {...context.f,requests,notifications,setFailSave(value){failSave=value}}
}

test('classification filters recommendations, search and library consistently using explicit category',()=>{
 const f=fixture()
 assert.deepEqual(names(f.quickExerciseRecommendations.value),['杠铃卧推','坐姿划船','自定义推举'])
 f.quickExerciseTerm.value='划船'
 assert.deepEqual(names(f.quickExerciseMatches.value),['坐姿划船'])
 f.exerciseLibrarySearch.value='划船'
 f.exerciseLibraryCategory.value='背部'
 f.switchExerciseType('cardio')
 assert.equal(f.exerciseLibraryCategory.value,'全部')
 assert.equal(f.quickExerciseTerm.value,'划船')
 assert.deepEqual(names(f.quickExerciseMatches.value),['划船机'])
 assert.deepEqual(names(f.recommendedExercises.value),['划船机'])
 f.exerciseLibrarySearch.value=''
 assert.deepEqual(names(f.matchingExercises.value),['划船机','跑步','自定义单车'])
 assert.deepEqual(names(f.customExercises.value),['自定义单车'])
 f.exerciseLibrarySearch.value='卧推'
 assert.equal(f.matchingExercises.value.length,0)
 f.switchExerciseType('strength')
 assert.deepEqual(names(f.recommendedExercises.value),['杠铃卧推'])
})

test('switching preserves current input and sets; save in progress prevents switching',()=>{
 const f=fixture()
 f.firstExercise.value.exercise_name='杠铃卧推'
 f.firstExercise.value.sets[0].weight_kg=30
 f.firstExercise.value.sets[0].reps=10
 const before=JSON.stringify(f.draft.value)
 f.switchExerciseType('cardio')
 assert.equal(JSON.stringify(f.draft.value),before)
 f.saving.value=true;f.switchExerciseType('strength')
 assert.equal(f.exerciseType.value,'cardio')
 f.saving.value=false;f.savingCustomExercise.value=true;f.switchExerciseType('strength')
 assert.equal(f.exerciseType.value,'cardio')
})

test('custom action inherits selected category and selection submits the existing schema; failures retain draft',async()=>{
 const f=fixture()
 f.switchExerciseType('cardio');f.openExerciseLibrary('custom')
 assert.equal(f.customExerciseCategory.value,'有氧')
 assert.deepEqual(Array.from(f.customExerciseCategories.value),['有氧'])
 await new Promise(setImmediate)
 f.customExerciseName.value='自定义有氧测试'
 await f.saveCustomExercise()
 assert.equal(f.requests.find(x=>x.method==='POST').body.category,'有氧')
 assert.equal(f.firstExercise.value.exercise_name,'自定义有氧测试')
 assert.equal(f.draft.value.title,'有氧训练')
 f.firstExercise.value.sets[0].distance_km=3.5
 f.firstExercise.value.sets[0].duration_minutes=20
 f.editorOpen.value=true;f.setFailSave(true)
 await f.saveSession()
 assert.equal(f.editorOpen.value,true)
 assert.equal(f.firstExercise.value.exercise_name,'自定义有氧测试')
 assert.match(f.notice.value,/保存失败/)
 f.setFailSave(false);await f.saveSession()
 const payload=f.requests.filter(x=>x.url==='/api/workout-sessions'&&x.method==='POST').at(-1).body
 assert.equal(payload.title,'有氧训练')
 assert.deepEqual(Object.keys(payload.exercises[0]).sort(),['difficulty','exercise_name','note','sets'])
 assert.equal(f.editorOpen.value,false)
})

test('editing cardio initializes its category without rewriting the title; a new record resets to strength',async()=>{
 const f=fixture()
 f.editSession({id:5,session_date:'2026-10-20',title:'原训练标题',note:'',exercises:[{exercise_name:'跑步',difficulty:'未记录',note:'',sets:[{weight_kg:0,reps:1,difficulty:'正常',completed:true,note:''}]}]})
 await new Promise(setImmediate)
 assert.equal(f.exerciseType.value,'cardio')
 f.selectExercise(exercises[3]);assert.equal(f.draft.value.title,'原训练标题')
 f.createNew();await new Promise(setImmediate)
 assert.equal(f.exerciseType.value,'strength')
 assert.equal(f.draft.value.title,'力量训练')
 assert.equal(f.firstExercise.value.exercise_name,'')
})


test('manually entered cardio gets its category title; browsing another category does not relabel a chosen exercise',async()=>{
 const f=fixture()
 f.switchExerciseType('cardio');f.firstExercise.value.exercise_name='新有氧动作'
 f.firstExercise.value.sets[0].distance_km=2
 f.firstExercise.value.sets[0].duration_minutes=15
 await f.saveSession()
 assert.equal(f.requests.find(x=>x.url==='/api/workout-sessions'&&x.method==='POST').body.title,'有氧训练')
 const other=fixture()
 other.selectExercise(exercises[0]);other.switchExerciseType('cardio')
 await other.saveSession()
 assert.equal(other.requests.find(x=>x.url==='/api/workout-sessions'&&x.method==='POST').body.title,'力量训练')
})

test('cardio validation, payload and interval inheritance use km/min and never encode weight/reps',async()=>{
 const f=fixture()
 f.switchExerciseType('cardio');f.selectExercise(exercises[3])
 assert.equal(f.cardioMode.value,true)
 assert.match(f.validateDraft(),/距离/)
 const set=f.firstExercise.value.sets[0]
 set.distance_km=5.25;set.duration_minutes=31.5;set.weight_kg=60;set.reps=8
 f.addSet(f.firstExercise.value)
 assert.equal(f.firstExercise.value.sets[1].duration_minutes,31.5)
 assert.equal(f.cardioDistance.value,10.5)
 assert.equal(f.cardioDuration.value,63)
 await f.saveSession()
 const payload=f.requests.find(x=>x.url==='/api/workout-sessions'&&x.method==='POST').body
 for(const item of payload.exercises[0].sets){
  assert.equal(item.distance_km,5.25);assert.equal(item.duration_minutes,31.5)
  assert.equal(item.weight_kg,0);assert.equal(item.reps,0)
 }
})

test('saved cardio metrics round-trip without needing an action-library match',async()=>{
 const f=fixture()
 f.editSession({id:9,session_date:'2026-10-21',title:'有氧测试',note:'',exercises:[{id:5,exercise_name:'库外有氧',note:'',difficulty:'未记录',sets:[{id:6,weight_kg:0,reps:0,distance_km:7.2,duration_minutes:45.5,difficulty:'正常',completed:true,note:''}]}]})
 await new Promise(setImmediate)
 assert.equal(f.exerciseType.value,'cardio')
 assert.equal(f.cardioMode.value,true)
 assert.equal(f.cardioDistance.value,7.2)
 assert.equal(f.cardioDuration.value,45.5)
 assert.equal(f.validateDraft(),'')
 f.switchExerciseType('strength')
 assert.equal(f.cardioMode.value,true)
 assert.equal(f.cardioDuration.value,45.5)
})

test('selecting strength after cardio restores weight/reps controls without submitting cardio fields',async()=>{
 const f=fixture()
 f.selectExercise(exercises[3]);const set=f.firstExercise.value.sets[0]
 set.distance_km=2;set.duration_minutes=15
 f.selectExercise(exercises[0]);set.weight_kg=40;set.reps=8
 assert.equal(f.cardioMode.value,false)
 await f.saveSession()
 const item=f.requests.find(x=>x.url==='/api/workout-sessions'&&x.method==='POST').body.exercises[0].sets[0]
 assert.equal(item.weight_kg,40);assert.equal(item.reps,8)
 assert.equal('duration_minutes' in item,false)
})


test('cardio starts at one minute, preserves entered time, inherits intervals and keeps strength defaults separate',async()=>{
 const f=fixture()
 assert.equal(f.cardioMode.value,false)
 assert.equal(f.firstExercise.value.sets[0].duration_minutes,undefined)
 f.switchExerciseType('cardio')
 assert.equal(f.firstExercise.value.sets[0].duration_minutes,1)
 assert.equal(f.cardioDuration.value,1)
 f.switchExerciseType('strength');assert.equal(f.cardioMode.value,false)
 f.switchExerciseType('cardio');assert.equal(f.cardioMode.value,true)
 f.selectExercise(exercises[3])
 const set=f.firstExercise.value.sets[0]
 assert.equal(set.duration_minutes,1)
 set.distance_km=0.25
 await f.saveSession()
 const posted=f.requests.find(x=>x.url==='/api/workout-sessions'&&x.method==='POST').body.exercises[0].sets[0]
 assert.equal(posted.duration_minutes,1)
 set.duration_minutes=12.5;f.selectExercise(exercises[1])
 assert.equal(set.duration_minutes,12.5)
 f.addSet(f.firstExercise.value)
 assert.equal(f.firstExercise.value.sets[1].duration_minutes,12.5)
 const direct=fixture();direct.selectExercise(exercises[3])
 assert.equal(direct.firstExercise.value.sets[0].duration_minutes,1)
})
