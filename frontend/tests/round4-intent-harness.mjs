import {readFileSync} from 'node:fs'
import {pathToFileURL} from 'node:url'
import ts from 'typescript'

// Current production script with deterministic deferred API responses. This
// layer exercises version/readiness/payload logic, separate from actual DOM.
export async function captureRound4Intent(source, fault) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'document')
  globalThis.document = {activeElement:null,body:{style:{overflow:''}}}
  try {
    const code = readFileSync(new URL('../src/components/ui/NutritionEditModal.vue',import.meta.url),'utf8').match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1]
    const js = ts.transpileModule(code,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ES2022}}).outputText
      .replace(/import DeleteButton from '[^']+';?/g, '')
      .replace(/import \{\s*deleteConfirmation\s*\} from '[^']+';?/g, 'const deleteConfirmation={clear(){}};')
      .replace(/import \{[^}]*\} from 'vue';?/, 'const ref=value=>({value});const computed=fn=>({get value(){return fn()}});const nextTick=async()=>{};const onMounted=()=>{};const onBeforeUnmount=()=>{};')
      .replace(/import \{[^}]*\} from '\.\.\/\.\.\/fitzen\/state';?/, `const calls=[],lookups=[];let held=null,defer=false;
        const result=name=>name==='鸡蛋'?{food_name:'鸡蛋',per_100g:{kcal:144,protein_g:12.6,carb_g:1.1,fat_g:9.5,fiber_g:0}}:{food_name:'熟大米',per_100g:{kcal:116,protein_g:2.6,carb_g:25.9,fat_g:.3,fiber_g:.3}};
        const api=async(url,init)=>{if(url.includes('/lookup?')){const name=new URL(url,'http://fixture.test').searchParams.get('name');lookups.push(name);if(defer){defer=false;return new Promise(resolve=>held=()=>resolve(result(name)))}return result(name)}calls.push({url,method:init?.method,body:JSON.parse(init.body)});return {}};
        const customFoods=[],foodsCatalog=[];const ensureCustomFoods=async()=>{};const ensureFoodCatalog=async()=>{};const notify=()=>{};const openRecords=()=>{};const refresh=async()=>{};const saveCustomFood=async()=>{};const ui={targets:{},summary:{}};
        const holdNext=()=>{defer=true};const release=()=>held();`)
    const m = await import('data:text/javascript;base64,'+Buffer.from(js+'\nexport {open,save,foods,ready,changeName,lookup,appendFood,addLocalByName,addCatalogFood,calls,lookups,holdNext,release};').toString('base64')+'#'+crypto.randomUUID())
    await m.open({detail:{id:'round4-synthetic-meal',source:'ai_food_log',date:'2026-10-03T03:00:00Z',meal_type:'午餐',items:[{id:1,raw_food_name:'米饭',food_name:'熟大米',weight_g:100,calories:116,protein_g:2.6,carbs_g:25.9,fat_g:.3,source:'local'}]}})
    if(source==='local')await m.addLocalByName('白饭')
    else if(source==='catalog')m.addCatalogFood({name:'熟大米',kcal:116,protein:2.6,carbs:25.9,fat:.3,fiber:.3})
    else if(source!=='existing')m.appendFood({name:'合成自制食物',grams:100,kcal:200,protein:5,carbs:30,fat:6,fiber:1},source)
    const food=m.foods.value.at(-1),stages=[]
    const snapshot=()=>({name:food.name,requestedName:food.requestedName,basis:food.basis,pending:food.pending,ready:m.ready.value})
    const edit=name=>{food.name=name;m.changeName(food)}
    if(fault==='twice'){
      edit('鸡蛋');await m.lookup(food);stages.push(snapshot())
      edit('白饭');await m.lookup(food);stages.push(snapshot())
    }else{
      m.holdNext();edit('白饭');const old=m.lookup(food)
      edit('鸡蛋')
      if(fault==='stale')await m.lookup(food)
      stages.push(snapshot());m.release();await old;stages.push(snapshot())
      if(fault==='pending'){
        await m.save();stages.push({blockedSaveCalls:m.calls.length,ready:m.ready.value})
        await m.lookup(food);stages.push(snapshot())
      }
    }
    await m.save()
    return {source,fault,stages,lookups:m.lookups,calls:m.calls}
  }finally{if(previous)Object.defineProperty(globalThis,'document',previous);else delete globalThis.document}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)console.log(JSON.stringify(await captureRound4Intent(process.argv[2],process.argv[3])))
