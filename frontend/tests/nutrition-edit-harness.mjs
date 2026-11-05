import {readFileSync} from 'node:fs'
import ts from 'typescript'
import {pathToFileURL} from 'node:url'

// Execute the current component's script. Vue lifecycle/DOM/state are stubs,
// so this captures actual save payloads but does not claim DOM coverage.
export async function captureNutritionEdit(mode = 'weight') {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'document')
  globalThis.document = {activeElement: null, body: {style: {overflow: ''}}}
  try {
    const source = readFileSync(new URL('../src/components/ui/NutritionEditModal.vue', import.meta.url), 'utf8')
    const script = source.match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1]
    const js = ts.transpileModule(script, {compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022}}).outputText
      .replace(/import DeleteButton from '[^']+';?/g, '')
      .replace(/import \{\s*deleteConfirmation\s*\} from '[^']+';?/g, 'const deleteConfirmation={clear(){}};')
      .replace(/import \{[^}]*\} from 'vue';?/, 'const ref = value => ({value}); const computed = fn => ({get value(){return fn()}}); const nextTick = async () => {}; const onBeforeUnmount = () => {}; const onMounted = () => {};')
      .replace(/import \{[^}]*\} from '\.\.\/\.\.\/fitzen\/state';?/, 'const calls=[]; const api=async(url,init)=>{if(url.includes("/lookup?")){const name=new URL(url,"http://fixture.test").searchParams.get("name");return name==="鸡蛋"?{food_name:"鸡蛋",per_100g:{kcal:144,protein_g:12.6,carb_g:1.1,carbs_g:1.1,fat_g:9.5,fiber_g:0}}:{food_name:"熟大米",per_100g:{kcal:116,protein_g:2.6,carb_g:25.9,carbs_g:25.9,fat_g:.3,fiber_g:.3}}};calls.push({url,method:init?.method,body:JSON.parse(init.body)});return {}}; const customFoods=[], foodsCatalog=[]; const ensureCustomFoods=async()=>{}; const ensureFoodCatalog=async()=>{}; const notify=()=>{}; const openRecords=()=>{}; const refresh=async()=>{}; const saveCustomFood=async()=>{}; const ui={targets:{},summary:{}};')
    const module = await import('data:text/javascript;base64,' + Buffer.from(js + '\nexport {open,save,foods,meal,date,calls,lookup,changeName,addLocalByName,appendFood,addCatalogFood};').toString('base64') + '#' + crypto.randomUUID())
    await module.open({detail: {id: 'synthetic-raw-meal', source:'ai_food_log', date:'2026-10-03T03:00:00Z', meal_type:'午餐',
      items:[{id:1, raw_food_name:'米饭', food_name:'熟大米', weight_g:100, calories:116, protein_g:2.6, carbs_g:25.9, fat_g:0.3, source:'local'}]}})
    if (mode === 'weight') module.foods.value[0].grams = 150
    if (mode === 'meal') module.meal.value = '晚餐'
    if (mode === 'date') module.date.value = '2026-10-04T11:00'
    if (mode === 'rename' || mode === 'same_name') {
      module.foods.value[0].name = mode === 'rename' ? '白米饭' : '米饭'
      module.changeName(module.foods.value[0]); await module.lookup(module.foods.value[0])
    }
    if (mode === 'lookup_unchanged') await module.lookup(module.foods.value[0])
    if (mode === 'add_local_alias') await module.addLocalByName('白饭')
    if (mode === 'add_custom' || mode === 'add_ai') module.appendFood({name:'合成自制食物',grams:80,kcal:200,protein:5,carbs:30,fat:6,fiber:1},mode === 'add_ai' ? 'ai' : 'custom')
    if (mode === 'add_catalog_custom') module.addCatalogFood({id:7,name:'熟大米',grams:100,kcal:240,protein:8,carbs:45,fat:3,fiber:2,category:'自定义',preparation_state:'用户自定义',source:'custom'})
    if (mode === 'add_local_alias_weight') {
      await module.addLocalByName('白饭'); module.foods.value.at(-1).grams = 150
    }
    if (mode.startsWith('add_') && mode.endsWith('_then_rename')) {
      if (mode === 'add_local_then_rename') await module.addLocalByName('白饭')
      else if (mode === 'add_catalog_then_rename') module.addCatalogFood({name:'熟大米',kcal:116,protein:2.6,carbs:25.9,fat:.3,fiber:.3})
      else module.appendFood({name:'合成自制食物',grams:80,kcal:200,protein:5,carbs:30,fat:6,fiber:1},mode === 'add_ai_then_rename' ? 'ai' : 'custom')
      const added = module.foods.value.at(-1)
      added.name = '鸡蛋'; module.changeName(added); await module.lookup(added)
    }
    await module.save()
    if (module.calls.length !== 1) throw new Error('fixture expected exactly one save call')
    return module.calls[0]
  } finally {
    if (previous) Object.defineProperty(globalThis, 'document', previous)
    else delete globalThis.document
  }
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  console.log(JSON.stringify(await captureNutritionEdit(process.argv[2] || 'weight')))
