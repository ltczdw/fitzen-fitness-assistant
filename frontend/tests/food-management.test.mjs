import test from 'node:test'
import assert from 'node:assert/strict'
import {isolatedState} from './isolated-state.mjs'

test('library delete waits for server, preserves meals and keeps entry on failure',async t=>{
 const state=await isolatedState(t), food={id:7,name:'合成鸡蛋',source:'local'}
 state.foodsCatalog.push({...food,grams:100,kcal:144,protein:12,carbs:1,fat:9})
 state.ui.foods.push({...state.foodsCatalog[0],grams:50})
 const meal=structuredClone(state.ui.foods);let requests=[]
 globalThis.fetch=async(url,init)=>{requests.push({url,method:init.method});return new Response(JSON.stringify({detail:'模拟删除失败'}),{status:503})}
 await assert.rejects(state.deleteLibraryFood(food),/模拟删除失败/)
 assert.equal(state.foodsCatalog.length,1);assert.deepEqual(state.ui.foods,meal)
 globalThis.fetch=async(url,init)=>{requests.push({url,method:init.method});return new Response(JSON.stringify({deleted:true,id:7}))}
 await state.deleteLibraryFood(food)
 assert.equal(state.foodsCatalog.length,0);assert.deepEqual(state.ui.foods,meal)
 assert(requests.every(r=>r.method==='DELETE'&&r.url.includes('/nutrition/food/catalog/7?user_id=demo-user')))
 await assert.rejects(state.deleteLibraryFood({name:'无编号'}),/缺少编号/)
 assert.equal(requests.length,2)
})

test('custom deletion is independent of same-id built-in food and failed response never prunes',async t=>{
 const state=await isolatedState(t),food={id:5,name:'自制食物',grams:100,kcal:100,protein:5,carbs:10,fat:4,fiber:1}
 state.customFoods.push(food);state.foodsCatalog.push({...food,name:'内置食物'})
 globalThis.fetch=async()=>new Response('{}')
 await assert.rejects(state.deleteLibraryFood({...food,source:'custom'}),/删除结果无效/)
 assert.equal(state.customFoods.length,1)
 globalThis.fetch=async(url)=>{assert(url.includes('/nutrition/custom-foods/5?user_id=demo-user'));return new Response('{"deleted":true}')}
 await state.deleteLibraryFood({...food,source:'custom'})
 assert.equal(state.customFoods.length,0);assert.equal(state.foodsCatalog.length,1)
})

test('new food appears immediately even if catalog reload is stale; malformed save does not succeed',async t=>{
 const state=await isolatedState(t),food={id:8,name:'合成沙拉',grams:100,kcal:130,protein:9,carbs:8,fat:6,fiber:2}
 const writes=[]
 globalThis.fetch=async(url,init)=>{if(init.method==='POST'){writes.push(JSON.parse(init.body));return new Response(JSON.stringify({saved:true,food}))}return new Response('{"foods":[]}')}
 assert.equal((await state.saveCustomFood(food)).id,8)
 assert.equal(state.customFoods[0].name,food.name);assert.equal(state.ui.foods.length,0)
 assert.equal(writes.length,1);assert.equal(writes[0].nutrition_per_100g.kcal,130)
 globalThis.fetch=async()=>new Response('{"foods":[],"saved":true}')
 await assert.rejects(state.saveCustomFood(food),/保存结果无效/)
})


test('editing updates the exact source/id and never rewrites selected meal snapshots',async t=>{
 const state=await isolatedState(t),original={id:4,name:'原食物',grams:100,kcal:100,protein:5,carbs:10,fat:4,fiber:1},saved={...original,name:'新食物',kcal:160}
 state.foodsCatalog.push({...original});state.customFoods.push({...original});state.ui.foods.push({...original,grams:80});const meal=structuredClone(state.ui.foods)
 globalThis.fetch=async(url,init)=>{assert.equal(init.method,'PUT');assert(url.includes('/nutrition/custom-foods/4'));assert.equal(JSON.parse(init.body).food_name,'新食物');return new Response(JSON.stringify({saved:true,food:saved}))}
 await state.updateLibraryFood({...original,source:'custom'},saved)
 assert.equal(state.customFoods.length,1);assert.equal(state.customFoods[0].name,'新食物');assert.equal(state.foodsCatalog[0].name,'原食物');assert.deepEqual(state.ui.foods,meal)
 globalThis.fetch=async(url,init)=>{assert.equal(init.method,'PUT');assert(url.includes('/nutrition/food/catalog/4'));return new Response(JSON.stringify({saved:true,food:saved}))}
 await state.updateLibraryFood(original,saved)
 assert.equal(state.foodsCatalog.length,1);assert.equal(state.foodsCatalog[0].kcal,160);assert.deepEqual(state.ui.foods,meal)
})

test('failed edit, wrong returned id and missing identity preserve original food',async t=>{
 const state=await isolatedState(t),original={id:4,name:'原食物',grams:100,kcal:100,protein:5,carbs:10,fat:4,fiber:1}
 state.customFoods.push({...original});let requests=0
 globalThis.fetch=async()=>{requests++;return new Response('{"detail":"同名冲突"}',{status:409})}
 await assert.rejects(state.updateLibraryFood({...original,source:'custom'},{...original,name:'冲突名称'}),/同名冲突/)
 assert.equal(state.customFoods[0].name,'原食物')
 globalThis.fetch=async()=>{requests++;return new Response(JSON.stringify({saved:true,food:{...original,id:999,name:'其他'}}))}
 await assert.rejects(state.updateLibraryFood({...original,source:'custom'},original),/保存结果无效/)
 assert.equal(state.customFoods[0].name,'原食物')
 await assert.rejects(state.updateLibraryFood({name:'无编号'},original),/缺少编号/)
 assert.equal(requests,2)
})


test('opening a meal initializes local time immediately and first food preserves manual time',async t=>{
 const state=await isolatedState(t)
 state.openFood('2026-10-12')
 assert.equal(state.ui.foodOpen,true);assert.equal(state.ui.foods.length,0)
 assert.match(state.ui.mealTime,/^2026-10-12T\d{2}:\d{2}$/)
 state.ui.mealTime='2026-10-12T08:05'
 state.addNutritionFood({name:'合成食物',grams:100,kcal:100,protein:5,carbs:10,fat:4,source:'custom'})
 assert.equal(state.ui.mealTime,'2026-10-12T08:05')
 state.ui.foodOpen=false;state.openFood()
 assert.equal(state.ui.mealTime,'2026-10-12T08:05','reopening a populated draft keeps its time')
})

test('a fresh empty opening captures current minute while an already-open meal keeps edited time',async t=>{
 const state=await isolatedState(t)
 state.ui.mealTime='2000-01-01T01:01';state.openFood()
 const now=new Date(),local=new Date(now.getTime()-now.getTimezoneOffset()*60000).toISOString().slice(0,16)
 assert.equal(state.ui.mealTime,local)
 state.ui.mealTime='2026-10-18T11:21';state.openFood()
 assert.equal(state.ui.mealTime,'2026-10-18T11:21')
})

test('meal submission uses the chosen opening time and resets it only after successful save',async t=>{
 const state=await isolatedState(t),writes=[]
 globalThis.fetch=async(url,init)=>{if(init.method==='POST'){writes.push(JSON.parse(init.body));return new Response('{"saved":true}')}return new Response('{"foods":[],"sessions":[],"nutrition":[],"daily":[],"today":{}}')}
 state.openFood('2026-10-12');state.ui.mealTime='2026-10-12T09:17'
 state.addNutritionFood({name:'合成食物',grams:80,kcal:100,protein:5,carbs:10,fat:4,source:'custom'})
 await state.saveFood()
 assert.equal(writes.length,1);assert.equal(writes[0].record_time,new Date('2026-10-12T09:17').toISOString())
 assert.equal(state.ui.mealTime,'');assert.equal(state.ui.foodOpen,false)
 state.openFood();assert.notEqual(state.ui.mealTime,'')
})
