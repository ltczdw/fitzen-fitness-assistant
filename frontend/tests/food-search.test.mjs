import test from 'node:test'
import assert from 'node:assert/strict'
import {isolatedState} from './isolated-state.mjs'

test('空搜索不发送请求，也不改变餐次草稿',async t=>{
  const state=await isolatedState(t)
  assert.deepEqual(await state.findFoodSuggestions('  '),[])
  assert.deepEqual(state.ui.foods,[])
})

test('相关食物搜索只读合并两个库，保留同名来源，完整匹配优先且过滤无效营养',async t=>{
  const state=await isolatedState(t),requests=[],controller=new AbortController()
  const food={name:'牛肉',category:'肉类',preparation_state:'熟',kcal:194,protein:26.7,carbs:0,fat:9.4,fiber:0}
  globalThis.fetch=async(url,init)=>{
    const parsed=new URL(url,'http://fixture.test');requests.push({parsed,init})
    return new Response(JSON.stringify({foods:parsed.pathname.endsWith('/custom-foods')?[{...food,id:5,kcal:250}]:[{...food,name:'炖牛肉'},food,{...food,name:'无效食物',kcal:-1}]}))
  }
  const results=await state.findFoodSuggestions(' 牛肉 ',controller.signal)
  assert.equal(requests.length,2)
  for(const {parsed,init} of requests){
    assert.equal(parsed.searchParams.get('search'),'牛肉')
    assert.equal(parsed.searchParams.get('user_id'),'demo-user')
    assert.equal(init.method,undefined)
    assert.equal(init.signal,requests[0].init.signal)
  }
  controller.abort();assert.equal(requests[0].init.signal.aborted,true,'caller cancellation reaches the shared request signal')
  assert.deepEqual(results.map(food=>[food.name,food.source,food.kcal]),[['牛肉','local',194],['牛肉','custom',250],['炖牛肉','local',194]])
  assert.deepEqual(state.ui.foods,[])
  assert.deepEqual(state.customFoods,[],'search must not replace the personal-library cache')
})

test('搜索失败会向界面报告错误，不回退到单项查找或添加食物',async t=>{
  const state=await isolatedState(t),requests=[]
  globalThis.fetch=async(url)=>{requests.push(url);return url.includes('/custom-foods')?new Response(JSON.stringify({detail:'模拟搜索失败'}),{status:503}):new Response(JSON.stringify({foods:[]}))}
  await assert.rejects(state.findFoodSuggestions('牛肉'),/模拟搜索失败/)
  assert(requests.every(url=>!url.includes('/lookup')))
  assert.deepEqual(state.ui.foods,[])
})

test('取消过期搜索会传播中止状态且不改变草稿',async t=>{
  const state=await isolatedState(t),controller=new AbortController();controller.abort()
  globalThis.fetch=async(url,init)=>{assert.equal(init.signal.aborted,true);throw new DOMException('Aborted','AbortError')}
  await assert.rejects(state.findFoodSuggestions('牛肉',controller.signal),{name:'AbortError'})
  assert.deepEqual(state.ui.foods,[])
})

test('搜索保留12秒超时限制，调用方取消信号不能覆盖超时',async t=>{
  const state=await isolatedState(t),deadline=new AbortController(),caller=new AbortController()
  t.mock.method(AbortSignal,'timeout',milliseconds=>{assert.equal(milliseconds,12000);return deadline.signal})
  globalThis.fetch=async(url,init)=>new Promise((resolve,reject)=>init.signal.addEventListener('abort',()=>reject(init.signal.reason),{once:true}))
  const pending=state.findFoodSuggestions('牛肉',caller.signal)
  deadline.abort(new DOMException('Search timed out','TimeoutError'))
  await assert.rejects(pending,{name:'TimeoutError'})
  assert.equal(caller.signal.aborted,false)
  assert.deepEqual(state.ui.foods,[])
})
