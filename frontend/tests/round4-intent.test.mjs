import test from 'node:test'
import assert from 'node:assert/strict'
import {captureRound4Intent} from './round4-intent-harness.mjs'

for(const source of ['existing','local','catalog','custom','ai'])for(const fault of ['twice','stale','pending'])
  test(`R013 ${source}: ${fault} follows final input and current lookup`,async()=>{
    const r=await captureRound4Intent(source,fault)
    assert.equal(r.calls.length,1)
    assert.equal(r.calls[0].body.items.at(-1).food_name,fault==='twice'?'白饭':'鸡蛋')
    assert.equal(r.calls[0].body.items.at(-1).source,'local')
    if(source==='existing')assert.equal(r.calls[0].body.items[0].id,1)
    else assert.deepEqual(r.calls[0].body.items[0],{id:1,weight_g:100,source:'local'})
    if(fault==='stale'){
      assert.deepEqual(r.stages[0],r.stages[1])
      assert.equal(r.stages[1].basis.calories,144)
    }
    if(fault==='pending'){
      assert.equal(r.stages[1].basis,null)
      assert.deepEqual(r.stages[2],{blockedSaveCalls:0,ready:false})
    }
    if(fault==='twice')assert.equal(r.stages.at(-1).basis.calories,116)
  })
