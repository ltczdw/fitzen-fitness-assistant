import test from 'node:test'
import assert from 'node:assert/strict'
import {captureNutritionEdit} from './nutrition-edit-harness.mjs'
import {readFileSync} from 'node:fs'
import ts from 'typescript'
for (const mode of ['init','lookup_unchanged','same_name','rename','add_local_alias','add_custom','add_ai']) {
  test('round2 actual Vue script intent: '+mode,async()=>{
    const {body}=await captureNutritionEdit(mode)
    assert.equal(body.items[0].id,1)
    if (mode==='same_name' || mode==='rename') assert.equal(body.items[0].food_name,mode==='rename'?'白米饭':'米饭')
    else assert.equal(Object.hasOwn(body.items[0],'food_name'),false)
    if (mode==='add_local_alias') assert.equal(body.items[1].food_name,'白饭')
    if (mode==='add_custom' || mode==='add_ai') {
      assert.equal(body.items[1].food_name,'合成自制食物')
      assert.equal(body.items[1].source,mode==='add_ai'?'ai':'custom')
    }
  })
}
for (const mode of ['first_set','uniform']) {
  test('actual frontend metrics respect '+mode,async()=>{
    const source=readFileSync(new URL('../src/fitzen/metrics.ts',import.meta.url),'utf8')
    const js=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ES2022}}).outputText
      .replace(/import \{\s*deleteConfirmation\s*\} from '[^']+';?/g, 'const deleteConfirmation={clear(){}};')
      .replace(/import \{[^}]+\} from 'vue';?/,'const computed=fn=>({get value(){return fn()}});const ref=value=>({value});')
      .replace(/import \{[^}]+\} from '\.\/state';?/,'const ui={};const api=()=>{throw Error("unexpected API")};const notify=()=>{};const refresh=()=>{};')
    const metrics=await import('data:text/javascript;base64,'+Buffer.from(js).toString('base64'))
    const row={sets:2,weight_kg:60,reps:8,summary_mode:mode,...(mode==='first_set'?{set_details:[{weight_kg:60,reps:8},{weight_kg:62.5,reps:6}]}:{})}
    const details=metrics.setsFor(row)
    assert.equal(details.length,2)
    assert.equal(details.reduce((n,s)=>n+s.weight_kg*s.reps,0),mode==='first_set'?855:960)
  })
}
