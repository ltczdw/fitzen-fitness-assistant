import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import ts from 'typescript'
import {loadDeleteController,mountDeleteButton} from './delete-button-harness.mjs'

// Real DeleteButton and controller around the production drawer action; HTTP is mocked.
let instance = 0
async function harness(t) {
  t.mock.timers.enable({apis: ['setTimeout']})
  const controller=await loadDeleteController()
  const previousDocument=globalThis.document
  globalThis.document=new EventTarget()
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'Element')
  globalThis.Element = class Element {
    constructor(inside) { this.inside = inside }
    closest() { return this.inside ? {} : null }
  }
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'Element', previous)
    else delete globalThis.Element
  })
  const source = readFileSync(new URL('../src/components/ui/RecordsDrawer.vue', import.meta.url), 'utf8')
  const script = source.match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1]
  const js = ts.transpileModule(script, {compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022}}).outputText
    .replace(/import \{[^}]+\} from 'vue';/, `
      const unmountHandlers = [];
      const ref = value => ({value});
      const computed = fn => ({get value() {return fn()}});
      const watch = () => {};
      const nextTick = async () => {};
      const onBeforeUnmount = fn => unmountHandlers.push(fn);`)
    .replace(/import \{\s*deleteConfirmation\s*\} from '[^']+';?/, `import {deleteConfirmation} from '${controller.url}';`)
    .replace(/import \{[^}]*\} from '\.\.\/\.\.\/fitzen\/state';?/, `
      const calls = [], notifications = [];
      let apiError = null;
      const api = async (url, init) => {
        calls.push({url, method: init?.method || 'GET'});
        if (apiError) throw apiError;
        return {};
      };
      const useFitzen = () => ({ui: {recordsTab: 'training', recordSearch: '', recordPage: 1, nutrition: []},
        refresh: async () => {}, openFood: () => {}, openWorkout: () => {}, exportRecords: () => {}, notify: x => notifications.push(x)});`)
    .replace(/import \{[^}]*\} from '\.\.\/\.\.\/fitzen\/metrics';?/, 'const records = {value: []};')
  const mod = await import('data:text/javascript;base64,' + Buffer.from(js + `
    // isolated instance ${++instance}
    export {remove, unmountHandlers, calls, notifications};
    export const fail = () => {apiError = new Error('fixture offline')};`).toString('base64'))
  const mounted=await mountDeleteButton(controller,{subject:'fixture record',action:()=>mod.remove(item)})
  const unmount=()=>mounted.unmount()
  t.after(()=>{unmount();globalThis.document=previousDocument})
  return {...mod,remove:()=>mounted.click(),
   confirmId:{get value(){return mounted.isArmed()?'session-42':null}},
   unmountHandlers:[unmount],
   resetDeleteOnOtherAction(event){if(controller.shouldResetDeleteConfirmation(event))controller.deleteConfirmation.clear()}}

}
const item = {id: 42, exercises: [{sets: [{}]}], date: '2026-10-03'}

test('first click sends zero requests', async t => {
  const h = await harness(t)
  await h.remove(item)
  assert.equal(h.confirmId.value, 'session-42')
  assert.deepEqual(h.calls, [])
})
test('same button twice deletes exactly once', async t => {
  const h = await harness(t)
  await h.remove(item); await h.remove(item)
  assert.deepEqual(h.calls, [{url: '/workout-sessions/42?user_id=demo-user', method: 'DELETE'}])
  assert.equal(h.confirmId.value, null)
})
test('timeout makes next click the first click again', async t => {
  const h = await harness(t)
  await h.remove(item)
  t.mock.timers.tick(2501)
  assert.equal(h.confirmId.value, null)
  await h.remove(item)
  assert.deepEqual(h.calls, [])
})
test('other action resets confirmation', async t => {
  const h = await harness(t)
  await h.remove(item)
  h.resetDeleteOnOtherAction({type: 'click', target: new Element(false)})
  await h.remove(item)
  assert.deepEqual(h.calls, [])
})
test('inside-button target retains confirmation', async t => {
  const h = await harness(t)
  await h.remove(item)
  h.resetDeleteOnOtherAction({type: 'click', target: new Element(true)})
  await h.remove(item)
  assert.equal(h.calls.length, 1)
})
test('unmount clears confirmation and timeout', async t => {
  const h = await harness(t)
  await h.remove(item)
  h.unmountHandlers.forEach(fn => fn())
  assert.equal(h.confirmId.value, null)
  t.mock.timers.tick(3000)
  assert.deepEqual(h.calls, [])
})
test('failure leaves confirmation reset and reports error', async t => {
  const h = await harness(t)
  h.fail()
  await h.remove(item); await h.remove(item)
  assert.equal(h.calls.length, 1)
  assert.equal(h.confirmId.value, null)
  assert.match(h.notifications.at(-1), /fixture offline/)
})
