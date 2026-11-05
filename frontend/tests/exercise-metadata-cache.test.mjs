import test from 'node:test'
import assert from 'node:assert/strict'
import {isolatedState} from './isolated-state.mjs'

function catalog(names) {
  return names.map((name, index) => ({
    id: index + 1, name, category: '胸部', target_muscles: '胸大肌', source: 'local',
  }))
}

test('exercise metadata loads once, updates after a write, and does not refetch per row', async t => {
  const state = await isolatedState(t)
  let calls = 0
  globalThis.fetch = async url => {
    calls += 1
    assert.match(String(url), /\/api\/workout-exercises\?/)
    return {ok: true, json: async () => ({exercises: catalog(
      Array.from({length: 300}, (_, index) => `动作${index + 1}`))})}
  }

  await state.ensureExerciseLibrary()
  assert.equal(calls, 1)
  assert.equal(state.exerciseLibrary.length, 300)

  state.upsertExerciseMeta({name: '自定义推举', category: '肩部',
    target_muscles: '三角肌前束', source: 'custom'})
  const known = state.exerciseLibrary.find(item => item.name === '自定义推举')
  assert.deepEqual(known, {name: '自定义推举', category: '肩部',
    target_muscles: '三角肌前束', source: 'custom'})
  assert.equal(state.exerciseLibrary.length, 301)

  await state.ensureExerciseLibrary()
  await state.ensureExerciseLibrary()
  assert.equal(calls, 1, 'already-loaded metadata must not refetch per row or per call')
  assert.equal(state.exerciseLibrary.find(item => item.name === '自定义推举')?.target_muscles,
    '三角肌前束')
})

test('a write during an older load wins and an unknown action stays explicit', async t => {
  const state = await isolatedState(t)
  let release
  const gate = new Promise(resolve => { release = resolve })
  globalThis.fetch = async () => {
    await gate
    return {ok: true, json: async () => ({exercises: catalog(['旧动作'])})}
  }

  const loading = state.ensureExerciseLibrary()
  state.upsertExerciseMeta({name: '新自定义动作', category: '背部',
    target_muscles: '背阔肌', source: 'custom'})
  release()
  await loading

  assert.equal(state.exerciseLibrary.find(item => item.name === '旧动作')?.target_muscles, '胸大肌')
  assert.deepEqual(state.exerciseLibrary.find(item => item.name === '新自定义动作'),
    {name: '新自定义动作', category: '背部', target_muscles: '背阔肌', source: 'custom'})
  assert.equal(state.exerciseLibrary.find(item => item.name === '未登记动作'), undefined)
})

test('a failed metadata load reports nothing fabricated and remains retryable', async t => {
  const state = await isolatedState(t)
  let calls = 0
  globalThis.fetch = async () => {
    calls += 1
    throw new Error('synthetic metadata outage')
  }
  await state.ensureExerciseLibrary()
  assert.equal(calls, 1)
  assert.deepEqual(state.exerciseLibrary, [])
  await state.ensureExerciseLibrary(true)
  assert.equal(calls, 2)
  assert.deepEqual(state.exerciseLibrary, [])
})
