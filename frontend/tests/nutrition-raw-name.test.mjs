import {test} from 'node:test'
import assert from 'node:assert/strict'
import {captureNutritionEdit} from './nutrition-edit-harness.mjs'

for (const mode of ['weight','meal','date']) {
  test('R013 unchanged name survives actual component ' + mode + ' save payload', async t => {
    const call = await captureNutritionEdit(mode)
    t.diagnostic(JSON.stringify(call))
    assert.equal(call.method, 'PUT')
    const name = call.body.items[0].food_name
    assert.ok(name === undefined || name === '米饭', 'unchanged raw name must not be passively replaced by canonical name: ' + name)
  })
}
