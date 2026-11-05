import test from 'node:test'
import assert from 'node:assert/strict'
import {captureNutritionEdit} from './nutrition-edit-harness.mjs'

for (const source of ['local','catalog','custom','ai']) {
  test(`R013 new ${source} then explicit supported rename submits last intent`, async () => {
    const {body} = await captureNutritionEdit(`add_${source}_then_rename`)
    console.log('R3 component actual payload', source, JSON.stringify(body))
    assert.equal(body.items.at(-1).food_name, '鸡蛋')
    assert.equal(body.items.at(-1).source, 'local')
    assert.equal(body.items[0].id, 1)
    assert.equal(Object.hasOwn(body.items[0], 'food_name'), false)
  })
}
test('R013 added alias weight-only keeps initial intent', async () => {
  const {body} = await captureNutritionEdit('add_local_alias_weight')
  assert.equal(body.items.at(-1).food_name, '白饭')
  assert.equal(body.items.at(-1).weight_g, 150)
})
