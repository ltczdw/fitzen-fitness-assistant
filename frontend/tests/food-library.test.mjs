import test from 'node:test'
import assert from 'node:assert/strict'
import {captureNutritionEdit} from './nutrition-edit-harness.mjs'

test('统一食物库选择同名自定义食物时保留个人来源和营养，不能回退到内置数据',async()=>{
  const {body}=await captureNutritionEdit('add_catalog_custom')
  assert.equal(body.items[0].id,1)
  assert.equal(Object.hasOwn(body.items[0],'food_name'),false)
  assert.deepEqual(body.items[1],{
    food_name:'熟大米',weight_g:100,source:'custom',
    nutrition_per_100g:{kcal:240,protein_g:8,carbs_g:45,fat_g:3,fiber_g:2},
  })
})
