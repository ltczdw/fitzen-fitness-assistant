import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFile} from 'node:fs/promises'
import {colorIssues, classIssues, inventoryIssues, componentImportIssue, inspectSource, checkDesignSystem} from '../scripts/check-design-system.mjs'
import {approvedVueFiles} from '../scripts/design-system-policy.mjs'
import {readTokens, renderTokenList, exportTokens} from '../scripts/export-tokens.mjs'

const css=await readFile(new URL('../src/fitzen/tokens.css',import.meta.url),'utf8')
const tokens=readTokens(css)

test('拒绝颜色字面值，接受语义 Token / 透明度 / 继承关键字',()=>{
  for (const value of ['#fff','#ffffffff','rgb(255 255 255)','rgba(0,0,0,.3)','oklch(50% .2 100)','white','rebeccapurple','var(--fz-primary, #175c43)']) assert.ok(colorIssues(value,{tokens}).length,value)
  for (const value of ['var(--fz-primary)','rgb(var(--fz-rgb-primary) / .12)','transparent','currentColor','none']) assert.deepEqual(colorIssues(value,{tokens}),[],value)
  assert.ok(colorIssues('var(--made-up-color)',{tokens}).length)
})
test('拒绝默认调色板和任意色值类，接受排版 / 布局 / 语义颜色',()=>{
  for (const value of ['bg-white','hover:text-gray-500','md:hover:border-green-200','bg-[#fff]','bg-[color:#ffffff]','fill-[rgb(0,0,0)]','bg-made-up']) assert.ok(classIssues(value).length,value)
  for (const value of ['bg-surface-card text-on-surface','text-[16px] border-0','hover:bg-primary/20','bg-transparent stroke-current','bg-gradient-to-r from-primary to-primary-container']) assert.deepEqual(classIssues(value),[],value)
})
test('冻结组件清单，并限制 Vue 组件导入来源',()=>{
  assert.deepEqual(inventoryIssues(approvedVueFiles),[])
  assert.ok(inventoryIssues([...approvedVueFiles,'src/components/ui/NewButton.vue']).length)
  assert.ok(inventoryIssues([...approvedVueFiles,'src/screens/Other.tsx']).length)
  assert.ok(inventoryIssues(approvedVueFiles.slice(1)).length)
  assert.equal(componentImportIssue('src/fitzen/App.vue','src/fitzen/screens/Chat.vue'),null)
  assert.ok(componentImportIssue('src/fitzen/screens/Training.vue','src/fitzen/screens/Chat.vue'))
  assert.equal(componentImportIssue('src/fitzen/screens/Stats.vue','src/components/ui/EnergyBalanceChart.vue'),null)
})
test('实际 SFC 模板、脚本、样式都参与扫描；选择器不会误判为 HEX',()=>{
  assert.ok(inspectSource('<template><div style="color:#fff"/></template>','src/fitzen/App.vue',tokens).length)
  assert.ok(inspectSource('<template><svg><path fill="white"/></svg></template>','src/fitzen/App.vue',tokens).length)
  assert.ok(inspectSource('<template><div :style="{color:\'blue\'}"/></template>','src/fitzen/App.vue',tokens).length)
  assert.ok(inspectSource('<script setup>const color="#fff"</script>','src/fitzen/App.vue',tokens).length)
  assert.ok(inspectSource('<script setup>const styles={backgroundColor:"white"}</script>','src/fitzen/App.vue',tokens).length)
  assert.ok(inspectSource('export {default as Bad} from "../../fitzen/screens/Chat.vue"','src/components/ui/index.ts',tokens).length)
  assert.ok(inspectSource('<script setup>defineComponent({})</script>','src/fitzen/App.vue',tokens).length)
  assert.ok(inspectSource('<script setup>import Bad from "../fitzen/screens/Chat.vue"</script>','src/components/ui/FoodModal.vue',tokens).length)
  assert.deepEqual(inspectSource('<script setup>document.querySelector("#add-food-modal-overlay")</script><template><button class="fz-button text-on-surface"/></template><style>#abc{color:var(--fz-primary)}</style>','src/fitzen/App.vue',tokens),[])
})
test('Token 导出确定性、覆盖全部变量，并验证当前仓库与文档',async()=>{
  const output=renderTokenList(css)
  assert.equal(output,renderTokenList(css))
  for (const name of tokens.keys()) assert.ok(output.includes(name),name)
  await exportTokens({check:true})
  assert.equal((await checkDesignSystem()).components,8)
})

test('删除入口必须复用登记组件，不能重新手写确认按钮',()=>{
  for (const template of ['<button><span>delete</span>删除</button>','<button aria-label="删除训练">×</button>','<button>清空全部</button>']) {
    assert.ok(inspectSource(`<template>${template}</template>`,'src/fitzen/screens/Training.vue',tokens).some(issue=>issue.includes('DeleteButton')))
  }
  assert.deepEqual(inspectSource('<template><DeleteButton :action="remove" subject="记录" compact /></template>','src/fitzen/screens/Training.vue',tokens),[])
})
