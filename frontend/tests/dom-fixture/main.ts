import {createApp, nextTick, h} from 'vue'
import Drawer from '../../src/components/ui/RecordsDrawer.vue'
import {ui, evidence, resetFixture} from './state'
createApp({setup() {
  async function reset() {resetFixture(); await nextTick(); ui.records = true}
  return () => h('main', [h('h1', '隔离的真实组件 DOM 测试'),
    h('button', {onClick: reset}, '重置用例'),
    h('button', {onClick: () => {ui.records = false}}, '卸载抽屉'),
    h('button', {onClick: () => {evidence.fail = !evidence.fail}}, '切换失败模式'),
    h('p', {id: 'evidence'}, `DELETE次数=${evidence.requests.length}；${evidence.requests.join(' | ')}`),
    h('p', {id: 'notifications'}, evidence.notifications.join(' | ')),
    ui.records ? h(Drawer) : null])
}}).mount('#app')
