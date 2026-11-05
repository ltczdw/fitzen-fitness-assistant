import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import ts from 'typescript'

let instance = 0
async function harness(t) {
  const names = ['location', 'localStorage', 'document', 'window', 'fetch']
  const descriptors = Object.fromEntries(names.map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]))
  t.after(() => names.forEach(k => {
    if (descriptors[k]) Object.defineProperty(globalThis, k, descriptors[k]); else delete globalThis[k]
  }))
  t.mock.timers.enable({apis: ['setTimeout']})
  const store = {}, calls = []
  let conversations = [{conversation_id: 'conv-a'}, {conversation_id: 'conv-b'}, {conversation_id: 'conv-current'}]
  globalThis.location = {hash: ''}
  globalThis.localStorage = {getItem: k => store[k] ?? null, setItem: (k,v) => {store[k] = String(v)}, removeItem: k => {delete store[k]}}
  globalThis.document = {querySelector: () => null, addEventListener: () => {}, removeEventListener: () => {}}
  globalThis.window = {setTimeout}
  globalThis.fetch = async (url, init) => {
    if (init?.method === 'DELETE') {
      calls.push(String(url))
      const id = String(url).split('?')[0].split('/').pop()
      conversations = conversations.filter(c => c.conversation_id !== id)
      return new Response(JSON.stringify({deleted: true}))
    }
    return new Response(JSON.stringify(String(url).includes('/chat/conversations') ? {conversations} : []))
  }
  const source = readFileSync(new URL('../src/fitzen/state.ts', import.meta.url), 'utf8')
  const js = ts.transpileModule(source, {compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022}}).outputText
    .replace(/import \{[^}]+\} from 'vue';/, 'const reactive = x => x; const nextTick = async () => {};')
  const state = await import('data:text/javascript;base64,' + Buffer.from(js + '\n// instance ' + (++instance)).toString('base64'))
  state.ui.conversationId = 'conv-current'
  store['fitness-assistant-chat-conversation-id-v1'] = 'conv-current'
  return {...state, store, calls}
}
test('deleting another conversation preserves current state', async t => {
  const h = await harness(t)
  await h.loadConversations()
  await h.deleteConversation('conv-a')
  assert.deepEqual(h.calls, ['/api/chat/conversations/conv-a?user_id=demo-user'])
  assert.equal(h.ui.conversationId, 'conv-current')
  assert.equal(h.store['fitness-assistant-chat-conversation-id-v1'], 'conv-current')
  assert.deepEqual(h.ui.conversations.map(c => c.conversation_id), ['conv-b', 'conv-current'])
})
test('deleting current conversation rotates persisted id and clears messages', async t => {
  const h = await harness(t)
  h.ui.messages = [{role: 'user', content: 'synthetic'}]
  await h.deleteConversation('conv-current')
  assert.deepEqual(h.calls, ['/api/chat/conversations/conv-current?user_id=demo-user'])
  assert.notEqual(h.ui.conversationId, 'conv-current')
  assert.equal(h.ui.messages.length, 0)
  assert.equal(h.store['fitness-assistant-chat-conversation-id-v1'], h.ui.conversationId)
})
