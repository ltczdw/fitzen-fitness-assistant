import {readFileSync} from 'node:fs'
import ts from 'typescript'

let instance = 0
export async function isolatedState(t) {
  const names = ['location', 'localStorage', 'document', 'window', 'fetch']
  const descriptors = Object.fromEntries(names.map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]))
  t.after(() => names.forEach(k => {
    if (descriptors[k]) Object.defineProperty(globalThis, k, descriptors[k]); else delete globalThis[k]
  }))
  t.mock.timers.enable({apis: ['setTimeout']})
  globalThis.location = {hash: ''}
  globalThis.localStorage = {getItem: () => null, setItem: () => {}, removeItem: () => {}}
  globalThis.document = {querySelector: () => null, addEventListener: () => {}, removeEventListener: () => {}}
  // UI animation pauses run as microtasks; notice timers stay under the test clock.
  globalThis.window = {setTimeout: fn => {queueMicrotask(fn); return 0}}
  globalThis.fetch = async () => {throw new Error('unexpected fixture request')}
  const source = readFileSync(new URL('../src/fitzen/state.ts', import.meta.url), 'utf8')
  const js = ts.transpileModule(source, {compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022}}).outputText
    .replace(/import \{[^}]+\} from 'vue';/, 'const reactive = x => x; const nextTick = async () => {};')
  return import('data:text/javascript;base64,' + Buffer.from(js + '\n// isolated instance ' + (++instance)).toString('base64'))
}
