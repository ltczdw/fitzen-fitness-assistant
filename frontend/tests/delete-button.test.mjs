import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFile} from 'node:fs/promises'
import ts from 'typescript'
import {parse, compileScript} from 'vue/compiler-sfc'
import {createRenderer, nextTick} from 'vue'

const moduleUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const transpile = source => ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
const controllerUrl = moduleUrl(transpile(await readFile(new URL('../src/components/ui/delete-confirmation.ts',import.meta.url),'utf8')))
const {createDeleteConfirmation, shouldResetDeleteConfirmation, DELETE_CONFIRMATION_DELAY, deleteConfirmation} = await import(controllerUrl)

test('只有同一入口第二次点击执行，其他入口会替换确认，超时 / 取消会复位',()=>{
  const callbacks=new Map(); let sequence=0, updates=0
  const controller=createDeleteConfirmation((callback,delay)=>{assert.equal(delay,2500);callbacks.set(++sequence,callback);return sequence},id=>callbacks.delete(id))
  const a=Symbol(), b=Symbol(), unsubscribe=controller.subscribe(()=>updates++)
  assert.equal(controller.activate(a),false)
  assert.equal(controller.isArmed(a),true)
  assert.equal(controller.activate(b),false)
  assert.equal(controller.isArmed(a),false)
  controller.clear(a)
  assert.equal(controller.isArmed(b),true)
  assert.equal(callbacks.size,1)
  callbacks.values().next().value()
  assert.equal(controller.isArmed(b),false)
  assert.equal(controller.activate(a),false)
  assert.equal(controller.activate(a),true)
  assert.equal(controller.isArmed(a),false)
  assert.equal(callbacks.size,0)
  unsubscribe(); const previous=updates
  controller.activate(b);controller.clear()
  assert.equal(updates,previous)
  assert.equal(DELETE_CONFIRMATION_DELAY,2500)
})

test('支持键盘二次确认，点击别处 / 编辑输入 / Tab 移焦会取消',()=>{
  const target={closest:()=>({})}, outside={closest:()=>null}
  for (const event of [{type:'click',target},{type:'keydown',key:'Enter',target},{type:'keydown',key:' ',target}]) assert.equal(shouldResetDeleteConfirmation(event),false)
  for (const event of [{type:'click',target:outside},{type:'input',target},{type:'keydown',key:'Tab',target},{type:'keydown',key:'Escape',target}]) assert.equal(shouldResetDeleteConfirmation(event),true)
})

// Mount the real SFC on Vue's in-memory renderer: no browser / API / user data needed.
const sfc=parse(await readFile(new URL('../src/components/ui/DeleteButton.vue',import.meta.url),'utf8')).descriptor
const compiled=compileScript(sfc,{id:'delete-button-test',inlineTemplate:true}).content
const componentUrl=moduleUrl(transpile(compiled).replaceAll('"vue"',JSON.stringify(import.meta.resolve('vue'))).replaceAll("'vue'",JSON.stringify(import.meta.resolve('vue'))).replace("'./delete-confirmation'",JSON.stringify(controllerUrl)))
const DeleteButton=(await import(componentUrl)).default
const makeNode=type=>({type,props:{},children:[],parent:null,text:''})
const renderer=createRenderer({
  createElement:makeNode,createText:text=>({...makeNode('#text'),text}),createComment:text=>({...makeNode('#comment'),text}),
  insert(node,parent,anchor){if(node.parent){node.parent.children.splice(node.parent.children.indexOf(node),1)}node.parent=parent;const index=anchor?parent.children.indexOf(anchor):-1;parent.children.splice(index<0?parent.children.length:index,0,node)},
  remove(node){node.parent?.children.splice(node.parent.children.indexOf(node),1);node.parent=null},
  setText(node,text){node.text=text},setElementText(node,text){node.text=text;node.children=[]},
  parentNode:node=>node.parent,nextSibling:node=>node.parent?.children[node.parent.children.indexOf(node)+1],
  patchProp(node,key,previous,value){node.props[key]=value},
})
function mount(props){
  const root=makeNode('root'), app=renderer.createApp(DeleteButton,props)
  app.mount(root)
  return {button:root.children[0],unmount:()=>app.unmount()}
}
const click=button=>button.props.onClick({stopPropagation(){}})

test('实际按钮先确认再调用 action；Promise 执行中阻止重复操作，完成后恢复',async()=>{
  const previousDocument=globalThis.document;globalThis.document=new EventTarget()
  let calls=0, finish
  const mounted=mount({subject:'测试草稿',action:()=>{calls++;return new Promise(resolve=>{finish=resolve})}})
  try {
    await click(mounted.button);await nextTick()
    assert.equal(calls,0)
    assert.match(mounted.button.props['aria-label'],/再次点击确认/)
    const executing=click(mounted.button);await nextTick()
    assert.equal(calls,1)
    assert.equal(mounted.button.props.disabled,true)
    assert.equal(mounted.button.props['aria-busy'],true)
    await click(mounted.button)
    assert.equal(calls,1)
    finish();await executing;await nextTick()
    assert.equal(mounted.button.props.disabled,false)
    assert.equal(mounted.button.props['aria-label'],'删除测试草稿')
  } finally {mounted.unmount();globalThis.document=previousDocument}
})

test('失败可重试；disabled 不执行；卸载和点击别处清除确认状态',async()=>{
  const previousDocument=globalThis.document;globalThis.document=new EventTarget()
  let calls=0, errors=[]
  const failure=Error('test failure')
  const mounted=mount({subject:'测试草稿',action:()=>{calls++;throw failure},onError:error=>errors.push(error)})
  const disabled=mount({subject:'禁用草稿',disabled:true,action:()=>calls++})
  try {
    await click(disabled.button);assert.equal(calls,0)
    await click(mounted.button);await nextTick()
    document.dispatchEvent(new Event('click'));await nextTick()
    assert.equal(mounted.button.props['aria-label'],'删除测试草稿')
    await click(mounted.button);await click(mounted.button);await nextTick()
    assert.equal(calls,1);assert.deepEqual(errors,[failure])
    assert.match(mounted.button.props['aria-label'],/失败，请重试/)
    assert.equal(mounted.button.props.disabled,false)
    await click(mounted.button);await nextTick()
    assert.match(mounted.button.props['aria-label'],/再次点击确认/)
    mounted.unmount()
    document.dispatchEvent(new Event('input'))
    deleteConfirmation.clear()
  } finally {disabled.unmount();globalThis.document=previousDocument}
})
