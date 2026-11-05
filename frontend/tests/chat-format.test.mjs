import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFile} from 'node:fs/promises'
import ts from 'typescript'
import {parse,compileScript} from 'vue/compiler-sfc'
import {createRenderer,nextTick} from 'vue'

const moduleUrl=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64')
const transpile=source=>ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
const formatUrl=moduleUrl(transpile(await readFile(new URL('../src/components/ui/chat-format.ts',import.meta.url),'utf8')))
const {renderReply,replyPlainText}=await import(formatUrl)

test('已有回复的标题、加粗和列表排成正文，复制不带标记',()=>{
  const text='**恢复原则 **\n\n## 需要注意\n\n- 避免加重疼痛\n- 保持**温和活动**'
  const html=renderReply(text)
  assert.match(html,/<strong>恢复原则<\/strong>/)
  assert.match(html,/<h3>需要注意<\/h3>/)
  assert.match(html,/<ul><li>避免加重疼痛<\/li>/)
  assert.ok(!html.includes('**')&&!html.includes('##'))
  assert.equal(replyPlainText(text),'恢复原则\n\n需要注意\n\n避免加重疼痛\n保持温和活动')
})

test('表格保留表头、空格与所有数值，复制使用制表符',()=>{
  const text='| 动作 | 组数 | 重量 |\n|---|---:|---:|\n| 深蹲 | 3 | 20 kg |\n| 弯举 | 2 | 7.5 kg |'
  const html=renderReply(text)
  assert.match(html,/<thead><tr><th scope="col">动作/)
  assert.match(html,/<td>7\.5 kg<\/td>/)
  assert.match(html,/role="region" aria-label="回答表格" tabindex="0"/)
  assert.equal(replyPlainText(text),'动作\t组数\t重量\n深蹲\t3\t20 kg\n弯举\t2\t7.5 kg')
})

test('重量、乘号、负数、单位及普通换行不被当作装饰删掉',()=>{
  const text='卧推 20kg*10次*4组，配速 -2%，休息90–120秒。\n体重 67.5kg，蛋白质 1.6 g/kg。'
  assert.equal(replyPlainText(text),text)
  assert.ok(renderReply(text).includes('20kg*10次*4组'))
  assert.match(renderReply('3. 热身\n4. 训练'),/<ol start="3"><li>热身/)
})

test('回答中的HTML与属性注入只作为文字，危险链接无可点击地址',()=>{
  const html=renderReply('<img src=x onerror=alert(1)>\n**<script>alert(1)</script>**\n[打开](javascript:alert)\n[数据](data:text/html,x)\n[凭据](https://user:secret@example.org)')
  assert.ok(!html.includes('<img')&&!html.includes('<script')&&!html.includes('href='))
  assert.ok(html.includes('&lt;img')&&html.includes('&lt;script&gt;'))
  const link=renderReply('[官方资料](https://www.nhs.uk/a?q=1&next=%22)')
  assert.match(link,/rel="noopener noreferrer"/)
  assert.match(link,/q=1&amp;next=%22/)
  assert.equal((link.match(/<a /g)||[]).length,1)
})

test('代码按文字保留，未结束的流式片段也不会执行或丢失',()=>{
  const text='```text\n20kg*10次*4组\n<script>示例</script>'
  assert.match(renderReply(text),/<pre><code>20kg\*10次\*4组\n&lt;script&gt;/)
  assert.equal(replyPlainText(text),'20kg*10次*4组\n<script>示例</script>')
  assert.doesNotThrow(()=>renderReply('## 说明\n\n**尚在回复'))
  assert.match(renderReply('| 名称 | 备注 |\n|---|---|\n| A\\|B | 保留 |'),/<td>A\|B<\/td>/)
})

test('实际ChatMessages：助手安全排版、用户原文不改、复制可读并保留流式光标',async()=>{
  const vueUrl=JSON.stringify(import.meta.resolve('vue'))
  const stateUrl=moduleUrl(`import {reactive} from ${vueUrl};
    export const fake={ui:reactive({messages:[{role:'user',content:'**用户原文**'},{role:'assistant',content:'## 示例\\n\\n**保持活动**'}],sending:false}),
      notify(){},stopChat(){},retryChat(){},rateAnswer(){},confirmPending(){},resumeChat(){},pendingUnavailable(){return null}};
    export function useFitzen(){return fake}`)
  const {fake}=await import(stateUrl)
  const sfc=parse(await readFile(new URL('../src/components/ui/ChatMessages.vue',import.meta.url),'utf8')).descriptor
  let script=transpile(compileScript(sfc,{id:'chat-format-test',inlineTemplate:true}).content)
  script=script.replaceAll('"vue"',vueUrl).replaceAll("'vue'",vueUrl).replace("'../../fitzen/state'",JSON.stringify(stateUrl)).replace("'./chat-format'",JSON.stringify(formatUrl))
  const component=(await import(moduleUrl(script))).default
  const node=type=>({type,props:{},children:[],parent:null,text:''})
  const renderer=createRenderer({createElement:node,createText:text=>({...node('#text'),text}),createComment:text=>({...node('#comment'),text}),
    insert(el,parent,anchor){if(el.parent)el.parent.children.splice(el.parent.children.indexOf(el),1);el.parent=parent;const at=anchor?parent.children.indexOf(anchor):-1;parent.children.splice(at<0?parent.children.length:at,0,el)},
    remove(el){el.parent?.children.splice(el.parent.children.indexOf(el),1)},setText(el,text){el.text=text},setElementText(el,text){el.text=text;el.children=[]},
    parentNode:el=>el.parent,nextSibling:el=>el.parent?.children[el.parent.children.indexOf(el)+1],patchProp(el,key,old,value){el.props[key]=value}})
  const root=node('root'),app=renderer.createApp(component);app.mount(root)
  const all=()=>{const result=[];const visit=el=>{result.push(el);el.children.forEach(visit)};visit(root);return result}
  const previous=Object.getOwnPropertyDescriptor(globalThis,'navigator');let copied=''
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{clipboard:{writeText:async text=>{copied=text}}}})
  try{
    assert.ok(all().some(el=>el.text==='**用户原文**'))
    assert.ok(all().some(el=>el.props.innerHTML==='<h3>示例</h3><p><strong>保持活动</strong></p>'))
    await all().find(el=>el.props['aria-label']==='复制回答').props.onClick({stopPropagation(){}})
    assert.equal(copied,'示例\n\n保持活动')
    fake.ui.sending=true;fake.ui.messages[1].content='**新片段**';await nextTick()
    assert.ok(all().some(el=>String(el.props.class).includes('chat-stream-cursor')))
    assert.ok(all().some(el=>el.props.innerHTML==='<p><strong>新片段</strong></p>'))
  }finally{app.unmount();if(previous)Object.defineProperty(globalThis,'navigator',previous);else delete globalThis.navigator}
})
