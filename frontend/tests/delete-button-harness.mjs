import {readFileSync} from 'node:fs'
import ts from 'typescript'
import {parse,compileScript} from 'vue/compiler-sfc'
import {createRenderer,nextTick} from 'vue'
export const transpile=source=>ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
const moduleUrl=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64')
export async function loadDeleteController(){
 const url=moduleUrl(transpile(readFileSync(new URL('../src/components/ui/delete-confirmation.ts',import.meta.url),'utf8'))+'\n// '+crypto.randomUUID())
 return {url,...await import(url)}
}
export async function mountDeleteButton(controller,props){
 const descriptor=parse(readFileSync(new URL('../src/components/ui/DeleteButton.vue',import.meta.url),'utf8').replace('</script>','defineExpose({confirmationId:id})\n</script>')).descriptor
 let js=transpile(compileScript(descriptor,{id:'delete-integration-test',inlineTemplate:true}).content)
 js=js.replaceAll('"vue"',JSON.stringify(import.meta.resolve('vue'))).replaceAll("'vue'",JSON.stringify(import.meta.resolve('vue'))).replace("'./delete-confirmation'",JSON.stringify(controller.url))
 const component=(await import(moduleUrl(js))).default
 const node=type=>({type,props:{},children:[],parent:null})
 const renderer=createRenderer({createElement:node,createText:text=>({...node('#text'),text}),createComment:text=>({...node('#comment'),text}),
 insert(el,parent,anchor){if(el.parent)el.parent.children.splice(el.parent.children.indexOf(el),1);el.parent=parent;const i=anchor?parent.children.indexOf(anchor):-1;parent.children.splice(i<0?parent.children.length:i,0,el)},
 remove(el){el.parent?.children.splice(el.parent.children.indexOf(el),1);el.parent=null},
 setText(el,text){el.text=text},setElementText(el,text){el.text=text;el.children=[]},parentNode:el=>el.parent,nextSibling:el=>el.parent?.children[el.parent.children.indexOf(el)+1],patchProp(el,key,old,value){el.props[key]=value}})
 const root=node('root'),app=renderer.createApp(component,props);const vm=app.mount(root)
 const button=root.children[0]
 return {button,isArmed:()=>controller.deleteConfirmation.isArmed(vm.confirmationId),async click(){await button.props.onClick({stopPropagation(){}});await nextTick()},unmount:()=>app.unmount()}
}
