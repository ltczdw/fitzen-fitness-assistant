import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import vm from 'node:vm'
import ts from 'typescript'
import {parse} from 'vue/compiler-sfc'
import {ref,reactive,computed,watch,nextTick,effectScope} from 'vue'
const compile=source=>ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
const moduleText=compile(readFileSync(new URL('../src/fitzen/profile-form.ts',import.meta.url),'utf8'))
const {profileForm,profilePayload}=await import('data:text/javascript;base64,'+Buffer.from(moduleText).toString('base64'))

test('profile payload normalizes optional body data and custom goals without inventing values',()=>{
 const form=profileForm({goal:'保持健康'})
 assert.equal(form.body_fat_pct,null)
 Object.assign(form,{height_cm:'175.5',body_weight_kg:'70.2',body_fat_pct:'18.5',target_weight_kg:'',available_days:'4',goal:'  提升跑步耐力  '})
 const payload=profilePayload(form)
 assert.equal(payload.height_cm,175.5);assert.equal(payload.body_fat_pct,18.5)
 assert.equal(payload.target_weight_kg,null);assert.equal(payload.goal,'提升跑步耐力')
 assert.equal(payload.available_days,4)
})

test('invalid profile values do not become zero or invalid saved metrics',()=>{
 for(const patch of [{body_fat_pct:'abc'},{body_fat_pct:0},{body_fat_pct:100},{height_cm:99},{body_weight_kg:401},{available_days:2.5},{available_days:0},{goal:' '},{display_name:' '}]){
  assert.throws(()=>profilePayload({...profileForm({}),...patch}))
 }
})

function page(t){
 const ui=reactive({profile:{display_name:'测试用户',goal:'保持健康',experience:'初学者',available_days:3,height_cm:175,body_weight_kg:70,target_weight_kg:null,body_fat_pct:18.5},loading:false,online:true,notice:''})
 let succeed=true,calls=[]
 const saveProfileData=async(payload)=>{calls.push(payload);if(!succeed){ui.notice='模拟保存失败';return false}ui.profile={...payload};return true}
 const context={ref,computed,watch,profileForm,profilePayload,useFitzen:()=>({ui,saveProfileData})}
 const source=parse(readFileSync(new URL('../src/fitzen/screens/Profile.vue',import.meta.url),'utf8')).descriptor.scriptSetup.content
 const scope=effectScope();t.after(()=>scope.stop())
 scope.run(()=>vm.runInNewContext(compile(source).replace(/^import .*$/gm,'')+'\nglobalThis.page={form,saving,error,feedback,dirty,reset,save};',context))
 return {...context.page,ui,calls,fail(){succeed=false}}
}

test('background refresh does not erase edits; save failure retains draft and reports an error',async t=>{
 const p=page(t);p.form.value.goal='提升耐力';p.form.value.body_fat_pct='19.2'
 p.ui.profile={...p.ui.profile,goal:'服务端旧目标',height_cm:176};await nextTick()
 assert.equal(p.form.value.goal,'提升耐力');assert.equal(p.form.value.body_fat_pct,'19.2')
 p.fail();await p.save();assert.equal(p.error.value,'模拟保存失败')
 assert.equal(p.form.value.goal,'提升耐力');assert.equal(p.dirty.value,true)
 p.reset();assert.equal(p.form.value.goal,'服务端旧目标');assert.equal(p.dirty.value,false)
})

test('successful save roundtrips body fat and goals and restores the clean state',async t=>{
 const p=page(t);p.form.value.body_fat_pct='20.1';p.form.value.goal='增强体能'
 await p.save();await nextTick()
 assert.equal(p.calls.length,1);assert.equal(p.calls[0].body_fat_pct,20.1)
 assert.equal(p.form.value.goal,'增强体能');assert.equal(p.dirty.value,false)
 assert.equal(p.feedback.value,'个人信息已保存')
})

test('older service cannot silently ignore and falsely confirm a body-fat save',async t=>{
 const p=page(t);delete p.ui.profile.body_fat_pct;await nextTick()
 p.form.value.body_fat_pct='18.5';await p.save()
 assert.equal(p.calls.length,0);assert.match(p.error.value,/无法保存体脂率/)
})
