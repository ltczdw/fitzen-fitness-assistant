<script setup lang="ts">
import {computed,ref,watch} from 'vue'
import {useFitzen} from '../state'
import {profileForm,profilePayload} from '../profile-form'
const {ui,saveProfileData}=useFitzen()
const form=ref(profileForm(ui.profile)),saving=ref(false),error=ref(''),feedback=ref('')
const savedSignature=ref(JSON.stringify(form.value))
const dirty=computed(()=>JSON.stringify(form.value)!==savedSignature.value)
const goals=['保持健康','增肌','减脂','提升体能']
function reset(){form.value=profileForm(ui.profile);savedSignature.value=JSON.stringify(form.value);error.value='';feedback.value=''}
watch(()=>ui.profile,()=>{if(!saving.value&&!dirty.value)reset()},{deep:true})
async function save(){
 if(saving.value||ui.loading)return
 error.value='';feedback.value=''
 let payload
 try{payload=profilePayload(form.value)}catch(reason){error.value=reason instanceof Error?reason.message:'请检查填写内容';return}
 if(payload.body_fat_pct!=null&&!Object.prototype.hasOwnProperty.call(ui.profile,'body_fat_pct')){error.value='当前服务暂时无法保存体脂率，请更新服务后重试。';return}
 saving.value=true
 try{
  if(await saveProfileData(payload)){reset();feedback.value='个人信息已保存'}
  else error.value=ui.notice||'保存未完成，请稍后重试。'
 }finally{saving.value=false}
}
</script>

<template>
<div class="screen-profile bg-background text-on-surface">
  <main class="profile-page fz-page-inset">
    <header class="profile-heading">
      <div><h1>个人信息</h1><p>设置身体数据与训练目标</p></div>
      <span class="profile-sync" role="status"><span class="material-symbols-outlined" aria-hidden="true">{{dirty?'edit_note':'person'}}</span>{{ui.loading?'正在读取':dirty?'有未保存的修改':ui.online?'资料已同步':'暂未连接'}}</span>
    </header>
    <form @submit.prevent.stop="save" novalidate>
      <fieldset class="profile-fields" :disabled="saving||ui.loading">
        <div class="profile-panels">
          <section class="profile-panel" aria-labelledby="profile-body-heading">
            <div class="profile-section-heading"><span class="material-symbols-outlined" aria-hidden="true">accessibility_new</span><h2 id="profile-body-heading">身体数据</h2></div>
            <div class="profile-input-grid">
              <label class="profile-field profile-full" for="profile-name"><span>称呼</span><input id="profile-name" v-model="form.display_name" class="fz-input" autocomplete="nickname" maxlength="100" placeholder="你希望被怎样称呼" /></label>
              <label class="profile-field" for="profile-height"><span>身高 <small>cm</small></span><input id="profile-height" v-model="form.height_cm" class="fz-input" type="number" inputmode="decimal" min="100" max="250" step="0.1" placeholder="未填写" /></label>
              <label class="profile-field" for="profile-weight"><span>体重 <small>kg</small></span><input id="profile-weight" v-model="form.body_weight_kg" class="fz-input" type="number" inputmode="decimal" min="20" max="400" step="0.1" placeholder="未填写" /></label>
              <label class="profile-field" for="profile-body-fat"><span>体脂率 <small>%</small></span><input id="profile-body-fat" v-model="form.body_fat_pct" class="fz-input" type="number" inputmode="decimal" min="0.1" max="99.9" step="0.1" placeholder="未填写" /></label>
              <label class="profile-field" for="profile-target-weight"><span>目标体重 <small>kg</small></span><input id="profile-target-weight" v-model="form.target_weight_kg" class="fz-input" type="number" inputmode="decimal" min="20" max="400" step="0.1" placeholder="选填" /></label>
            </div>
          </section>
          <section class="profile-panel" aria-labelledby="profile-training-heading">
            <div class="profile-section-heading"><span class="material-symbols-outlined" aria-hidden="true">flag</span><h2 id="profile-training-heading">训练目标</h2></div>
            <div class="profile-goals" role="group" aria-label="快捷选择训练目标"><button v-for="goal in goals" :key="goal" type="button" :aria-pressed="form.goal===goal" @click="form.goal=goal;feedback=''">{{goal}}</button></div>
            <label class="profile-field" for="profile-goal"><span>训练目标</span><input id="profile-goal" v-model="form.goal" class="fz-input" maxlength="100" placeholder="也可以填写自己的目标" /></label>
            <label class="profile-field" for="profile-experience"><span>训练经验</span><input id="profile-experience" v-model="form.experience" class="fz-input" maxlength="100" placeholder="例如：初学者、1年力量训练" /></label>
            <label class="profile-field" for="profile-days"><span>每周训练天数 <small>天 / 周</small></span><input id="profile-days" v-model="form.available_days" class="fz-input" type="number" inputmode="numeric" min="1" max="7" step="1" /></label>
          </section>
        </div>
      </fieldset>
      <div class="profile-save-area">
        <div class="profile-result" aria-live="polite"><p v-if="error" role="alert" class="text-error">{{error}}</p><p v-else-if="feedback" role="status" class="text-primary">{{feedback}}</p></div>
        <div class="profile-actions"><button type="button" class="fz-button" :disabled="saving||ui.loading||!dirty" @click="reset">恢复已保存</button><button type="submit" class="fz-button fz-primary" :disabled="saving||ui.loading" :aria-busy="saving"><span v-if="!saving" class="material-symbols-outlined" aria-hidden="true">check</span>{{saving?'保存中…':'保存个人信息'}}</button></div>
      </div>
    </form>
  </main>
</div>
</template>

<style scoped>
.profile-page{max-width:1100px;margin-inline:auto;padding-bottom:32px}
.profile-heading{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:var(--fz-section-gap)}
.profile-heading h1{margin:0;font-size:28px;font-weight:650;line-height:1.4;letter-spacing:-.025em}
.profile-heading p{margin:8px 0 0;font-size:14px;color:var(--fz-on-surface-variant)}
.profile-sync{display:inline-flex;align-items:center;gap:6px;flex-shrink:0;padding:6px 10px;border-radius:999px;background:var(--fz-surface-container);font-size:12px;color:var(--fz-on-surface-variant)}
.profile-sync .material-symbols-outlined{font-size:16px}
.profile-fields{margin:0;padding:0;border:0;min-width:0}
.profile-panels{display:grid;grid-template-columns:1fr 1fr;gap:var(--fz-module-gap)}
.profile-panel{min-width:0;padding:24px;border:1px solid var(--fz-outline-variant);border-radius:16px;background:var(--fz-surface-card)}
.profile-section-heading{display:flex;align-items:center;gap:10px;margin-bottom:24px}
.profile-section-heading .material-symbols-outlined{font-size:22px;color:var(--fz-primary)}
.profile-section-heading h2{margin:0;font-size:17px;font-weight:600}
.profile-input-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px 16px}
.profile-full{grid-column:1/-1}
.profile-field{display:flex;flex-direction:column;gap:8px;min-width:0;font-size:13px;color:var(--fz-on-surface)}
.profile-field>span{display:flex;align-items:center;justify-content:space-between;gap:8px}
.profile-field small{color:var(--fz-on-surface-variant);font-size:12px}
.profile-field input{width:100%;min-width:0;min-height:44px;font-size:14px;background:var(--fz-background)}
.profile-panel>.profile-field+.profile-field{margin-top:20px}
.profile-goals{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.profile-goals button{min-height:36px;padding:6px 14px;border:1px solid var(--fz-outline-variant);border-radius:999px;background:var(--fz-background);color:var(--fz-on-surface-variant);font-size:13px;transition:background-color .15s,color .15s,border-color .15s}
.profile-goals button:hover:not(:disabled){border-color:var(--fz-primary);color:var(--fz-primary)}
.profile-goals button[aria-pressed="true"]{border-color:var(--fz-primary);background:var(--fz-primary);color:var(--fz-on-primary)}
.profile-goals button:focus-visible{outline:2px solid var(--fz-primary);outline-offset:3px}
.profile-save-area{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-top:var(--fz-module-gap)}
.profile-result{min-width:0;font-size:13px;line-height:1.6}.profile-result p{margin:0}
.profile-actions{display:flex;align-items:center;gap:12px;flex-shrink:0}
.profile-actions button{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:42px;padding:9px 18px;border-radius:999px;white-space:nowrap}
.profile-actions .material-symbols-outlined{font-size:18px}
@media(max-width:850px){.profile-panels{grid-template-columns:1fr}}
@media(max-width:480px){.profile-heading{align-items:flex-start;flex-direction:column;gap:12px}.profile-heading h1{font-size:24px}.profile-panel{padding:20px}.profile-save-area{align-items:stretch;flex-direction:column}.profile-actions{justify-content:flex-end}.profile-actions button{padding-inline:14px}}
@media(prefers-reduced-motion:reduce){.profile-goals button{transition:none}}
</style>
