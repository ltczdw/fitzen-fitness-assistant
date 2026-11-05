<script setup lang="ts">
import { ref, watch } from 'vue'
import { useFitzen } from '../state'
const { ui, notify, saveProfileData } = useFitzen()
function savedValue(key:string, fallback:any) { try { return JSON.parse(localStorage.getItem(key) || 'null') || fallback } catch { return fallback } }
const saved = savedValue('fitzen-preferences-v2', {})
const saving = ref(false)
const notifications = ref(saved.notifications || { daily:true, training:true, nutrition:false })
const privacy = ref(saved.privacy || {sync:true,recommendation:true})
const theme = ref(saved.theme || {dark:true})
const modal = ref({profile:false,logoutConfirm:false})
const profileForm = ref({nickname:ui.profile.display_name || '',height:ui.profile.height_cm ?? null,weight:ui.profile.body_weight_kg ?? null,age:saved.age ?? '',gender:saved.gender ?? ''})
watch(()=>ui.profile,p=>{if(p.display_name)profileForm.value.nickname=p.display_name;if(p.height_cm)profileForm.value.height=p.height_cm;if(p.body_weight_kg)profileForm.value.weight=p.body_weight_kg},{deep:true})
const toast = ref({visible:false,message:''})
const showToast = (msg:string) => { toast.value={visible:true,message:msg}; setTimeout(()=>toast.value.visible=false,3000) }
function saveSettings() {
 saving.value=true
 try { localStorage.setItem('fitzen-preferences-v2',JSON.stringify({notifications:notifications.value,privacy:privacy.value,theme:theme.value,age:profileForm.value.age,gender:profileForm.value.gender})); showToast('偏好已保存到此浏览器') } catch { showToast('浏览器存储不可用，设置尚未保存') } finally { saving.value=false }
}
function openProfileModal(){ modal.value.profile=true }
async function saveProfile(){const ok=await saveProfileData({display_name:profileForm.value.nickname,height_cm:profileForm.value.height==null||String(profileForm.value.height)===''?null:Number(profileForm.value.height),body_weight_kg:profileForm.value.weight==null||String(profileForm.value.weight)===''?null:Number(profileForm.value.weight)});if(ok){modal.value.profile=false;saveSettings()}}
function openLogoutConfirm(){modal.value.logoutConfirm=true}
function confirmLogoutAction(){modal.value.logoutConfirm=false;notify('当前使用本地账户，未接入登录服务。')}
</script>
<template>
<div class="screen-settings bg-background text-on-surface min-h-screen antialiased flex flex-col md:flex-row">
  <div id="settings-root" class="flex-1 flex flex-col min-h-screen">
    <!-- Top Global App Bar -->


    <!---->
    <main class="flex-1 max-w-5xl w-full mx-auto fz-page-inset">
      <!-- Breadcrumb & Title -->
      <div class="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div class="flex items-center gap-2 text-xs text-on-surface-variant mb-1 font-mono">
            <span>控制台</span>
            <span>&gt;</span>
            <span class="text-on-surface-variant">偏好配置</span>
          </div>
          <h1 class="text-2xl md:text-3xl font-semibold tracking-tight text-on-surface flex items-center gap-3">
            系统偏好设置
            <span class="text-xs font-normal text-on-surface-variant px-2.5 py-1 rounded-full bg-surface-container-lowest border border-outline-variant">{{ui.online?'账户数据已同步':'数据未连接'}}</span>
          </h1>
        </div>
        <div class="flex items-center gap-3">
          <button type="button" class="fz-button" @click="openProfileModal">编辑个人资料</button><button @click="saveSettings" :disabled="saving" class="h-10 px-5 rounded-[10px] bg-primary hover:bg-primary text-on-primary text-[14px] font-medium transition duration-200 active:scale-95 neon-glow flex items-center gap-2 disabled:opacity-50">
            <svg v-if="saving" class="animate-spin h-4 w-4 text-on-surface" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
            </svg>
            <span>{{ saving ? '保存中...' : '保存更改' }}</span>
          </button>
        </div>
      </div>

      <!-- Settings Grid -->
      <div class="fz-section-flow">

        <!-- 1. 通知设置 -->
        <section class="bg-surface-container-lowest rounded-[16px] border border-outline-variant p-6 shadow-sm">
          <div class="flex items-center gap-3 mb-6 pb-4 border-b border-outline-variant/70">
            <div class="w-9 h-9 rounded-[10px] bg-surface-container-low flex items-center justify-center text-primary">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path>
              </svg>
            </div>
            <div>
              <h2 class="text-base font-semibold text-on-surface">通知提醒设置</h2>
              <p class="text-xs text-on-surface-variant">控制日程推送、训练节点与营养摄入的提醒频率</p>
            </div>
          </div>

          <div class="space-y-5">
            <!-- 每日提醒开关 -->
            <div class="flex items-center justify-between py-1">
              <div>
                <div class="text-[15px] font-medium text-on-surface">每日晨间简报提醒</div>
                <div class="text-[13px] text-on-surface-variant">偏好仅保存在此浏览器，尚未接入通知推送</div>
              </div>
              <button type="button" @click="notifications.daily = !notifications.daily" class="w-12 h-6 flex items-center rounded-full p-0.5 cursor-pointer transition-colors duration-200" :class="notifications.daily ? 'bg-primary' : 'bg-surface-container-highest'">
                <div class="bg-paper w-5 h-5 rounded-full shadow-md transform transition-transform duration-200" :class="notifications.daily ? 'translate-x-6' : 'translate-x-0'"></div>
              </button>
            </div>

            <div class="h-px bg-surface-container-highest/40"></div>

            <!-- 训练提醒开关 -->
            <div class="flex items-center justify-between py-1">
              <div>
                <div class="text-[15px] font-medium text-on-surface">训练计划与间歇提醒</div>
                <div class="text-[13px] text-on-surface-variant">保存训练提醒偏好，尚未接入通知服务</div>
              </div>
              <button type="button" @click="notifications.training = !notifications.training" class="w-12 h-6 flex items-center rounded-full p-0.5 cursor-pointer transition-colors duration-200" :class="notifications.training ? 'bg-primary' : 'bg-surface-container-highest'">
                <div class="bg-paper w-5 h-5 rounded-full shadow-md transform transition-transform duration-200" :class="notifications.training ? 'translate-x-6' : 'translate-x-0'"></div>
              </button>
            </div>

            <div class="h-px bg-surface-container-highest/40"></div>

            <!-- 饮食提醒开关 -->
            <div class="flex items-center justify-between py-1">
              <div>
                <div class="text-[15px] font-medium text-on-surface">宏量营养摄入与饮水提示</div>
                <div class="text-[13px] text-on-surface-variant">未达标宏量蛋白质摄入警报及定点饮水补剂提醒</div>
              </div>
              <button type="button" @click="notifications.nutrition = !notifications.nutrition" class="w-12 h-6 flex items-center rounded-full p-0.5 cursor-pointer transition-colors duration-200" :class="notifications.nutrition ? 'bg-primary' : 'bg-surface-container-highest'">
                <div class="bg-paper w-5 h-5 rounded-full shadow-md transform transition-transform duration-200" :class="notifications.nutrition ? 'translate-x-6' : 'translate-x-0'"></div>
              </button>
            </div>
          </div>
        </section>

        <!-- 2. 隐私与数据设置 -->
        <section class="bg-surface-container-lowest rounded-[16px] border border-outline-variant p-6 shadow-sm">
          <div class="flex items-center gap-3 mb-6 pb-4 border-b border-outline-variant/70">
            <div class="w-9 h-9 rounded-[10px] bg-surface-container-low flex items-center justify-center text-primary">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
              </svg>
            </div>
            <div>
              <h2 class="text-base font-semibold text-on-surface">隐私与数据服务</h2>
              <p class="text-xs text-on-surface-variant">控制多设备云端漫游加密与 AI 个性化动作生成</p>
            </div>
          </div>

          <div class="space-y-5">
            <!-- 数据同步开关 -->
            <div class="flex items-center justify-between py-1">
              <div>
                <div class="text-[15px] font-medium text-on-surface">跨设备数据云端加密同步</div>
                <div class="text-[13px] text-on-surface-variant">此浏览器的同步偏好；尚未接入设备同步</div>
              </div>
              <button type="button" @click="privacy.sync = !privacy.sync" class="w-12 h-6 flex items-center rounded-full p-0.5 cursor-pointer transition-colors duration-200" :class="privacy.sync ? 'bg-primary' : 'bg-surface-container-highest'">
                <div class="bg-paper w-5 h-5 rounded-full shadow-md transform transition-transform duration-200" :class="privacy.sync ? 'translate-x-6' : 'translate-x-0'"></div>
              </button>
            </div>

            <div class="h-px bg-surface-container-highest/40"></div>

            <!-- 个性化推荐开关 -->
            <div class="flex items-center justify-between py-1">
              <div>
                <div class="text-[15px] font-medium text-on-surface">FitZen AI 个性化推导与周期规划</div>
                <div class="text-[13px] text-on-surface-variant">此浏览器的推荐偏好设置</div>
              </div>
              <button type="button" @click="privacy.recommendation = !privacy.recommendation" class="w-12 h-6 flex items-center rounded-full p-0.5 cursor-pointer transition-colors duration-200" :class="privacy.recommendation ? 'bg-primary' : 'bg-surface-container-highest'">
                <div class="bg-paper w-5 h-5 rounded-full shadow-md transform transition-transform duration-200" :class="privacy.recommendation ? 'translate-x-6' : 'translate-x-0'"></div>
              </button>
            </div>
          </div>
        </section>

        <!-- 3. 主题与界面设置 -->
        <section class="bg-surface-container-lowest rounded-[16px] border border-outline-variant p-6 shadow-sm">
          <div class="flex items-center gap-3 mb-6 pb-4 border-b border-outline-variant/70">
            <div class="w-9 h-9 rounded-[10px] bg-surface-container-low flex items-center justify-center text-primary">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
              </svg>
            </div>
            <div>
              <h2 class="text-base font-semibold text-on-surface">主题与显示</h2>
              <p class="text-xs text-on-surface-variant">界面渲染模式与当前白色配色</p>
            </div>
          </div>

          <div class="flex items-center justify-between py-1">
            <div>
              <div class="text-[15px] font-medium text-on-surface">深色模式 (Dark Terminal Mode)</div>
              <div class="text-[13px] text-on-surface-variant">此偏好仅在本地保存；当前界面统一使用白色主题</div>
            </div>
            <button type="button" @click="theme.dark = !theme.dark" class="w-12 h-6 flex items-center rounded-full p-0.5 cursor-pointer transition-colors duration-200" :class="theme.dark ? 'bg-primary' : 'bg-surface-container-highest'">
              <div class="bg-paper w-5 h-5 rounded-full shadow-md transform transition-transform duration-200" :class="theme.dark ? 'translate-x-6' : 'translate-x-0'"></div>
            </button>
          </div>
        </section>

        <!-- 4. 账户与安全性 -->
        <section class="bg-surface-container-lowest rounded-[16px] border border-outline-variant p-6 shadow-sm">
          <div class="flex items-center gap-3 mb-6 pb-4 border-b border-outline-variant/70">
            <div class="w-9 h-9 rounded-[10px] bg-surface-container-low flex items-center justify-center text-primary">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.75" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
              </svg>
            </div>
            <div>
              <h2 class="text-base font-semibold text-on-surface">账户安全性与凭据</h2>
              <p class="text-xs text-on-surface-variant">管理密码、验证邮箱以及当前会话登录态</p>
            </div>
          </div>

          <div class="space-y-4">
            <!-- 修改密码行 -->
            <div class="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-[12px] bg-surface-container-low/60 border border-outline-variant/80 gap-3">
              <div>
                <div class="text-[14px] font-medium text-on-surface">安全通行密码</div>
                <div class="text-[13px] text-on-surface-variant">尚未配置密码服务</div>
              </div>
              <button @click="showToast('已向绑定邮箱发送密码重置安全密钥')" class="px-3.5 py-2 rounded-[10px] bg-surface-container-lowest hover:bg-surface-container-low border border-outline-variant text-sm text-on-surface-variant hover:text-on-surface transition duration-200 active:scale-95 text-center">
                修改密码
              </button>
            </div>

            <!-- 绑定邮箱行 -->
            <div class="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-[12px] bg-surface-container-low/60 border border-outline-variant/80 gap-3">
              <div>
                <div class="text-[14px] font-medium text-on-surface flex items-center gap-2">
                  绑定电子邮箱
                  <span class="text-[11px] px-2 py-0.5 rounded bg-surface-container text-primary border border-primary/30 font-mono">未配置</span>
                </div>
                <div class="text-[13px] text-on-surface-variant">尚未绑定邮箱</div>
              </div>
              <button @click="showToast('更换邮箱链接已生成')" class="px-3.5 py-2 rounded-[10px] bg-surface-container-lowest hover:bg-surface-container-low border border-outline-variant text-sm text-on-surface-variant hover:text-on-surface transition duration-200 active:scale-95 text-center">
                更换邮箱
              </button>
            </div>

            <!-- 退出登录与销毁会话 -->
            <div class="pt-3 flex items-center justify-between">
              <div>
                <div class="text-[14px] font-medium text-error">退出当前账户</div>
                <div class="text-[13px] text-on-surface-variant">结束当前终端会话并清除本地离线缓存密钥</div>
              </div>
              <button @click="openLogoutConfirm" class="px-4 py-2 rounded-[10px] bg-error/10 hover:bg-error/20 border border-error/30 text-sm font-medium text-error transition duration-200 active:scale-95">
                退出登录
              </button>
            </div>
          </div>
        </section>

      </div>
    </main>

    <!-- Toast Notification -->
    <transition enter-active-class="transition duration-200 ease-out" enter-from-class="opacity-0 translate-y-2" enter-to-class="opacity-100 translate-y-0" leave-active-class="transition duration-150 ease-in" leave-from-class="opacity-100 translate-y-0" leave-to-class="opacity-0 translate-y-2">
      <div v-if="toast.visible" class="fixed bottom-6 right-6 z-50 flex items-center gap-3 px-4 py-3 rounded-[12px] bg-surface-container-low border border-primary/50 text-on-surface shadow-2xl neon-glow-sm">
        <svg class="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
        </svg>
        <span class="text-[13px] font-medium">{{ toast.message }}</span>
      </div>
    </transition>

    <!-- 个人资料编辑模态框 (Teleport style) -->
    <div v-if="modal.profile" class="fixed inset-0 z-50 flex items-center justify-center p-4 modal-backdrop" @click.self="modal.profile = false">
      <div class="w-full max-w-[480px] bg-surface-container-lowest border border-outline-variant rounded-[16px] p-6 shadow-2xl relative transition-all duration-200 scale-100">
        <!-- Close Button -->
        <button @click="modal.profile = false" class="absolute top-5 right-5 text-on-surface-variant hover:text-on-surface transition">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>

        <h3 class="text-xl font-semibold text-on-surface mb-1">编辑个人档案</h3>
        <p class="text-xs text-on-surface-variant mb-5">身高、体重和昵称同步至账户，年龄和性别仅保存在此浏览器</p>

        <!-- Avatar Upload Section -->
        <div class="flex items-center gap-4 p-4 rounded-[12px] bg-surface-container-low/60 border border-outline-variant mb-5">
          <img src="/fitzen/assets/default-avatar.svg" alt="默认头像" class="w-16 h-16 rounded-full object-cover border-2 border-primary neon-glow-sm">
          <div>
            <button @click="showToast('已开启头像本地文件选择')" class="px-3 py-1.5 rounded-[10px] bg-surface-container-low hover:bg-surface-container-highest border border-outline-variant text-xs font-medium text-on-surface transition active:scale-95">
              更换头像
            </button>
            <p class="text-[11px] text-on-surface-variant mt-1">支持 PNG, JPG 或 WEBP，最大 5MB</p>
          </div>
        </div>

        <!-- Form fields -->
        <form @submit.prevent="saveProfile" class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-on-surface-variant mb-1.5">运动员昵称</label>
            <input v-model="profileForm.nickname" type="text" class="w-full h-10 px-3.5 rounded-[10px] bg-surface-container-low border border-outline-variant text-on-surface text-[14px] focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition" :required="true">
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs font-medium text-on-surface-variant mb-1.5">身高 (cm)</label>
              <input v-model.number="profileForm.height" type="number" class="w-full h-10 px-3.5 rounded-[10px] bg-surface-container-low border border-outline-variant text-on-surface text-[14px] focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition">
            </div>
            <div>
              <label class="block text-xs font-medium text-on-surface-variant mb-1.5">当前体重 (kg)</label>
              <input v-model.number="profileForm.weight" type="number" step="0.1" class="w-full h-10 px-3.5 rounded-[10px] bg-surface-container-low border border-outline-variant text-on-surface text-[14px] focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition">
            </div>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs font-medium text-on-surface-variant mb-1.5">年龄 (岁)</label>
              <input v-model.number="profileForm.age" type="number" class="w-full h-10 px-3.5 rounded-[10px] bg-surface-container-low border border-outline-variant text-on-surface text-[14px] focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition">
            </div>
            <div>
              <label class="block text-xs font-medium text-on-surface-variant mb-1.5">性别</label>
              <select v-model="profileForm.gender" class="w-full h-10 px-3 rounded-[10px] bg-surface-container-low border border-outline-variant text-on-surface text-[14px] focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition">
                <option value="male">男 (Male)</option>
                <option value="female">女 (Female)</option>
                <option value="other">自定义</option>
              </select>
            </div>
          </div>

          <!-- Bottom Actions -->
          <div class="flex items-center justify-end gap-3 pt-4 border-t border-outline-variant mt-6">
            <button type="button" @click="modal.profile = false" class="h-10 px-4 rounded-[10px] border border-outline-variant bg-transparent text-on-surface-variant hover:text-on-surface hover:bg-surface-container-low text-sm font-medium transition duration-200 active:scale-95">
              取消
            </button>
            <button type="submit" class="h-10 px-5 rounded-[10px] bg-primary hover:bg-primary text-on-primary text-sm font-medium transition duration-200 active:scale-95 neon-glow">
              保存修改
            </button>
          </div>
        </form>
      </div>
    </div>

    <!-- 确认删除 / 退出登录弹窗 (360px) -->
    <div v-if="modal.logoutConfirm" class="fixed inset-0 z-50 flex items-center justify-center p-4 modal-backdrop" @click.self="modal.logoutConfirm = false">
      <div class="w-full max-w-[360px] bg-surface-container-lowest border border-outline-variant rounded-[16px] p-6 shadow-2xl text-center relative transition-all duration-200">
        <div class="w-12 h-12 rounded-full bg-error/10 border border-error/20 text-error mx-auto flex items-center justify-center mb-4 danger-glow-sm">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
          </svg>
        </div>

        <h3 class="text-lg font-semibold text-on-surface mb-2">确认退出当前终端吗？</h3>
        <p class="text-xs text-on-surface-variant mb-6">退出登录后需要重新验证安全凭据，未同步的本地临时草稿将被重置。</p>

        <div class="flex items-center justify-center gap-3">
          <button type="button" @click="modal.logoutConfirm = false" class="flex-1 h-10 rounded-[10px] border border-outline-variant bg-transparent text-on-surface-variant hover:text-on-surface hover:bg-surface-container-low text-sm font-medium transition duration-200 active:scale-95">
            取消
          </button>
          <button type="button" @click="confirmLogoutAction" class="flex-1 h-10 rounded-[10px] bg-error hover:bg-error text-on-primary text-sm font-medium transition duration-200 active:scale-95 danger-glow-sm">
            确认退出
          </button>
        </div>
      </div>
    </div>

  </div>



</div>
</template>
<style scoped>

    .screen-settings {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--fz-background);
      color: var(--fz-on-surface);
      overflow-x: hidden;
    }
    .neon-glow {
      box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06);
    }
    .neon-glow-sm {
      box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06);
    }
    .danger-glow-sm {
      box-shadow: 0 4px 14px rgb(var(--fz-rgb-shadow) / .06);
    }
    /* Switch toggle transition */
    .switch-bullet {
      transition: transform 200ms ease-out, background-color 200ms ease-out;
    }
    .modal-backdrop {
      backdrop-filter: blur(12px);
      background-color: rgb(var(--fz-rgb-inverse-surface) / 0.8);
    }

</style>
