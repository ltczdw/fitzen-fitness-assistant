<script setup lang="ts">
import { notify } from "../../fitzen/state"
import { onBeforeUnmount, onMounted, ref } from 'vue'
import DeleteButton from './DeleteButton.vue'
import {deleteConfirmation} from './delete-confirmation'

type Workout = { id: number; date: string; exercise: string; sets: number; reps: number; weight_kg: number; difficulty: string; completed: boolean; note: string }

const OPEN_EVENT = 'fitness-open-workout-records'
const open = ref(false)
const loading = ref(false)
const saving = ref(false)
const notice = ref('')
const workouts = ref<Workout[]>([])
const editingWorkout = ref<Workout | null>(null)

function clone<T>(value: T): T { return JSON.parse(JSON.stringify(value)) as T }
function formatDate(value: string) { return new Date(value).toLocaleDateString('zh-CN') }
function clearDeleteConfirmation(){deleteConfirmation.clear()}

async function loadWorkouts() {
  loading.value = true
  clearDeleteConfirmation()
  try {
    const response = await fetch('/api/workouts?user_id=demo-user&limit=100')
    if (!response.ok) throw new Error('load failed')
    workouts.value = await response.json()
  } catch {
    notice.value = '训练记录加载失败，请确认后端服务正常。'
  } finally {
    loading.value = false
  }
}

function openManager() {
  open.value = true
  editingWorkout.value = null
  notice.value = ''
  void loadWorkouts()
}

function editRecord(event: Event) {
  open.value = true
  notice.value = ''
  clearDeleteConfirmation()
  editingWorkout.value = clone((event as CustomEvent<Workout>).detail)
}
function startEdit(item: Workout) { clearDeleteConfirmation(); editingWorkout.value = clone(item) }

async function saveWorkout() {
  if (!editingWorkout.value || saving.value) return
  saving.value = true
  const item = editingWorkout.value
  try {
    const response = await fetch('/api/workouts/' + item.id + '?user_id=demo-user', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date: item.date, exercise: item.exercise, sets: Number(item.sets), reps: Number(item.reps), weight_kg: Number(item.weight_kg), difficulty: item.difficulty || '未记录', completed: Boolean(item.completed), note: item.note || '' }),
    })
    if (!response.ok) throw new Error('save failed')
    editingWorkout.value = null
    notice.value = '训练记录已修改。'
    await loadWorkouts()
    window.dispatchEvent(new Event('fitness-workout-records-changed'))
  } catch {
    notice.value = '训练记录修改失败，请检查输入或后端服务。'
  } finally {
    saving.value = false
  }
}

async function deleteWorkout(item: Workout) {
  clearDeleteConfirmation()
  try {
    const response = await fetch('/api/workouts/' + item.id + '?user_id=demo-user', { method: 'DELETE' })
    if (!response.ok) throw new Error('delete failed')
    notice.value = ''
    notify('训练记录已删除')
    await loadWorkouts()
    window.dispatchEvent(new Event('fitness-workout-records-changed'))
  } catch {
    notice.value = '训练记录删除失败。'
  }
}

onMounted(() => {
  window.addEventListener(OPEN_EVENT, openManager)
  window.addEventListener('fitness-edit-workout-record', editRecord)
})
onBeforeUnmount(() => {
  window.removeEventListener(OPEN_EVENT, openManager)
  window.removeEventListener('fitness-edit-workout-record', editRecord)
  clearDeleteConfirmation()
})
</script>

<template>
  <div v-if="open" class="workout-manager-backdrop" @click.self="open = false">
    <section class="workout-manager-dialog" role="dialog" aria-modal="true" aria-label="训练记录管理">
      <header class="workout-manager-header"><div><span>TRAINING RECORDS</span><h2>训练记录</h2><small>这里只管理训练记录；多动作训练请使用“训练会话”。</small></div><button type="button" @click="open = false; clearDeleteConfirmation()">×</button></header>
      <p v-if="notice" class="workout-manager-notice">{{ notice }}</p>
      <div class="workout-manager-toolbar"><span>{{ workouts.length }} 条记录</span><button type="button" @click="loadWorkouts">{{ loading ? '同步中…' : '刷新' }}</button></div>
      <div class="workout-manager-list">
        <article v-for="item in workouts" :key="item.id" class="workout-manager-row">
          <div class="workout-manager-main"><strong>{{ item.exercise }}</strong><span>{{ formatDate(item.date) }} · {{ item.weight_kg }} kg · {{ item.reps }} 次 · {{ item.sets }} 组</span><small>难度：{{ item.difficulty || '未记录' }} · {{ item.completed ? '已完成' : '未完成' }} · {{ item.note || '无备注' }}</small></div>
          <div class="workout-manager-actions"><button type="button" @click="startEdit(item)">修改</button><DeleteButton :action="()=>deleteWorkout(item)" :subject="item.exercise" :disabled="saving||loading" compact /></div>
        </article>
        <p v-if="!workouts.length" class="workout-manager-empty">暂无训练记录。</p>
      </div>
      <form v-if="editingWorkout" class="workout-manager-form" @submit.prevent="saveWorkout">
        <h3>修改训练记录</h3>
        <div class="workout-manager-grid"><input v-model="editingWorkout.exercise" required placeholder="动作" /><input v-model="editingWorkout.date" type="date" required /><input v-model.number="editingWorkout.sets" type="number" min="1" required placeholder="组数" /><input v-model.number="editingWorkout.reps" type="number" min="1" required placeholder="次数" /><input v-model.number="editingWorkout.weight_kg" type="number" min="0" step="0.1" required placeholder="重量 kg" /><select v-model="editingWorkout.difficulty"><option>未记录</option><option>轻松</option><option>正常</option><option>困难</option></select><label class="workout-manager-completed"><input v-model="editingWorkout.completed" type="checkbox" /> 已完成</label><input v-model="editingWorkout.note" placeholder="备注" /></div>
        <div class="workout-manager-form-actions"><button type="button" @click="editingWorkout = null">取消</button><button class="primary" type="submit">{{ saving ? '保存中…' : '保存修改' }}</button></div>
      </form>
    </section>
  </div>
</template>

<style scoped>
.workout-manager-backdrop { position: fixed; inset: 0; z-index: 40; display: grid; place-items: center; padding: 20px; background: rgb(var(--fz-rgb-scrim) / .38); }
.workout-manager-dialog { width: min(760px, 100%); max-height: min(780px, 92vh); overflow: auto; border: 1px solid var(--fz-outline-variant); border-radius: 18px; background: var(--fz-surface-card); box-shadow: 0 24px 80px rgb(var(--fz-rgb-shadow) / .12); padding: 20px; color: var(--fz-on-surface-variant); }
.workout-manager-header, .workout-manager-toolbar, .workout-manager-row, .workout-manager-form-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; }.workout-manager-header span { color: var(--fz-on-surface-variant); font-size: 9px; letter-spacing: .12em; }.workout-manager-header h2 { margin: 5px 0 0; color: var(--fz-on-surface); font-size: 20px; }.workout-manager-header small { display: block; margin-top: 5px; color: var(--fz-on-surface-variant); font-size: 10px; }.workout-manager-header > button { border: 0; background: transparent; color: var(--fz-on-surface-variant); font-size: 24px; cursor: pointer; }
@media (max-width: 640px) { .workout-manager-backdrop { padding: 8px; }.workout-manager-dialog { padding: 14px; }.workout-manager-grid { grid-template-columns: 1fr; }.workout-manager-row { align-items: flex-start; }.workout-manager-actions { flex-direction: column; } }
 .workout-manager-grid select { width: 100%; box-sizing: border-box; border: 1px solid var(--fz-outline-variant); border-radius: 8px; padding: 9px 10px; background: var(--fz-paper); color: var(--fz-on-surface-variant); font-size: 11px; }
.workout-manager-completed { display: flex; align-items: center; gap: 6px; color: var(--fz-on-surface-variant); font-size: 10px; }
.workout-manager-completed input { width: 15px; height: 15px; }
</style>
