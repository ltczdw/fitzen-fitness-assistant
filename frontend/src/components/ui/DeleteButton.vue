<script setup lang="ts">
import {computed, onBeforeUnmount, onMounted, ref, watch} from 'vue'
import {bindDeleteConfirmationEvents, deleteConfirmation} from './delete-confirmation'

const props = withDefaults(defineProps<{
  action: () => void | Promise<unknown>
  subject: string
  label?: string
  disabled?: boolean
  loading?: boolean
  iconOnly?: boolean
  compact?: boolean
}>(), {label: '删除', disabled: false, loading: false, iconOnly: false, compact: false})
const emit = defineEmits<{error: [error: unknown]}>()
const id = Symbol('delete-button')
const armed = ref(false), running = ref(false), failed = ref(false)
const busy = computed(() => props.loading || running.value)
const blocked = computed(() => props.disabled || busy.value)
const accessibleLabel = computed(() => failed.value ? `删除${props.subject}失败，请重试` : busy.value ? `正在${props.label}${props.subject}` : armed.value ? `再次点击确认${props.label}${props.subject}` : `${props.label}${props.subject}`)
let unsubscribe: (() => void) | undefined, unbind: (() => void) | undefined
onMounted(() => {
  unsubscribe = deleteConfirmation.subscribe(() => { armed.value = deleteConfirmation.isArmed(id) })
  unbind = bindDeleteConfirmationEvents()
})
onBeforeUnmount(() => { deleteConfirmation.clear(id); unsubscribe?.(); unbind?.() })
watch(blocked, value => { if (value) deleteConfirmation.clear(id) })
async function activate() {
  if (blocked.value) return
  failed.value = false
  if (!deleteConfirmation.activate(id)) return
  running.value = true
  try { await props.action() }
  catch (error) { failed.value = true; emit('error', error) }
  finally { running.value = false }
}
function preventRepeatedKey(event: KeyboardEvent) {
  if (event.repeat && ['Enter', ' '].includes(event.key)) event.preventDefault()
}
</script>

<template>
  <button type="button" class="fz-delete-button" :class="{'is-armed': armed, 'is-icon-only': iconOnly, 'is-compact': compact, 'is-busy': busy}"
    data-delete-confirm :disabled="blocked" :aria-label="accessibleLabel" :title="accessibleLabel" :aria-busy="busy"
    @click.stop="activate" @keydown="preventRepeatedKey">
    <span class="material-symbols-outlined" aria-hidden="true">{{busy ? 'hourglass_top' : armed ? 'check' : 'delete'}}</span>
    <span class="fz-delete-label">{{busy ? '处理中…' : armed ? '确认' : failed ? '重试' : label}}</span>
  </button>
</template>

<style scoped>
button.fz-delete-button {
  display: inline-flex; align-items: center; justify-content: center; gap: 3px;
  flex-shrink: 0; height: var(--fz-delete-height); min-width: var(--fz-delete-height); padding: 0 6px;
  border: 1px solid var(--fz-error-container); border-radius: var(--fz-delete-radius);
  background: transparent; color: var(--fz-error); font-size: var(--fz-delete-font-size);
  font-weight: 400; line-height: 1; white-space: nowrap; cursor: pointer;
  transition: background .15s, border-color .15s, color .15s;
}
button.fz-delete-button:hover { background: var(--fz-error-container); }
button.fz-delete-button.is-armed { background: var(--fz-primary); color: var(--fz-on-primary); border-color: var(--fz-primary); }
button.fz-delete-button:focus-visible { outline: 2px solid var(--fz-primary); outline-offset: 2px; }
button.fz-delete-button:disabled { opacity: .5; cursor: not-allowed; }
button.fz-delete-button.is-busy { cursor: wait; }
.fz-delete-button .material-symbols-outlined { font-size: var(--fz-delete-icon-size); }
.fz-delete-button.is-icon-only .fz-delete-label { display: none; }
@media(max-width:560px) { .fz-delete-button.is-compact .fz-delete-label { display: none; } }
@media(prefers-reduced-motion:reduce) { button.fz-delete-button { transition: none; } }
</style>
