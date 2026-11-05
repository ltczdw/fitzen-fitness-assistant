<script setup lang="ts">
import { notify, upsertExerciseMeta } from "../../fitzen/state"
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import DeleteButton from './DeleteButton.vue'
import {deleteConfirmation} from './delete-confirmation'

type WorkoutSet = { id?: number; key?: string; set_index?: number; distance_km?: number | null; duration_minutes?: number | null; weight_kg: number; reps: number; difficulty: string; completed: boolean; note: string }
type WorkoutExercise = { measurement_type?: ExerciseType; id?: number; key?: string; exercise_name: string; difficulty: string; note: string; sets: WorkoutSet[] }
type WorkoutSession = { id: number; user_id: string; session_date: string; title: string; note: string; exercises: WorkoutExercise[] }
type WorkoutSessionDraft = { session_date: string; title: string; note: string; exercises: WorkoutExercise[] }
type ExerciseLibraryItem = { id?: number; name: string; category: string; target_muscles: string; source: 'local' | 'custom'; usage_count: number; recent_use_count: number; recent_use_days: number; rank_score: number }
type ExerciseLibraryTab = 'recommended' | 'all' | 'custom'
type ExerciseType = 'strength' | 'cardio'

const OPEN_EVENT = 'fitness-open-workout-sessions'
const EDIT_EVENT = 'fitness-edit-workout-session'
const CHANGED_EVENT = 'fitness-workout-records-changed'
const EXERCISE_LIBRARY_TABS: Array<{ id: ExerciseLibraryTab; label: string; icon: string }> = [
  { id: 'recommended', label: '常用推荐', icon: 'recommend' },
  { id: 'all', label: '全部动作', icon: 'grid_view' },
  { id: 'custom', label: '自定义动作', icon: 'edit_note' },
]
const EXERCISE_TYPES: Array<{ id: ExerciseType; label: string; icon: string }> = [
  { id: 'strength', label: '力量', icon: 'fitness_center' },
  { id: 'cardio', label: '有氧', icon: 'directions_run' },
]
const CUSTOM_EXERCISE_CATEGORIES = ['自定义', '胸部', '背部', '腿部', '肩部', '手臂', '核心']
const exerciseType = ref<ExerciseType>('strength')
let inferExerciseTypeOnLoad = false
function typeOfExercise(item: ExerciseLibraryItem): ExerciseType { return item.category === '有氧' ? 'cardio' : 'strength' }
function switchExerciseType(value: ExerciseType) {
  if (saving.value || savingCustomExercise.value) return
  inferExerciseTypeOnLoad = false
  exerciseType.value = value
  if (!firstExercise.value.exercise_name.trim()) firstExercise.value.measurement_type = value
  if (value === 'cardio') defaultCardioDuration(firstExercise.value)
  exerciseLibraryCategory.value = '全部'
  customExerciseCategory.value = value === 'cardio' ? '有氧' : '自定义'
  if (quickExerciseResults.value) quickExerciseResults.value.scrollTop = 0
}
const customExerciseCategories = computed(() => exerciseType.value === 'cardio' ? ['有氧'] : CUSTOM_EXERCISE_CATEGORIES)

const open = ref(false)
const editorOpen = ref(false)
const loading = ref(false)
const saving = ref(false)
const notice = ref('')
const sessions = ref<WorkoutSession[]>([])
const editingId = ref<number | null>(null)
const exerciseLibraryOpen = ref(false)
const exerciseLibraryLoading = ref(false)
const exerciseLibraryError = ref('')
const exerciseLibrary = ref<ExerciseLibraryItem[]>([])
const exerciseLibraryTab = ref<ExerciseLibraryTab>('recommended')
const exerciseLibrarySearch = ref('')
const exerciseLibraryCategory = ref('全部')
const customExerciseName = ref('')
const customExerciseCategory = ref('自定义')
const customExerciseMuscles = ref('')
const savingCustomExercise = ref(false)
const dialogRef = ref<HTMLElement | null>(null)
let previouslyFocused: HTMLElement | null = null

function clearPendingDeleteConfirmation(){deleteConfirmation.clear()}

const DIFFICULTY_OPTIONS = [
  { value: '很轻松', emoji: '😊' },
  { value: '刚好', emoji: '😐' },
  { value: '很吃力', emoji: '😫' },
] as const
const openDifficultyIndex = ref<number | null>(null)
function difficultyEmoji(value: string): string {
  return DIFFICULTY_OPTIONS.find((o) => o.value === normalizeDifficulty(value))?.emoji || '😐'
}
function normalizeDifficulty(value: string): string {
  if (value === '轻松') return '很轻松'
  if (value === '正常' || value === '未记录' || !value) return '刚好'
  if (value === '困难') return '很吃力'
  return value
}
function toBackendDifficulty(value: string): string {
  if (value === '很轻松') return '轻松'
  if (value === '刚好') return '正常'
  if (value === '很吃力') return '困难'
  if (value === '轻松' || value === '正常' || value === '困难' || value === '未记录') return value
  return '未记录'
}
const difficultyMenu = ref<HTMLElement | null>(null)
const difficultyMenuStyle = ref({left:'0px',top:'0px',width:'160px',visibility:'hidden' as 'hidden'|'visible'})
const activeDifficultySet = computed(() => openDifficultyIndex.value === null ? null : firstExercise.value?.sets[openDifficultyIndex.value])
let difficultyAnchor: HTMLElement | null = null
function closeDifficulty(restoreFocus = true) {
  const anchor = difficultyAnchor
  openDifficultyIndex.value = null
  difficultyAnchor = null
  if (restoreFocus) anchor?.focus({preventScroll:true})
}
function positionDifficulty() {
  if (openDifficultyIndex.value === null || !difficultyAnchor || !difficultyMenu.value) return
  const rect = difficultyAnchor.getBoundingClientRect(), form = difficultyAnchor.closest('form')?.getBoundingClientRect()
  if (!difficultyAnchor.isConnected || rect.bottom <= 0 || rect.top >= innerHeight || (form && (rect.bottom <= form.top || rect.top >= form.bottom))) { closeDifficulty(false); return }
  const margin = 8, gap = 4, height = difficultyMenu.value.getBoundingClientRect().height
  const width = Math.min(Math.max(rect.width,160),innerWidth-margin*2)
  const left = Math.max(margin,Math.min(rect.left,innerWidth-width-margin))
  const below = innerHeight-rect.bottom-margin, above = rect.top-margin
  const preferred = below >= height+gap || below >= above ? rect.bottom+gap : rect.top-height-gap
  const top = Math.max(margin,Math.min(preferred,innerHeight-height-margin))
  difficultyMenuStyle.value = {left:left+'px',top:top+'px',width:width+'px',visibility:'visible'}
}
async function showDifficulty(index:number, anchor:HTMLElement, edge?:'first'|'last') {
  if (saving.value) return
  difficultyAnchor = anchor
  openDifficultyIndex.value = index
  difficultyMenuStyle.value.visibility = 'hidden'
  await nextTick()
  positionDifficulty()
  await nextTick()
  const options = difficultyMenu.value?.querySelectorAll<HTMLButtonElement>('[role="option"]')
  const selected = edge === 'first' ? 0 : edge === 'last' ? DIFFICULTY_OPTIONS.length-1 : DIFFICULTY_OPTIONS.findIndex(option=>option.value===normalizeDifficulty(activeDifficultySet.value?.difficulty || ''))
  options?.[Math.max(0,selected)]?.focus({preventScroll:true})
}
function toggleDifficulty(index:number,event:MouseEvent) {
  if (openDifficultyIndex.value === index) closeDifficulty()
  else void showDifficulty(index,event.currentTarget as HTMLElement)
}
function difficultyTriggerKey(event:KeyboardEvent,index:number) {
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
  event.preventDefault();event.stopPropagation()
  void showDifficulty(index,event.currentTarget as HTMLElement,event.key==='ArrowDown'?'first':'last')
}
function difficultyMenuKey(event:KeyboardEvent) {
  if (event.key === 'Escape') {event.preventDefault();event.stopPropagation();closeDifficulty();return}
  if (event.key === 'Tab') {closeDifficulty();return}
  const keys=['ArrowDown','ArrowUp','Home','End'];if (!keys.includes(event.key)) return
  event.preventDefault();event.stopPropagation()
  const options=Array.from(difficultyMenu.value?.querySelectorAll<HTMLButtonElement>('[role="option"]') || [])
  if (!options.length) return
  const current=options.indexOf(document.activeElement as HTMLButtonElement)
  const index=event.key==='Home'?0:event.key==='End'?options.length-1:(current+(event.key==='ArrowDown'?1:options.length-1))%options.length
  options[index]?.focus({preventScroll:true})
}
function difficultyOutside(event:PointerEvent) {
  if (openDifficultyIndex.value===null) return
  const target=event.target as Node
  if (!difficultyMenu.value?.contains(target) && !difficultyAnchor?.contains(target)) closeDifficulty(false)
}
function pickDifficulty(set:WorkoutSet,value:string) {
  set.difficulty=value
  closeDifficulty()
}

let draftKeySeed = 0
function draftKey(prefix: string) { draftKeySeed += 1; return prefix + '-' + draftKeySeed }
function emptySet(): WorkoutSet { return { key: draftKey('set'), weight_kg: 0, reps: 1, difficulty: '刚好', completed: true, note: '' } }
function emptyExercise(): WorkoutExercise { return { key: draftKey('exercise'), exercise_name: '', difficulty: '未记录', note: '', sets: [emptySet()] } }
function emptyDraft(): WorkoutSessionDraft { return { session_date: new Date().toISOString().slice(0, 10), title: '力量训练', note: '', exercises: [emptyExercise()] } }
function withDraftKeys(value: WorkoutSessionDraft): WorkoutSessionDraft {
  return {
    ...value,
    exercises: value.exercises.map((exercise) => ({
      ...exercise,
      key: exercise.key || draftKey('exercise'),
      sets: exercise.sets.map((item) => ({ ...item, key: item.key || draftKey('set'), difficulty: normalizeDifficulty(item.difficulty) })),
    })),
  }
}
const draft = ref<WorkoutSessionDraft>(emptyDraft())

const firstExercise = computed(() => draft.value.exercises[0])
function cardioExercise(exercise: WorkoutExercise): boolean {
  if (exercise.measurement_type) return exercise.measurement_type === 'cardio'
  if (exercise.sets.some(set => set.duration_minutes != null || set.distance_km != null)) return true
  const item = exerciseLibrary.value.find(item => item.name === exercise.exercise_name.trim())
  return item ? typeOfExercise(item) === 'cardio' : exercise === firstExercise.value && exerciseType.value === 'cardio'
}
function defaultCardioDuration(exercise: WorkoutExercise) {
  if (!cardioExercise(exercise)) return
  for (const set of exercise.sets) if (set.duration_minutes == null) set.duration_minutes = 1
}
const cardioMode = computed(() => cardioExercise(firstExercise.value))
const cardioDistance = computed(() => firstExercise.value.sets.reduce((sum,set)=>sum+(Number(set.distance_km)||0),0))
const cardioDuration = computed(() => firstExercise.value.sets.reduce((sum,set)=>sum+(Number(set.duration_minutes)||0),0))

const typedExercises = computed(() => exerciseLibrary.value.filter(item => typeOfExercise(item) === exerciseType.value))
const exerciseCategories = computed(() => ['全部', ...Array.from(new Set(typedExercises.value.map((item) => item.category).filter(Boolean)))])
const matchingExercises = computed(() => {
  const query = exerciseLibrarySearch.value.trim().toLocaleLowerCase()
  return typedExercises.value.filter((item) =>
    (exerciseLibraryCategory.value === '全部' || item.category === exerciseLibraryCategory.value)
    && (!query || `${item.name} ${item.category} ${item.target_muscles}`.toLocaleLowerCase().includes(query)),
  )
})
const quickExerciseQuery = ref(''), quickExerciseTerm = ref('')
const quickExerciseResults = ref<HTMLElement|null>(null)
let quickExerciseTimer:ReturnType<typeof setTimeout>|null=null
function resetQuickExerciseSearch(){if(quickExerciseTimer)clearTimeout(quickExerciseTimer);quickExerciseTimer=null;quickExerciseQuery.value='';quickExerciseTerm.value='';if(quickExerciseResults.value)quickExerciseResults.value.scrollTop=0}
function searchQuickExercises(){if(quickExerciseTimer)clearTimeout(quickExerciseTimer);quickExerciseTimer=null;quickExerciseTerm.value=quickExerciseQuery.value.trim();if(quickExerciseResults.value)quickExerciseResults.value.scrollTop=0}
function queueQuickExerciseSearch(event?:Event){
 if((event as InputEvent|undefined)?.isComposing)return
 if(quickExerciseTimer)clearTimeout(quickExerciseTimer)
 if(!quickExerciseQuery.value.trim()){quickExerciseTerm.value='';quickExerciseTimer=null;if(quickExerciseResults.value)quickExerciseResults.value.scrollTop=0;return}
 quickExerciseTimer=setTimeout(searchQuickExercises,250)
}
function quickExerciseKey(event:KeyboardEvent){
 if(event.key==='Escape'&&(quickExerciseQuery.value||quickExerciseTerm.value)){event.preventDefault();event.stopPropagation();resetQuickExerciseSearch();return}
 if(event.key!=='Enter'||event.isComposing||event.keyCode===229)return
 event.preventDefault();event.stopPropagation();searchQuickExercises()
}
const quickExerciseMatches = computed(()=>{
 const normalize=(value:string)=>String(value||'').replace(/\s+/g,'').toLocaleLowerCase(),term=normalize(quickExerciseTerm.value)
 if(!term)return []
 return typedExercises.value.filter(item=>[item.name,item.category,item.target_muscles].some(value=>normalize(value).includes(term))).sort((a,b)=>Number(normalize(b.name)===term)-Number(normalize(a.name)===term)).slice(0,20)
})
async function selectQuickExercise(item:ExerciseLibraryItem){selectExercise(item);resetQuickExerciseSearch();await nextTick();document.getElementById('workout-exercise-name')?.focus({preventScroll:true})}

const quickExerciseRecommendations = computed(() => typedExercises.value.slice(0,8))
const customExercises = computed(() => matchingExercises.value.filter((item) => item.source === 'custom'))
const recommendedExercises = computed(() => matchingExercises.value.slice(0, 12))

const MUSCLE_MAP: Array<{ keywords: string[]; label: string }> = [
  { keywords: ['卧推', '俯卧撑', '推举'], label: '胸大肌 · 肱三头肌 · 前束' },
  { keywords: ['划船', '下拉', '引体', '硬拉'], label: '背阔肌 · 肱二头肌 · 前臂' },
  { keywords: ['深蹲', '箭步', '臀桥', '高脚杯'], label: '股四头肌 · 臀大肌' },
  { keywords: ['平板', '卷腹', '死虫', '核心'], label: '腹直肌 · 核心肌群' },
]
const muscleLabel = computed(() => {
  const name = firstExercise.value?.exercise_name || ''
  return exerciseLibrary.value.find((item) => item.name === name)?.target_muscles
    || MUSCLE_MAP.find((m) => m.keywords.some((k) => name.includes(k)))?.label
    || ''
})

const estimated1RM = computed(() => {
  const ex = firstExercise.value
  if (!ex || cardioExercise(ex) || !ex.sets.length) return 0
  const maxSet = ex.sets.reduce((max, s) => (Number(s.weight_kg) > Number(max.weight_kg) ? s : max), ex.sets[0])
  const w = Number(maxSet.weight_kg), r = Number(maxSet.reps)
  if (!w || !r) return 0
  return Math.round(w * (1 + r / 30))
})

const weekNumber = computed(() => {
  const d = new Date(draft.value.session_date)
  if (isNaN(d.getTime())) return 0
  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()))
  const dayNum = (date.getUTCDay() + 6) % 7
  date.setUTCDate(date.getUTCDate() - dayNum + 3)
  const firstThursday = new Date(Date.UTC(date.getUTCFullYear(), 0, 4))
  return 1 + Math.round(((date.getTime() - firstThursday.getTime()) / 86400000 - 3 + ((firstThursday.getUTCDay() + 6) % 4)) / 7)
})

const weekdayLabel = computed(() => {
  const d = new Date(draft.value.session_date)
  if (isNaN(d.getTime())) return ''
  return ['周日', '周一', '周二', '周三', '周四', '周五', '周六'][d.getDay()]
})

const totalVolume = computed(() =>
  draft.value.exercises.reduce((sum, ex) => sum + (cardioExercise(ex) ? 0 : ex.sets.reduce((s, set) => s + Number(set.weight_kg) * Number(set.reps), 0)), 0),
)

const originalVolume = ref(0)
const originalSets = ref<WorkoutSet[] | null>(null)
const volumeDiff = computed(() => totalVolume.value - originalVolume.value)
const changedSetHint = computed(() => {
  const ex = firstExercise.value
  if (!ex || cardioMode.value || !originalSets.value) return ''
  for (let i = 0; i < ex.sets.length; i += 1) {
    const cur = ex.sets[i]
    const orig = originalSets.value[i]
    if (!orig) continue
    const dw = Number(cur.weight_kg) - Number(orig.weight_kg)
    if (dw !== 0) return '第' + (i + 1) + '组净增 ' + (dw > 0 ? '+' : '') + dw + 'kg × ' + cur.reps + '次'
  }
  return ''
})
function isSetEdited(index: number): boolean {
  const ex = firstExercise.value
  if (!ex || !originalSets.value || !originalSets.value[index]) return false
  return cardioMode.value
    ? ex.sets[index].distance_km !== originalSets.value[index].distance_km || ex.sets[index].duration_minutes !== originalSets.value[index].duration_minutes
    : Number(ex.sets[index].weight_kg) !== Number(originalSets.value[index].weight_kg)
}

const noteLength = computed(() => (draft.value.note || '').length)

async function loadSessions() {
  loading.value = true
  try {
    const response = await fetch('/api/workout-sessions?user_id=demo-user&limit=50')
    if (!response.ok) throw new Error('load failed')
    sessions.value = await response.json()
  } catch { notice.value = '训练会话加载失败，请确认后端服务正常。' } finally { loading.value = false }
}

function snapshotOriginal() {
  const ex = draft.value.exercises[0]
  originalSets.value = ex ? ex.sets.map((s) => ({ ...s })) : null
  originalVolume.value = draft.value.exercises.reduce((sum, e) => sum + e.sets.reduce((s, set) => s + Number(set.weight_kg) * Number(set.reps), 0), 0)
}

function openManager() {
  previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null
  open.value = true
  createNew()
  void loadSessions()
}
function focusDialog() { void nextTick(() => dialogRef.value?.focus({preventScroll:true})) }
function closeManager() {
  resetQuickExerciseSearch()
  open.value = false
  editorOpen.value = false
  exerciseLibraryOpen.value = false
  openDifficultyIndex.value = null
  clearPendingDeleteConfirmation()
  previouslyFocused?.focus({preventScroll:true})
  previouslyFocused = null
}
function handleKeydown(event: KeyboardEvent) {
  if (!open.value || event.defaultPrevented) return
  if (event.key === 'Escape' && openDifficultyIndex.value !== null) {event.preventDefault();closeDifficulty();return}
  if (event.key === 'Escape') { event.preventDefault(); if (exerciseLibraryOpen.value) { exerciseLibraryOpen.value = false; return }; closeManager() }
}
function createNew() {
  exerciseType.value = 'strength'
  inferExerciseTypeOnLoad = false
  exerciseLibraryCategory.value = '全部'
  resetQuickExerciseSearch()
  editingId.value = null
  clearPendingDeleteConfirmation()
  draft.value = emptyDraft()
  editorOpen.value = true
  notice.value = ''
  snapshotOriginal()
  void loadExerciseLibrary()
  focusDialog()
}
function editSession(session: WorkoutSession) {
  exerciseType.value = 'strength'
  inferExerciseTypeOnLoad = true
  exerciseLibraryCategory.value = '全部'
  resetQuickExerciseSearch()
  editingId.value = session.id
  clearPendingDeleteConfirmation()
  draft.value = withDraftKeys(JSON.parse(JSON.stringify({ session_date: session.session_date, title: session.title, note: session.note, exercises: session.exercises })) as WorkoutSessionDraft)
  if (firstExercise.value.sets.some(set=>set.duration_minutes!=null)) exerciseType.value = 'cardio'
  editorOpen.value = true
  notice.value = ''
  snapshotOriginal()
  void loadExerciseLibrary()
}
function addSet(exercise: WorkoutExercise) {
  const lastSet = exercise.sets[exercise.sets.length - 1]
  exercise.sets.push({ distance_km: lastSet?.distance_km ?? null, duration_minutes: lastSet?.duration_minutes ?? (cardioExercise(exercise) ? 1 : null), key: draftKey('set'), weight_kg: lastSet?.weight_kg || 0, reps: lastSet?.reps || 8, difficulty: lastSet?.difficulty || '刚好', completed: true, note: '' })
}
function removeSet(exercise: WorkoutExercise, index: number) {
  if (exercise.sets.length > 1) exercise.sets.splice(index, 1)
  openDifficultyIndex.value = null
}

async function loadExerciseLibrary() {
  if (exerciseLibraryLoading.value) return
  exerciseLibraryLoading.value = true
  exerciseLibraryError.value = ''
  try {
    const response = await fetch('/api/workout-exercises?user_id=demo-user&limit=300')
    if (!response.ok) throw new Error('动作库加载失败，请确认后端服务正常。')
    const data = await response.json()
    exerciseLibrary.value = Array.isArray(data.exercises) ? data.exercises : []
    if (inferExerciseTypeOnLoad) {
      const selected = exerciseLibrary.value.find(item => item.name === firstExercise.value?.exercise_name)
      if (firstExercise.value.sets.some(set=>set.duration_minutes!=null)) exerciseType.value = 'cardio'
      else if (selected) exerciseType.value = typeOfExercise(selected)
      inferExerciseTypeOnLoad = false
    }
  } catch (error) {
    exerciseLibraryError.value = error instanceof Error ? error.message : '动作库加载失败。'
  } finally {
    exerciseLibraryLoading.value = false
  }
}

function openExerciseLibrary(tab: ExerciseLibraryTab = 'recommended') {
  exerciseLibraryTab.value = tab
  exerciseLibrarySearch.value = ''
  exerciseLibraryCategory.value = '全部'
  exerciseLibraryError.value = ''
  customExerciseName.value = ''
  customExerciseCategory.value = exerciseType.value === 'cardio' ? '有氧' : '自定义'
  customExerciseMuscles.value = ''
  exerciseLibraryOpen.value = true
  void loadExerciseLibrary()
}

function selectExercise(item: ExerciseLibraryItem) {
  inferExerciseTypeOnLoad = false
  exerciseType.value = typeOfExercise(item)
  if (editingId.value === null) draft.value.title = exerciseType.value === 'cardio' ? '有氧训练' : '力量训练'
  if (firstExercise.value) { firstExercise.value.exercise_name = item.name; firstExercise.value.measurement_type = typeOfExercise(item); defaultCardioDuration(firstExercise.value) }
  exerciseLibraryOpen.value = false
  notice.value = ''
}

async function saveCustomExercise() {
  const name = customExerciseName.value.trim()
  if (!name) { exerciseLibraryError.value = '请填写自定义动作名称。'; return }
  if (savingCustomExercise.value) return
  savingCustomExercise.value = true
  exerciseLibraryError.value = ''
  try {
    const response = await fetch('/api/workout-exercises/custom', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: 'demo-user', exercise_name: name, category: customExerciseCategory.value, target_muscles: customExerciseMuscles.value }),
    })
    if (!response.ok) throw new Error(await responseError(response))
    const data = await response.json()
    upsertExerciseMeta(data.exercise || {name, category: customExerciseCategory.value,
      target_muscles: customExerciseMuscles.value, source: 'custom'})
    await loadExerciseLibrary()
    const saved = exerciseLibrary.value.find((item) => item.name === (data.exercise?.name || name))
    if (saved) selectExercise(saved)
    else { exerciseLibraryOpen.value = false; if (firstExercise.value) firstExercise.value.exercise_name = name }
    notify('动作已保存到自定义动作库')
  } catch (error) {
    exerciseLibraryError.value = error instanceof Error ? error.message : '保存自定义动作失败。'
  } finally {
    savingCustomExercise.value = false
  }
}

function validateDraft(): string {
  if (!draft.value.session_date) return '请选择训练日期。'
  if (!draft.value.title.trim()) return '请填写训练名称。'
  const missingExercise = draft.value.exercises.findIndex((exercise) => !exercise.exercise_name.trim())
  if (missingExercise >= 0) return '请填写第 ' + (missingExercise + 1) + ' 个动作名称。'
  for (let exerciseIndex = 0; exerciseIndex < draft.value.exercises.length; exerciseIndex += 1) {
    const exercise = draft.value.exercises[exerciseIndex]
    if (!exercise.sets.length) return '第 ' + (exerciseIndex + 1) + ' 个动作至少需要一组。'
    if (cardioExercise(exercise)) {
      const invalid = exercise.sets.findIndex(item => item.distance_km == null || item.duration_minutes == null || String(item.distance_km) === '' || String(item.duration_minutes) === '' || !Number.isFinite(Number(item.distance_km)) || Number(item.distance_km) < 0 || Number(item.distance_km) > 10000 || !Number.isFinite(Number(item.duration_minutes)) || Number(item.duration_minutes) <= 0 || Number(item.duration_minutes) > 10080)
      if (invalid >= 0) return '请检查第 ' + (exerciseIndex + 1) + ' 个动作第 ' + (invalid + 1) + ' 段的距离（公里，至少0）和时间（分钟，大于0）。'
      continue
    }
    const invalidSet = exercise.sets.findIndex((item) => !Number.isFinite(Number(item.weight_kg)) || Number(item.weight_kg) < 0 || !Number.isInteger(Number(item.reps)) || Number(item.reps) < 1)
    if (invalidSet >= 0) return '请检查第 ' + (exerciseIndex + 1) + ' 个动作的第 ' + (invalidSet + 1) + ' 组重量和次数。'
  }
  return ''
}

async function responseError(response: Response): Promise<string> {
  try {
    const data = await response.json()
    if (typeof data.detail === 'string') return data.detail
    if (Array.isArray(data.detail)) return data.detail.map((item: { msg?: string }) => item.msg || '字段不合法').join('；')
  } catch { /* ignore */ }
  return '训练记录保存失败，请稍后重试。'
}

async function saveSession() {
  const validationError = validateDraft()
  if (validationError) { notice.value = validationError; return }
  if (saving.value) return
  saving.value = true
  try {
    const isCreating = editingId.value === null
    const selected = exerciseLibrary.value.find(item => item.name === firstExercise.value.exercise_name.trim())
    const selectedType = selected ? typeOfExercise(selected) : exerciseType.value
    const title = isCreating ? (selectedType === 'cardio' ? '有氧训练' : '力量训练') : draft.value.title.trim()
    const payload = {
      user_id: 'demo-user', session_date: draft.value.session_date, title, note: draft.value.note.trim(),
      exercises: draft.value.exercises.map((exercise) => ({
        ...(exercise.id ? { id: exercise.id } : {}),
        exercise_name: exercise.exercise_name.trim(), difficulty: toBackendDifficulty(exercise.difficulty), note: exercise.note.trim(),
        sets: exercise.sets.map((item) => ({ ...(item.id ? { id: item.id } : {}), ...(cardioExercise(exercise) ? {weight_kg:0,reps:0,distance_km:Number(item.distance_km),duration_minutes:Number(item.duration_minutes)} : {weight_kg:Number(item.weight_kg),reps:Number(item.reps)}), difficulty: toBackendDifficulty(item.difficulty), completed: Boolean(item.completed), note: item.note.trim() })),
      })),
    }
    const url = isCreating ? '/api/workout-sessions' : '/api/workout-sessions/' + editingId.value + '?user_id=demo-user'
    const response = await fetch(url, { method: isCreating ? 'POST' : 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    if (!response.ok) throw new Error(await responseError(response))
    editorOpen.value = false
    notice.value = isCreating ? '训练记录已保存。' : '训练记录已修改。'
    window.dispatchEvent(new Event(CHANGED_EVENT))
    await loadSessions()
  } catch (error) {
    notice.value = error instanceof Error ? error.message : '训练记录保存失败，请稍后重试。'
  } finally {
    saving.value = false
  }
}

async function confirmDelete() {
  if (editingId.value === null) return
  clearPendingDeleteConfirmation()
  try {
    const response = await fetch('/api/workout-sessions/' + editingId.value + '?user_id=demo-user', { method: 'DELETE' })
    if (!response.ok) throw new Error('delete failed')
    notice.value = ''
    notify('训练记录已删除')
    editorOpen.value = false
    window.dispatchEvent(new Event(CHANGED_EVENT))
    await loadSessions()
  } catch { notice.value = '训练记录删除失败，请稍后重试。' }
}

function handleEditEvent(event: Event) {
  const session = (event as CustomEvent<WorkoutSession>).detail
  if (!session) return
  previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null
  open.value = true
  editSession(session)
  focusDialog()
  void loadSessions()
}

onMounted(() => {
  window.addEventListener(OPEN_EVENT, openManager)
  window.addEventListener(EDIT_EVENT, handleEditEvent)
  window.addEventListener('keydown', handleKeydown)
  document.addEventListener('pointerdown', difficultyOutside)
  window.addEventListener('scroll', positionDifficulty, true)
  window.addEventListener('resize', positionDifficulty)
})
onBeforeUnmount(() => {
  resetQuickExerciseSearch()
  window.removeEventListener(OPEN_EVENT, openManager)
  window.removeEventListener(EDIT_EVENT, handleEditEvent)
  window.removeEventListener('keydown', handleKeydown)
  document.removeEventListener('pointerdown', difficultyOutside)
  window.removeEventListener('scroll', positionDifficulty, true)
  window.removeEventListener('resize', positionDifficulty)
  clearPendingDeleteConfirmation()
  document.body.style.overflow = ''
})
watch(open, (value) => { document.body.style.overflow = value ? 'hidden' : '' })
</script>
<template>
  <div v-if="open" class="workout-session-overlay fixed inset-0 z-50 flex items-center justify-center bg-surface-container-lowest/80 backdrop-blur-md p-4 overflow-hidden" @click.self="closeManager">
    <section ref="dialogRef" :class="{'is-editing':editorOpen}" class="workout-session-dialog w-full max-w-[620px] bg-surface rounded-xl shadow-2xl overflow-hidden flex flex-col outline-none" role="dialog" aria-modal="true" aria-label="编辑训练记录" tabindex="-1">
      <!-- Modal Header -->
      <div class="shrink-0 px-6 pt-5 pb-4 bg-surface-container-low flex flex-col gap-2">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="w-2.5 h-2.5 rounded-full bg-primary shadow-sm animate-pulse"></div>
            <h2 class="font-headline-sm text-headline-sm text-on-surface font-semibold tracking-tight">{{ editingId === null ? '新增训练记录' : '编辑训练记录' }}</h2>
          </div>
          <div class="flex items-center gap-2">
            <span class="font-code-inline text-code-inline px-2 py-0.5 rounded bg-surface-container-highest text-primary font-medium">
              {{ editingId === null ? '新增模式' : `修改模式 · ID: #TR-${String(editingId).padStart(8, '0')}` }}
            </span>
            <button type="button" class="w-7 h-7 rounded-lg bg-surface-container flex items-center justify-center text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors" title="关闭弹窗" @click="closeManager">
              <span class="material-symbols-outlined text-[16px]">close</span>
            </button>
          </div>
        </div>
        <p class="font-body-sm text-body-sm text-on-surface-variant">{{ cardioMode ? '填写每段距离、时间与体感反馈。' : '修改动作参数、逐组数据与体感反馈，系统将同步重算负荷容量与历史极值对比。' }}</p>
        <p v-if="notice" class="font-body-sm text-body-sm text-primary">{{ notice }}</p>
      </div>

      <!-- Modal Body -->
      <form v-if="editorOpen" class="workout-session-form p-6 fz-module-grid flex flex-col overflow-y-auto" novalidate @submit.prevent="saveSession">
        <div class="exercise-type-picker" role="group" aria-label="动作分类">
          <button v-for="kind in EXERCISE_TYPES" :key="kind.id" type="button" class="exercise-type-option" :aria-pressed="exerciseType === kind.id" :disabled="saving || savingCustomExercise" @click="switchExerciseType(kind.id)"><span aria-hidden="true" class="material-symbols-outlined">{{ kind.icon }}</span>{{ kind.label }}</button>
        </div>

        <!-- 动作名称 / 训练日期 -->
        <div class="workout-info-grid grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-xl bg-surface-container-lowest">
          <div class="workout-info-field">
            <div class="workout-field-header">
              <label for="workout-exercise-name">动作名称</label>
              <button type="button" class="inline-flex items-center gap-1 px-2 py-1 rounded-lg border border-primary/30 bg-primary/10 text-primary font-label-sm text-label-sm hover:bg-primary/20 transition-colors" @click="openExerciseLibrary()"><span aria-hidden="true" class="material-symbols-outlined text-[15px]">inventory_2</span>动作库</button>
            </div>
            <div class="workout-field-control bg-surface-container hover:bg-surface-container-high transition-colors">
              <span aria-hidden="true" class="material-symbols-outlined text-[18px] text-primary shrink-0">{{ exerciseType === 'cardio' ? 'directions_run' : 'fitness_center' }}</span>
              <input id="workout-exercise-name" v-model="firstExercise.exercise_name" class="font-body-md text-body-md text-on-surface font-medium placeholder-on-surface-variant/50" :placeholder="exerciseType === 'cardio' ? '例如：跑步、椭圆机' : '杠铃平板卧推'" />
              <button type="button" aria-label="从动作库选择动作" title="从动作库选择" class="workout-library-search text-on-surface-variant hover:bg-primary/10 hover:text-primary" @click="openExerciseLibrary()"><span aria-hidden="true" class="material-symbols-outlined text-[18px]">search</span></button>
            </div>
            <div v-if="muscleLabel || estimated1RM > 0" class="workout-field-meta">
              <span v-if="muscleLabel" class="workout-target-muscles">{{ muscleLabel }}</span>
              <span v-if="estimated1RM > 0" class="workout-estimated-rm text-primary">估算 1RM: {{ estimated1RM }} kg</span>
            </div>
          </div>
          <div class="workout-info-field">
            <div class="workout-field-header"><label for="workout-session-date">训练日期</label></div>
            <div class="workout-field-control bg-surface-container hover:bg-surface-container-high transition-colors">
              <span aria-hidden="true" class="material-symbols-outlined text-[18px] text-on-surface-variant shrink-0">calendar_today</span>
              <input id="workout-session-date" v-model="draft.session_date" type="date" class="font-body-md text-body-md text-on-surface font-medium" />
            </div>
            <div class="workout-field-meta workout-date-meta"><span>第 {{ weekNumber }} 周</span><span>{{ weekdayLabel }}</span></div>
          </div>
        </div>

        <section class="exercise-quick-picks space-y-3" aria-label="常用动作推荐">
          <div role="search" aria-label="搜索常用动作" class="relative">
            <span aria-hidden="true" class="material-symbols-outlined absolute left-3 top-2 text-on-surface-variant text-lg">search</span>
            <input v-model="quickExerciseQuery" type="search" aria-label="搜索常用动作" placeholder="搜索动作名称、部位或肌群…" :disabled="saving" @input="queueQuickExerciseSearch" @compositionend="queueQuickExerciseSearch" @search="queueQuickExerciseSearch" @keydown="quickExerciseKey" class="w-full bg-surface-container-low border border-outline-variant rounded-xl pl-9 pr-20 py-2 text-sm outline-none focus:border-primary">
            <button type="button" @click="searchQuickExercises" :disabled="saving" class="absolute right-1.5 top-1.5 px-2.5 py-1 rounded-lg bg-surface-variant text-xs text-primary">搜索</button>
          </div>
          <h3 class="text-xs font-semibold text-on-surface-variant">{{quickExerciseTerm?'相关动作':'常用动作'}}</h3>
          <div ref="quickExerciseResults" class="exercise-quick-results" :aria-busy="exerciseLibraryLoading">
            <p v-if="exerciseLibraryError && exerciseLibrary.length" role="alert" class="text-xs text-error">{{exerciseLibraryError}} <button type="button" @click="loadExerciseLibrary" :disabled="saving" class="underline">重试</button></p>
          <p v-if="exerciseLibraryLoading && !exerciseLibrary.length" role="status" class="text-xs text-on-surface-variant">正在加载常用动作…</p>
          <p v-else-if="exerciseLibraryError && !exerciseLibrary.length" role="alert" class="text-xs text-error">{{exerciseLibraryError}} <button type="button" @click="loadExerciseLibrary" :disabled="saving" class="underline">重试</button></p>
          <template v-else-if="quickExerciseTerm">
            <div v-if="quickExerciseMatches.length" class="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <button v-for="item in quickExerciseMatches" :key="item.name" type="button" :aria-label="'选用'+item.name" :disabled="saving" @click="selectQuickExercise(item)" class="p-3 rounded-xl border border-outline-variant bg-surface-container-low text-left hover:border-primary/50 hover:text-primary transition-colors">
                <strong class="block text-xs break-words">{{item.name}}</strong><span class="block mt-1 text-[10px] text-on-surface-variant break-words">{{item.category}} · {{item.target_muscles || (item.source==='custom'?'自定义动作':'内置动作')}}</span>
              </button>
            </div>
            <p v-else role="status" class="text-xs text-on-surface-variant">没有找到相关动作。</p>
          </template>
          <div v-else-if="quickExerciseRecommendations.length" class="flex flex-wrap gap-2">
            <button v-for="item in quickExerciseRecommendations" :key="item.name" type="button" :aria-label="'选用'+item.name" :aria-pressed="firstExercise.exercise_name === item.name" :disabled="saving" @click="selectQuickExercise(item)" class="max-w-full px-3 py-1.5 rounded-full border text-xs text-left break-words transition-colors" :class="firstExercise.exercise_name === item.name ? 'bg-primary/10 border-primary/40 text-primary' : 'bg-surface-container-low border-outline-variant text-on-surface-variant hover:border-primary/50 hover:text-primary'">{{item.name}}</button>
          </div>
          <p v-else class="text-xs text-on-surface-variant">暂无常用动作。</p>
          </div>
        </section>

        <!-- 逐组明细 -->
        <div class="flex flex-col gap-2.5">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="font-label-md text-label-md text-on-surface font-semibold">{{ cardioMode ? '分段明细与体感' : '逐组明细与体感' }}</span>
              <span class="px-1.5 py-0.5 rounded bg-surface-container text-on-surface-variant font-code-inline text-code-inline">共 {{ firstExercise.sets.length }} {{ cardioMode ? '段' : '组' }}</span>
            </div>
            <span class="font-label-sm text-label-sm text-on-surface-variant">支持实时微调数值</span>
          </div>

          <div class="overflow-x-auto"><div class="min-w-[280px]">
          <div class="workout-set-grid grid gap-2 px-3 py-2 rounded-lg bg-surface-container-high text-on-surface-variant font-label-sm text-label-sm uppercase tracking-wider">
            <div class="col-span-2 text-left">{{ cardioMode ? '段序' : '组序' }}</div>
            <div class="col-span-3 text-left whitespace-nowrap"><template v-if="cardioMode">距离<span class="block sm:inline normal-case"> (km)</span></template><template v-else>重量 (kg)</template></div>
            <div class="col-span-2 text-left whitespace-nowrap"><template v-if="cardioMode">时间<span class="block sm:inline"> (分钟)</span></template><template v-else>次数 (次)</template></div>
            <div class="col-span-4 text-left">体感反馈</div>
            <div class="col-span-1 text-center">操作</div>
          </div>

          <div class="fz-data-body flex flex-col">
            <div
              v-for="(set, setIndex) in firstExercise.sets"
              :key="set.key"
              class="workout-set-grid fz-data-row grid gap-2 items-center px-3 transition-colors"
            >
              <div class="col-span-2 flex items-center gap-1.5 min-w-0">
                <span :class="['w-5 h-5 rounded-full font-code-inline text-code-inline flex items-center justify-center shrink-0', setIndex === 2 ? 'bg-primary text-on-primary font-bold shadow-sm' : 'bg-surface-container-highest text-on-surface-variant']">{{ setIndex + 1 }}</span>
              </div>
              <div class="col-span-3">
                <div class="relative">
                  <input v-if="cardioMode" v-model.number="set.distance_km" :aria-label="'第'+(setIndex+1)+'段距离（公里）'" type="number" min="0" max="10000" step="0.1" placeholder="0" class="w-full px-2.5 py-1.5 rounded-lg font-body-md text-body-md text-center bg-surface-container text-on-surface focus:bg-surface-container-high" />
                  <input v-else
                    v-model.number="set.weight_kg"
                    type="number" min="0" step="0.5"
                    :class="['w-full px-2.5 py-1.5 rounded-lg font-body-md text-body-md font-medium text-center focus:outline-none', isSetEdited(setIndex) ? 'bg-surface-container-highest text-primary font-semibold shadow-sm' : 'bg-surface-container text-on-surface focus:bg-surface-container-high']"
                  />
                  <span v-if="isSetEdited(setIndex)" class="absolute -top-2 right-1 font-label-sm text-[9px] px-1 rounded bg-primary text-on-primary font-bold">已编辑</span>
                </div>
              </div>
              <div class="col-span-2">
                <input v-if="cardioMode" v-model.number="set.duration_minutes" :aria-label="'第'+(setIndex+1)+'段时间（分钟）'" type="number" min="0.1" max="10080" step="0.1" placeholder="分钟" class="w-full px-2.5 py-1.5 rounded-lg font-body-md text-body-md text-center bg-surface-container text-on-surface focus:bg-surface-container-high" />
                <input v-else v-model.number="set.reps" type="number" min="1" :class="['w-full px-2.5 py-1.5 rounded-lg font-body-md text-body-md font-medium text-center focus:outline-none', isSetEdited(setIndex) ? 'bg-surface-container-highest text-on-surface' : 'bg-surface-container text-on-surface focus:bg-surface-container-high']" />
              </div>
              <div class="col-span-4 relative">
                <button type="button" :class="['w-full px-2.5 py-1.5 rounded-lg flex items-center justify-between font-body-sm text-body-sm cursor-pointer transition-colors', openDifficultyIndex === setIndex ? 'bg-surface-container-high text-on-surface' : 'bg-surface-container text-on-surface hover:bg-surface-container-high']" @click="toggleDifficulty(setIndex,$event)" @keydown="difficultyTriggerKey($event,setIndex)" :disabled="saving" aria-haspopup="listbox" :aria-expanded="openDifficultyIndex === setIndex" :aria-controls="openDifficultyIndex === setIndex ? 'workout-difficulty-options' : undefined" :aria-label="'第'+(setIndex+1)+'组体感反馈，当前'+normalizeDifficulty(set.difficulty)">
                  <span class="whitespace-nowrap">{{ difficultyEmoji(set.difficulty) }} {{ normalizeDifficulty(set.difficulty) }}</span>
                  <span class="material-symbols-outlined text-[14px] text-on-surface-variant">expand_more</span>
                </button>

              </div>
              <div class="col-span-1 flex justify-center">
                <DeleteButton :action="()=>removeSet(firstExercise,setIndex)" :subject="'第 '+(setIndex+1)+' 组'" :disabled="saving" icon-only />
              </div>
            </div>
          </div>

          </div></div>
          <button type="button" class="w-full py-2.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface-variant hover:text-primary font-label-md text-label-md transition-all flex items-center justify-center gap-2" @click="addSet(firstExercise)">
            <span class="material-symbols-outlined text-[16px]">add_circle</span>
            <span>{{ cardioMode ? '添加一段' : '添加一组（沿用上一组重量和次数）' }}</span>
          </button>
        </div>

        <!-- 实时变动卡片 -->
        <div v-if="!cardioMode" class="p-3.5 rounded-xl bg-surface-container-lowest flex items-center justify-between gap-3">
          <div class="flex items-center gap-3 min-w-0">
            <div class="w-8 h-8 rounded-lg bg-surface-container text-primary flex items-center justify-center shrink-0">
              <span class="material-symbols-outlined text-[18px]">sync_alt</span>
            </div>
            <div class="min-w-0">
              <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider block">本次修改变动（实时重算）</span>
              <div class="flex items-center gap-2 mt-0.5">
                <span class="font-body-sm text-body-sm text-on-surface-variant line-through">原总负荷 {{ originalVolume.toFixed(1) }} kg</span>
                <span class="material-symbols-outlined text-[14px] text-primary">arrow_forward</span>
                <span class="font-body-md text-body-md text-on-surface font-semibold">{{ totalVolume.toFixed(1) }} kg</span>
              </div>
            </div>
          </div>
          <div class="text-right shrink-0">
            <span :class="['inline-flex items-center gap-1 px-2 py-0.5 rounded font-code-inline text-code-inline font-semibold', volumeDiff !== 0 ? 'bg-primary/20 text-primary' : 'bg-surface-container text-on-surface-variant']">
              {{ volumeDiff > 0 ? '+' : '' }}{{ volumeDiff.toFixed(1) }} kg {{ volumeDiff === 0 ? '无变化' : '增量' }}
            </span>
            <span v-if="changedSetHint" class="block font-label-sm text-label-sm text-on-surface-variant/80 mt-0.5">{{ changedSetHint }}</span>
          </div>
        </div>

        <div v-if="cardioMode" class="flex flex-wrap gap-4 p-3 rounded-lg bg-primary/5 border border-primary/20 text-on-surface">
          <span>总距离 <strong class="text-primary">{{ cardioDistance.toFixed(2) }} km</strong></span>
          <span>总时间 <strong class="text-primary">{{ cardioDuration.toFixed(1) }} 分钟</strong></span>
        </div>

        <!-- 备注 -->
        <div class="flex flex-col gap-1.5">
          <label class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider flex items-center justify-between">
            <span>动作备注与技巧心得</span>
            <span class="font-code-inline text-code-inline text-on-surface-variant">{{ noteLength }} / 200 字</span>
          </label>
          <textarea v-model="draft.note" rows="2" maxlength="200" class="w-full p-3 rounded-lg bg-surface-container-low text-on-surface font-body-sm text-body-sm placeholder-on-surface-variant/50 focus:outline-none focus:bg-surface-container resize-none transition-colors" :placeholder="cardioMode ? '例如：轻松慢跑，后半程略吃力。' : '例如：握距微调比肩宽 1.5 倍，第 3 组尝试慢速推起，无保护员，感觉良好。'"></textarea>
        </div>

        <!-- Footer -->
        <div class="flex items-center justify-between -mb-1">
          <DeleteButton v-if="editingId !== null" :action="confirmDelete" subject="此条训练记录" label="删除此条记录" :disabled="saving||loading" compact />
          <div v-else></div>
          <div class="flex items-center gap-2.5">
            <button type="button" class="px-4 py-2 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface font-label-md text-label-md transition-colors" @click="closeManager">放弃修改</button>
            <button type="submit" :disabled="saving" class="px-5 py-2 rounded-lg bg-primary hover:bg-primary-container text-on-primary font-label-md text-label-md font-semibold flex items-center gap-2 shadow-sm hover:shadow-sm transition-all">
              <span class="material-symbols-outlined text-[18px]">check</span>
              <span>{{ saving ? '保存中…' : '保存修改记录' }}</span>
            </button>
          </div>
        </div>
      </form>
    </section>
  </div>
  <Teleport to="body">
    <div v-if="exerciseLibraryOpen" class="fixed inset-0 z-[120] flex items-center justify-center bg-scrim/80 backdrop-blur-md p-4" @click.self="exerciseLibraryOpen = false">
      <section class="exercise-library-dialog w-full max-w-[760px] rounded-2xl bg-surface-card border border-outline-variant shadow-2xl text-on-surface flex flex-col overflow-hidden" role="dialog" aria-modal="true" aria-labelledby="exercise-library-title">
        <header class="shrink-0 flex items-start justify-between gap-3 px-5 py-4 border-b border-outline-variant">
          <div class="flex gap-3 min-w-0">
            <div class="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shrink-0"><span class="material-symbols-outlined">fitness_center</span></div>
            <div class="min-w-0">
              <div class="flex flex-wrap items-center gap-2"><h3 id="exercise-library-title" class="text-base font-semibold">选择训练动作</h3><span class="rounded-full bg-primary/10 text-primary text-[10px] px-2 py-0.5 border border-primary/20">本地动作库 · 自定义</span></div>
              <p class="text-[11px] text-on-surface-variant mt-1">选择后填入当前训练记录；推荐结合累计次数和近 30 天使用频率排序</p>
            </div>
          </div>
          <button type="button" aria-label="关闭动作库" class="p-1 text-on-surface-variant hover:text-on-surface" @click="exerciseLibraryOpen = false"><span class="material-symbols-outlined">close</span></button>
        </header>

        <div class="grid grid-cols-3 border-b border-outline-variant shrink-0">
          <button v-for="tab in EXERCISE_LIBRARY_TABS" :key="tab.id" type="button" :aria-pressed="exerciseLibraryTab === tab.id" :class="exerciseLibraryTab === tab.id ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-on-surface-variant hover:text-on-surface'" class="py-3 text-xs flex items-center justify-center gap-1.5 transition-colors" @click="exerciseLibraryTab = tab.id">
            <span class="material-symbols-outlined text-base">{{ tab.icon }}</span>{{ tab.label }}
          </button>
        </div>

        <div class="p-4 border-b border-outline-variant space-y-3 shrink-0">
          <div class="exercise-type-picker" role="group" aria-label="动作库分类">
          <button v-for="kind in EXERCISE_TYPES" :key="kind.id" type="button" class="exercise-type-option" :aria-pressed="exerciseType === kind.id" :disabled="saving || savingCustomExercise" @click="switchExerciseType(kind.id)"><span aria-hidden="true" class="material-symbols-outlined">{{ kind.icon }}</span>{{ kind.label }}</button>
        </div>
          <div class="relative">
            <span class="material-symbols-outlined absolute left-3 top-2.5 text-on-surface-variant text-lg">search</span>
            <input v-model="exerciseLibrarySearch" aria-label="搜索动作库" placeholder="搜索动作名称、部位或肌群…" class="w-full bg-surface-container-low border border-outline-variant rounded-xl pl-9 pr-3 py-2.5 text-sm outline-none focus:border-primary/60">
          </div>
          <div v-if="exerciseLibraryTab === 'all' && exerciseType === 'strength'" class="flex gap-2 overflow-x-auto pb-0.5">
            <button v-for="category in exerciseCategories" :key="category" type="button" :aria-pressed="exerciseLibraryCategory === category" :class="exerciseLibraryCategory === category ? 'bg-primary/15 border-primary/40 text-primary' : 'bg-surface-container-low border-outline-variant text-on-surface-variant'" class="shrink-0 px-3 py-1 rounded-full border text-[11px]" @click="exerciseLibraryCategory = category">{{ category }}</button>
          </div>
          <p v-if="exerciseLibraryError" role="alert" class="text-error text-xs">{{ exerciseLibraryError }}</p>
        </div>

        <div class="exercise-library-results p-4 overflow-y-auto min-h-0">
          <p v-if="exerciseLibraryLoading" class="text-xs text-on-surface-variant text-center py-12">正在读取本地动作库…</p>

          <template v-else-if="exerciseLibraryTab === 'recommended'">
            <div class="flex items-center justify-between gap-2 mb-3"><div><h4 class="text-xs font-semibold text-on-surface">为你推荐</h4><p class="text-[10px] text-on-surface-variant mt-0.5">次数体现长期偏好，活跃天数体现近期频率</p></div><span class="text-[10px] text-on-surface-variant">{{ recommendedExercises.length }} 个</span></div>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <button v-for="(item, index) in recommendedExercises" :key="item.name" type="button" class="text-left p-3 rounded-xl bg-surface-container-low border border-outline-variant hover:border-primary/50 hover:bg-primary/5 transition-colors" @click="selectExercise(item)">
                <div class="flex items-start justify-between gap-2"><div class="min-w-0"><div class="flex items-center gap-2"><span class="text-[10px] text-primary font-code-inline">{{ String(index + 1).padStart(2, '0') }}</span><strong class="text-xs truncate">{{ item.name }}</strong></div><p class="text-[10px] text-on-surface-variant mt-1">{{ item.category }} · {{ item.target_muscles || '自定义动作' }}</p></div><span class="material-symbols-outlined text-primary text-base">add_circle</span></div>
                <div class="flex gap-2 mt-2"><span class="rounded bg-surface-card px-1.5 py-0.5 text-[9px] text-on-surface-variant">使用 {{ item.usage_count }} 次</span><span class="rounded bg-primary/10 px-1.5 py-0.5 text-[9px] text-primary">近30天 {{ item.recent_use_days }} 天 · {{ item.recent_use_count }} 次</span></div>
              </button>
              <p v-if="!recommendedExercises.length" class="sm:col-span-2 text-center text-xs text-on-surface-variant py-12">暂无匹配动作，切换到“全部动作”浏览或创建自定义动作。</p>
            </div>
          </template>

          <template v-else-if="exerciseLibraryTab === 'all'">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <button v-for="item in matchingExercises" :key="item.name" type="button" class="text-left p-3 rounded-xl bg-surface-container-low border border-outline-variant hover:border-primary/50 transition-colors" @click="selectExercise(item)">
                <div class="flex items-center justify-between gap-2"><strong class="text-xs">{{ item.name }}</strong><span class="rounded bg-primary/10 text-primary px-1.5 py-0.5 text-[9px]">{{ item.category }}</span></div>
                <p class="text-[10px] text-on-surface-variant mt-1">{{ item.target_muscles || '自定义动作' }}</p>
                <p class="text-[10px] text-on-surface-variant mt-2">使用 {{ item.usage_count }} 次 · 近30天活跃 {{ item.recent_use_days }} 天</p>
              </button>
              <p v-if="!matchingExercises.length" class="sm:col-span-2 text-center text-xs text-on-surface-variant py-12">没有匹配动作，可以到“自定义动作”中添加。</p>
            </div>
          </template>

          <template v-else>
            <div class="rounded-xl border border-primary/20 bg-primary/5 p-3 mb-4 space-y-3">
              <div><h4 class="text-xs font-semibold text-on-surface">创建个人动作</h4><p class="text-[10px] text-on-surface-variant mt-1">保存后会出现在你的自定义动作库和推荐列表中</p></div>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <label class="text-[10px] text-on-surface-variant">动作名称<input v-model="customExerciseName" maxlength="200" class="w-full mt-1 px-3 py-2 rounded-lg bg-surface-container-low border border-outline-variant text-sm text-on-surface outline-none focus:border-primary/50" placeholder="例如：绳索髋外展"></label>
                <label class="text-[10px] text-on-surface-variant">{{ exerciseType === 'cardio' ? '动作分类' : '训练部位' }}<select v-model="customExerciseCategory" class="w-full mt-1 px-3 py-2 rounded-lg bg-surface-container-low border border-outline-variant text-sm text-on-surface outline-none focus:border-primary/50"><option v-for="category in customExerciseCategories" :key="category" :value="category">{{ category }}</option></select></label>
                <label class="sm:col-span-2 text-[10px] text-on-surface-variant">目标肌群（选填）<input v-model="customExerciseMuscles" maxlength="200" class="w-full mt-1 px-3 py-2 rounded-lg bg-surface-container-low border border-outline-variant text-sm text-on-surface outline-none focus:border-primary/50" placeholder="例如：臀中肌 · 髋外展肌群"></label>
              </div>
              <button type="button" :disabled="savingCustomExercise" class="w-full px-3 py-2 rounded-lg bg-primary text-on-primary text-xs font-semibold disabled:opacity-50" @click="saveCustomExercise">{{ savingCustomExercise ? '保存中…' : '保存并选中动作' }}</button>
            </div>
            <div class="flex items-center justify-between gap-2 mb-3"><div><h4 class="text-xs font-semibold text-on-surface">我的自定义动作</h4><p class="text-[10px] text-on-surface-variant mt-0.5">按累计使用和近期频率推荐</p></div><span class="text-[10px] text-on-surface-variant">{{ customExercises.length }} 个</span></div>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <button v-for="item in customExercises" :key="item.name" type="button" class="text-left p-3 rounded-xl bg-surface-container-low border border-outline-variant hover:border-primary/50 transition-colors" @click="selectExercise(item)">
                <div class="flex justify-between gap-2"><strong class="text-xs">{{ item.name }}</strong><span class="text-[10px] text-primary">{{ item.category }}</span></div>
                <p class="text-[10px] text-on-surface-variant mt-1">{{ item.target_muscles || '未填写目标肌群' }}</p>
                <p class="text-[10px] text-on-surface-variant mt-1">使用 {{ item.usage_count }} 次 · 近30天活跃 {{ item.recent_use_days }} 天</p>
              </button>
              <p v-if="!customExercises.length" class="sm:col-span-2 text-center text-xs text-on-surface-variant py-6">暂无匹配的自定义动作。</p>
            </div>
          </template>
        </div>

        <footer class="px-5 py-3 border-t border-outline-variant flex items-center justify-between gap-3 shrink-0"><span class="text-[10px] text-on-surface-variant">点击动作即可填入当前训练记录</span><button type="button" class="px-3 py-1.5 rounded-lg border border-outline-variant text-xs text-on-surface hover:text-on-surface" @click="exerciseLibraryOpen = false">取消</button></footer>
      </section>
    </div>
  </Teleport>
  <Teleport to="body">
    <div v-if="openDifficultyIndex !== null && activeDifficultySet" ref="difficultyMenu" id="workout-difficulty-options" role="listbox" :aria-label="'第'+(openDifficultyIndex+1)+'组体感反馈'" :style="difficultyMenuStyle" @keydown="difficultyMenuKey" class="fixed z-[200] rounded-lg bg-surface-container-highest shadow-xl overflow-hidden border border-outline-variant/40">
      <button v-for="opt in DIFFICULTY_OPTIONS" :key="opt.value" type="button" role="option" :aria-label="opt.value" :aria-selected="normalizeDifficulty(activeDifficultySet.difficulty) === opt.value" :disabled="saving" class="w-full min-h-[40px] px-2.5 py-1.5 flex items-center gap-2 font-body-sm text-body-sm text-on-surface hover:bg-primary/20 hover:text-primary focus-visible:bg-primary/20 focus-visible:text-primary transition-colors text-left" @click="pickDifficulty(activeDifficultySet,opt.value)"><span aria-hidden="true">{{opt.emoji}}</span>{{opt.value}}</button>
    </div>
  </Teleport>
</template>

<style scoped>
.exercise-type-picker{display:inline-flex;align-self:flex-start;max-width:100%;gap:4px;padding:4px;border:1px solid var(--fz-outline-variant);border-radius:999px;background:var(--fz-surface-container-low)}
.exercise-type-option{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-width:104px;min-height:36px;padding:6px 20px;border:0;border-radius:999px;color:var(--fz-on-surface-variant);background:transparent;font-size:14px;font-weight:600;cursor:pointer;transition:background-color .15s,color .15s}
.exercise-type-option .material-symbols-outlined{font-size:18px}
.exercise-type-option:hover:not(:disabled){background:var(--fz-surface-container-high);color:var(--fz-on-surface)}
.exercise-type-option[aria-pressed="true"],.exercise-type-option[aria-pressed="true"]:hover{background:var(--fz-primary);color:var(--fz-on-primary)}
.exercise-type-option:focus-visible{outline:2px solid var(--fz-primary);outline-offset:3px}
.exercise-type-option:disabled{opacity:.5;cursor:not-allowed}
@media(prefers-reduced-motion:reduce){.exercise-type-option{transition:none}}

.exercise-quick-results{height:112px;overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable}
@media(max-width:639px){.exercise-quick-results{height:160px}}

.workout-session-dialog{max-height:calc(100dvh - 2rem);min-height:0}
.workout-session-dialog.is-editing{height:min(860px,92dvh)}
.workout-session-form{flex:1;min-height:0;overscroll-behavior:contain;scrollbar-gutter:stable}
.workout-session-form>*{flex-shrink:0}

.exercise-library-dialog{height:min(760px,90dvh);max-height:calc(100dvh - 2rem);min-height:0}
.exercise-library-results{flex:1;overscroll-behavior:contain;scrollbar-gutter:stable}


.workout-info-field{display:flex;flex-direction:column;gap:8px;min-width:0}
.workout-field-header{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:28px;font-size:12px;line-height:20px;color:var(--fz-on-surface-variant);font-weight:500}
.workout-field-header button{flex-shrink:0;white-space:nowrap}
.workout-field-control{display:flex;align-items:center;gap:8px;height:44px;min-width:0;padding:0 12px;border-radius:10px}
.workout-field-control:focus-within{outline:1px solid var(--fz-primary);outline-offset:2px}
.workout-field-control input{flex:1;width:100%;min-width:0;height:100%;padding:0;border:0;background:transparent;outline:none;color:var(--fz-on-surface)}
.workout-library-search{display:flex;align-items:center;justify-content:center;flex-shrink:0;width:28px;height:28px;border-radius:6px}
.workout-field-meta{display:flex;align-items:baseline;flex-wrap:wrap;gap:4px 8px;min-height:20px;font-size:12px;line-height:20px;color:var(--fz-on-surface-variant)}
.workout-target-muscles{flex:1 1 120px;min-width:0;overflow-wrap:anywhere}
.workout-estimated-rm{margin-left:auto;white-space:nowrap;font-weight:500}
.workout-date-meta{justify-content:space-between}

.workout-set-grid{grid-template-columns:24px minmax(40px,1fr) minmax(32px,.7fr) minmax(100px,1.4fr) 32px}
.workout-set-grid>div{grid-column:auto}

input[type="date"]::-webkit-calendar-picker-indicator {
  filter: none;
  cursor: pointer;
  opacity: 0.55;
}
input[type="date"]::-webkit-calendar-picker-indicator:hover {
  opacity: 1;
}
</style>
