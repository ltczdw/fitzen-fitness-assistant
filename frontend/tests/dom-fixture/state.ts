import {reactive} from 'vue'
export type RecordItem = any
export const rows = reactive([{id: 42, title: '合成训练42', date: '2026-10-03', session_date: '2026-10-03', exercises: [{exercise_name: '卧推', sets: [{weight_kg: 60, reps: 8}]}]}])
export const evidence = reactive({requests: [] as string[], notifications: [] as string[], fail: false})
export const ui = reactive({records: true, recordsTab: 'training', recordSearch: '', recordPage: 1, nutrition: []})
export async function api(url: string, init: any) {
  evidence.requests.push(`${init?.method || 'GET'} ${url}`)
  if (evidence.fail) throw new Error('fixture offline')
  rows.splice(0)
  return {deleted: true}
}
export function useFitzen() {
  return {ui, refresh: async () => {}, openFood: () => {}, openWorkout: () => {}, exportRecords: () => {}, notify: (message: string) => evidence.notifications.push(message)}
}
export function resetFixture() {
  ui.records = false
  evidence.requests.splice(0); evidence.notifications.splice(0); evidence.fail = false
  rows.splice(0, rows.length, {id: 42, title: '合成训练42', date: '2026-10-03', session_date: '2026-10-03', exercises: [{exercise_name: '卧推', sets: [{weight_kg: 60, reps: 8}]}]})
}
