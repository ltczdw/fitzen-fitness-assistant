export const DELETE_CONFIRMATION_DELAY = 2500

export function createDeleteConfirmation(
  schedule: (callback: () => void, delay: number) => unknown = setTimeout,
  cancel: (timer: any) => void = clearTimeout,
) {
  let active: symbol | null = null
  let timer: unknown
  const listeners = new Set<() => void>()
  const publish = () => listeners.forEach(listener => listener())
  function clear(id?: symbol) {
    if (id !== undefined && active !== id) return
    cancel(timer)
    timer = undefined
    active = null
    publish()
  }
  return {
    isArmed: (id: symbol) => active === id,
    subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener) } },
    clear,
    activate(id: symbol) {
      if (active === id) { clear(id); return true }
      cancel(timer)
      active = id
      timer = schedule(() => clear(id), DELETE_CONFIRMATION_DELAY)
      publish()
      return false
    },
  }
}

export const deleteConfirmation = createDeleteConfirmation()

export function shouldResetDeleteConfirmation(event: {type: string; key?: string; target?: any}) {
  const onDeleteButton = Boolean(event.target?.closest?.('[data-delete-confirm]'))
  if (event.type === 'click' && onDeleteButton) return false
  if (event.type === 'keydown' && onDeleteButton && ['Enter', ' '].includes(event.key || '')) return false
  return true
}

let users = 0
function resetOnOtherAction(event: Event) {
  if (shouldResetDeleteConfirmation(event)) deleteConfirmation.clear()
}
export function bindDeleteConfirmationEvents() {
  if (users++ === 0) for (const type of ['click', 'input', 'keydown']) document.addEventListener(type, resetOnOtherAction, true)
  return () => {
    if (--users === 0) {
      for (const type of ['click', 'input', 'keydown']) document.removeEventListener(type, resetOnOtherAction, true)
      deleteConfirmation.clear()
    }
  }
}
