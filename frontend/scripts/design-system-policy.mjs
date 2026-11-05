// Component inventory verified by check-design-system.mjs.
export const uiComponents = [
  'ChatMessages', 'DeleteButton', 'EnergyBalanceChart', 'FoodModal', 'NutritionEditModal',
  'RecordsDrawer', 'WorkoutRecordManager', 'WorkoutSessionManager',
]
export const pageContainers = [
  'Chat', 'Dashboard', 'Nutrition', 'NutritionModal', 'Profile',
  'Settings', 'Stats', 'Status', 'Training',
].map(name => `src/fitzen/screens/${name}.vue`)
export const approvedVueFiles = [
  'src/fitzen/App.vue', ...pageContainers,
  ...uiComponents.map(name => `src/components/ui/${name}.vue`),
]
export const tokenSource = 'src/fitzen/tokens.css'
