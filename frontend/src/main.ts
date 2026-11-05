import {createApp} from 'vue'
import {createPinia} from 'pinia'
import App from './fitzen/App.vue'
import './fitzen/fitzen.css'
createApp(App).use(createPinia()).mount('#app')
