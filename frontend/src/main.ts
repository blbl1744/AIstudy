import { createApp } from 'vue'
import { createPinia } from 'pinia'
import naive from 'naive-ui'
import Particles from '@tsparticles/vue3'
import { loadSlim } from '@tsparticles/slim'
import App from './App.vue'
import router from './router'
import './style.css'

// 应用入口：挂载 Pinia 状态管理与 Vue Router 路由
const app = createApp(App)
app.use(createPinia())
app.use(router)
// 全局注册 Naive UI 组件：登录/广场/督学房/个人档案页的表单与弹窗均依赖此注册
app.use(naive)
// 注册粒子引擎插件：使用 slim 轻量包，避免拉取完整体积
app.use(Particles, { init: (engine) => loadSlim(engine) })
app.mount('#app')
