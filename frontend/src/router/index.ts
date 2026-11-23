/**
 * 路由配置：/login 公开，其余页面均需鉴权；未登录统一重定向 /login。
 */
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { useUserStore } from '@/stores/user'

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/plaza' },
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/LoginView.vue'),
    meta: { public: true },
  },
  {
    path: '/plaza',
    name: 'plaza',
    component: () => import('@/views/PlazaView.vue'),
  },
  {
    path: '/study/:roomId',
    name: 'study',
    component: () => import('@/views/StudyRoomView.vue'),
  },
  {
    path: '/watch/:roomId',
    name: 'watch',
    component: () => import('@/views/WatchRoomView.vue'),
  },
  {
    path: '/profile',
    name: 'profile',
    component: () => import('@/views/ProfileView.vue'),
  },
  { path: '/:pathMatch(.*)*', redirect: '/plaza' },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 全局鉴权守卫：公开页放行，私有页未登录跳 /login 并携带回跳地址
router.beforeEach((to) => {
  const userStore = useUserStore()
  if (!to.meta.public && !userStore.isLoggedIn) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  if (to.name === 'login' && userStore.isLoggedIn) {
    return { name: 'plaza' }
  }
  return true
})

export default router
