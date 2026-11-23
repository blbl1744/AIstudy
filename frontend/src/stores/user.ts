/**
 * 用户状态仓库：Token 本地持久化、成长字段（军衔/护盾/分钟数）与登录注册拉资料。
 */
import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { authApi, clearToken, getToken, setToken } from '@/api/http'
import type { LoginRequest, LoginResponse, RegisterRequest, RegisterResponse, UserProfile } from '@/types'

export const useUserStore = defineStore('user', () => {
  // ===== 状态 =====
  const token = ref<string | null>(getToken())
  const profile = ref<UserProfile | null>(null)

  // ===== 派生：登录态与成长字段 =====
  const isLoggedIn = computed(() => Boolean(token.value))
  const nickname = computed(() => profile.value?.nickname ?? '')
  const rankTitle = computed(() => profile.value?.rank_title ?? '见习学员')
  const shieldCount = computed(() => profile.value?.shield_count ?? 0)
  const totalFocusMinutes = computed(() => profile.value?.total_focus_minutes ?? 0)
  const streakCleanSessions = computed(() => profile.value?.streak_clean_sessions ?? 0)

  /** 登录：写 Token 并拉取个人档案 */
  async function login(payload: LoginRequest): Promise<LoginResponse> {
    const res = await authApi.login(payload)
    setToken(res.access_token)
    token.value = res.access_token
    await fetchProfile()
    return res
  }

  /** 注册：仅提交表单，成功后调用方自行跳转登录 */
  async function register(payload: RegisterRequest): Promise<RegisterResponse> {
    return await authApi.register(payload)
  }

  /** 拉取并缓存个人档案 */
  async function fetchProfile(): Promise<UserProfile> {
    const data = await authApi.getProfile()
    profile.value = data
    return data
  }

  /** 一键清空个人学习数据后刷新档案 */
  async function clearMyData(): Promise<void> {
    await authApi.deleteMyData()
    await fetchProfile()
  }

  /** 退出登录：清 Token 与档案 */
  function logout(): void {
    clearToken()
    token.value = null
    profile.value = null
  }

  // 启动时若已有 Token，后台恢复档案（401 由响应拦截器统一跳登录）
  if (token.value) {
    void fetchProfile().catch(() => {
      // 静默失败：拦截器已处理 401，此处无需额外提示
    })
  }

  return {
    token,
    profile,
    isLoggedIn,
    nickname,
    rankTitle,
    shieldCount,
    totalFocusMinutes,
    streakCleanSessions,
    login,
    register,
    fetchProfile,
    clearMyData,
    logout,
  }
})
