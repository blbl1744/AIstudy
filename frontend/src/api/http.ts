/**
 * HTTP 通信层：Axios 实例 + 请求/响应拦截器 + 各业务接口封装。
 * 请求自动附加 Bearer Token；收到 401 时清除登录态并跳转 /login（登录接口自身的 401 除外）。
 */
import axios, { type AxiosError, type AxiosRequestConfig } from 'axios'
import type {
  CreateRoomRequest,
  LoginRequest,
  LoginResponse,
  PlazaRoom,
  RegisterRequest,
  RegisterResponse,
  RoomDetail,
  RoomMeta,
  SceneConfig,
  SessionDetail,
  UserProfile,
} from '@/types'

// Token 持久化键名，与 userStore 共享
export const TOKEN_KEY = 'aistudy_token'

/** 读取本地 Token */
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

/** 写入本地 Token */
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

/** 清除本地 Token */
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

const http = axios.create({
  baseURL: '/api',
  timeout: 15000,
})

// 请求拦截：自动附加 Bearer Token
http.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截：非登录接口的 401 统一清理登录态并跳转登录页
http.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const status = error.response?.status
    const url = (error.config as AxiosRequestConfig | undefined)?.url ?? ''
    const isLoginRequest = url.includes('/auth/login')
    if (status === 401 && !isLoginRequest) {
      clearToken()
      if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  },
)

/** 认证与个人成长接口 */
export const authApi = {
  register: (payload: RegisterRequest): Promise<RegisterResponse> =>
    http.post<RegisterResponse>('/auth/register', payload).then((res) => res.data),
  login: (payload: LoginRequest): Promise<LoginResponse> =>
    http.post<LoginResponse>('/auth/login', payload).then((res) => res.data),
  getProfile: (): Promise<UserProfile> =>
    http.get<UserProfile>('/users/me/profile').then((res) => res.data),
  deleteMyData: (): Promise<{ message: string }> =>
    http.delete<{ message: string }>('/users/me/data').then((res) => res.data),
}

/** AI 剧本编译接口 */
export const sceneApi = {
  presets: (): Promise<{ presets: SceneConfig[] }> =>
    http.get<{ presets: SceneConfig[] }>('/scenes/presets').then((res) => res.data),
  compile: (workName: string): Promise<SceneConfig> =>
    http.post<SceneConfig>('/scenes/compile', { work_name: workName }).then((res) => res.data),
}

/** 房间与公告广场接口 */
export const roomApi = {
  create: (payload: CreateRoomRequest): Promise<RoomDetail> =>
    http.post<RoomDetail>('/rooms', payload).then((res) => res.data),
  plaza: (): Promise<{ rooms: PlazaRoom[] }> =>
    http.get<{ rooms: PlazaRoom[] }>('/rooms/plaza').then((res) => res.data),
  /** 房间元信息：进入监督页前校验房主归属，不存在返回 404 */
  meta: (roomId: string): Promise<RoomMeta> =>
    http.get<RoomMeta>(`/rooms/${encodeURIComponent(roomId)}`).then((res) => res.data),
  /** 房主实时快照地址（受鉴权保护，img 标签直接引用即可随请求带 Token 需另行处理） */
  snapshotUrl: (roomId: string): string => `/api/rooms/${roomId}/snapshot.jpg`,
}

/** 战报明细与 CSV 导出接口（M5） */
export const statsApi = {
  /** 单局战报详情：摘要 + 专注度采样点 */
  sessionDetail: (sessionId: number): Promise<SessionDetail> =>
    http.get<SessionDetail>('/stats/sessions/' + sessionId).then((res) => res.data),
  /** 导出全部战报 CSV：以 Blob 返回，http 实例自动附加 Bearer Token */
  exportCsv: (): Promise<Blob> =>
    http.get<Blob>('/stats/export.csv', { responseType: 'blob' }).then((res) => res.data),
}

export default http
