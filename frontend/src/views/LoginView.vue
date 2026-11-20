<script setup lang="ts">
/**
 * 登录 / 注册页：暗色科幻 + 战地指挥风格切换卡片。
 * 注册表单含用户名、昵称、密码校验；登录成功跳转 /plaza（或路由携带的回跳地址）。
 * 全程使用 Naive UI 表单组件，未使用任何 v-html。
 */
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  useMessage,
  type FormInst,
  type FormRules,
} from 'naive-ui'
import { useUserStore } from '@/stores/user'
import type { LoginRequest, RegisterRequest } from '@/types'

const route = useRoute()
const router = useRouter()
const message = useMessage()
const userStore = useUserStore()

// 当前面板：登录 / 注册
const mode = ref<'login' | 'register'>('login')
const loading = ref(false)

const loginFormRef = ref<FormInst | null>(null)
const registerFormRef = ref<FormInst | null>(null)

const loginModel = reactive<LoginRequest>({ username: '', password: '' })
const registerModel = reactive<RegisterRequest>({ username: '', nickname: '', password: '' })

// 登录表单校验规则
const loginRules: FormRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
  ],
}

// 注册表单校验规则：与后端 RegisterRequest 的长度约束保持一致
const registerRules: FormRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 32, message: '用户名长度需为 3~32 位', trigger: 'blur' },
  ],
  nickname: [
    { required: true, message: '请输入昵称', trigger: 'blur' },
    { min: 1, max: 32, message: '昵称长度需为 1~32 位', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 64, message: '密码长度需为 6~64 位', trigger: 'blur' },
  ],
}

/** 从 Axios 错误中提取后端 detail 文案 */
function extractError(err: unknown): string {
  const e = err as { response?: { data?: { detail?: string } }; message?: string }
  return e.response?.data?.detail ?? e.message ?? '操作失败，请稍后重试'
}

/** 提交登录 */
async function handleLogin(): Promise<void> {
  try {
    await loginFormRef.value?.validate()
  } catch {
    return
  }
  loading.value = true
  try {
    await userStore.login({ ...loginModel })
    message.success('登录成功，欢迎归队')
    const redirect = route.query.redirect
    router.push(typeof redirect === 'string' ? redirect : '/plaza')
  } catch (err) {
    message.error(extractError(err))
  } finally {
    loading.value = false
  }
}

/** 提交注册：成功后切回登录面板并回填用户名 */
async function handleRegister(): Promise<void> {
  try {
    await registerFormRef.value?.validate()
  } catch {
    return
  }
  loading.value = true
  try {
    await userStore.register({ ...registerModel })
    message.success('注册成功，请登录')
    loginModel.username = registerModel.username
    loginModel.password = ''
    mode.value = 'login'
  } catch (err) {
    message.error(extractError(err))
  } finally {
    loading.value = false
  }
}

/** 切换面板 */
function switchMode(next: 'login' | 'register'): void {
  mode.value = next
}
</script>

<template>
  <div class="login-page">
    <div class="grid-bg" aria-hidden="true"></div>
    <div class="scanline" aria-hidden="true"></div>

    <div class="login-shell">
      <header class="hero">
        <div class="hero-badge">TACTICAL FOCUS SYSTEM · M3-A</div>
        <h1 class="hero-title">沉浸式督学指挥终端</h1>
        <p class="hero-subtitle">登录你的作战档案，进入专注战场</p>
      </header>

      <section class="panel-card">
        <div class="panel-tabs">
          <button
            type="button"
            class="panel-tab"
            :class="{ active: mode === 'login' }"
            @click="switchMode('login')"
          >
            作战登录
          </button>
          <button
            type="button"
            class="panel-tab"
            :class="{ active: mode === 'register' }"
            @click="switchMode('register')"
          >
            新兵注册
          </button>
        </div>

        <!-- 登录表单 -->
        <n-form
          v-if="mode === 'login'"
          ref="loginFormRef"
          :model="loginModel"
          :rules="loginRules"
          label-placement="top"
          class="auth-form"
        >
          <n-form-item label="用户名" path="username">
            <n-input
              v-model:value="loginModel.username"
              placeholder="输入用户名"
              size="large"
              clearable
              @keyup.enter="handleLogin"
            />
          </n-form-item>
          <n-form-item label="密码" path="password">
            <n-input
              v-model:value="loginModel.password"
              type="password"
              placeholder="输入密码"
              size="large"
              show-password-on="click"
              @keyup.enter="handleLogin"
            />
          </n-form-item>
          <n-button
            type="primary"
            size="large"
            block
            :loading="loading"
            class="submit-btn"
            @click="handleLogin"
          >
            进入指挥终端
          </n-button>
        </n-form>

        <!-- 注册表单 -->
        <n-form
          v-else
          ref="registerFormRef"
          :model="registerModel"
          :rules="registerRules"
          label-placement="top"
          class="auth-form"
        >
          <n-form-item label="用户名" path="username">
            <n-input
              v-model:value="registerModel.username"
              placeholder="3~32 位用户名"
              size="large"
              clearable
            />
          </n-form-item>
          <n-form-item label="昵称" path="nickname">
            <n-input
              v-model:value="registerModel.nickname"
              placeholder="战场代号（1~32 位）"
              size="large"
              clearable
            />
          </n-form-item>
          <n-form-item label="密码" path="password">
            <n-input
              v-model:value="registerModel.password"
              type="password"
              placeholder="6~64 位密码"
              size="large"
              show-password-on="click"
              @keyup.enter="handleRegister"
            />
          </n-form-item>
          <n-button
            type="primary"
            size="large"
            block
            :loading="loading"
            class="submit-btn"
            @click="handleRegister"
          >
            建立作战档案
          </n-button>
        </n-form>

        <p class="panel-footer">
          {{ mode === 'login' ? '尚无档案？' : '已有档案？' }}
          <button type="button" class="link-btn" @click="switchMode(mode === 'login' ? 'register' : 'login')">
            {{ mode === 'login' ? '立即注册' : '返回登录' }}
          </button>
        </p>
      </section>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background:
    radial-gradient(circle at 20% 15%, rgba(0, 229, 255, 0.12), transparent 45%),
    radial-gradient(circle at 85% 85%, rgba(255, 42, 85, 0.1), transparent 45%),
    linear-gradient(135deg, #050b14 0%, #0d1f33 100%);
}

.grid-bg {
  position: absolute;
  inset: -50%;
  background-image:
    linear-gradient(rgba(0, 229, 255, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 229, 255, 0.06) 1px, transparent 1px);
  background-size: 44px 44px;
  transform: perspective(600px) rotateX(58deg);
  pointer-events: none;
}

.scanline {
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    to bottom,
    rgba(255, 255, 255, 0.018) 0,
    rgba(255, 255, 255, 0.018) 1px,
    transparent 1px,
    transparent 4px
  );
  pointer-events: none;
}

.login-shell {
  position: relative;
  z-index: 1;
  width: min(92vw, 440px);
}

.hero {
  text-align: center;
  margin-bottom: 26px;
}

.hero-badge {
  display: inline-block;
  padding: 4px 12px;
  border: 1px solid rgba(0, 229, 255, 0.4);
  border-radius: 999px;
  color: #00e5ff;
  font-size: 11px;
  letter-spacing: 2px;
  margin-bottom: 14px;
}

.hero-title {
  margin: 0;
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 2px;
  color: #e6f1ff;
  text-shadow: 0 0 18px rgba(0, 229, 255, 0.35);
}

.hero-subtitle {
  margin: 10px 0 0;
  font-size: 13px;
  color: #8aa4c0;
}

.panel-card {
  padding: 26px 26px 20px;
  background: rgba(10, 24, 40, 0.86);
  border: 1px solid rgba(0, 229, 255, 0.28);
  border-radius: 12px;
  box-shadow:
    0 0 0 1px rgba(0, 229, 255, 0.06),
    0 18px 60px rgba(0, 0, 0, 0.55),
    inset 0 0 40px rgba(0, 229, 255, 0.04);
  backdrop-filter: blur(6px);
}

.panel-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 22px;
  border-bottom: 1px solid rgba(138, 164, 192, 0.18);
}

.panel-tab {
  flex: 1;
  padding: 10px 0;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  color: #8aa4c0;
  font-size: 15px;
  letter-spacing: 1px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.panel-tab:hover {
  color: #cfe6ff;
}

.panel-tab.active {
  color: #00e5ff;
  border-bottom-color: #00e5ff;
  text-shadow: 0 0 12px rgba(0, 229, 255, 0.5);
}

.auth-form {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.submit-btn {
  margin-top: 8px;
  letter-spacing: 2px;
}

.panel-footer {
  margin: 18px 0 0;
  text-align: center;
  font-size: 13px;
  color: #8aa4c0;
}

.link-btn {
  background: transparent;
  border: none;
  color: #00e5ff;
  cursor: pointer;
  font-size: 13px;
  padding: 0 2px;
}

.link-btn:hover {
  text-decoration: underline;
}
</style>
