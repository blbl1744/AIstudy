<script setup lang="ts">
/**
 * 督学公告广场（M3-A 基础版）：展示公开活跃房间列表与用户成长入口。
 * 完整开房配置台与实时状态将在 M4 联调阶段补齐。
 */
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { roomApi } from '@/api/http'
import { useUserStore } from '@/stores/user'
import type { PlazaRoom, RoomMode } from '@/types'

const router = useRouter()
const userStore = useUserStore()
const rooms = ref<PlazaRoom[]>([])
const loading = ref(true)
const loadError = ref('')

/** 房间模式中文Label */
function modeLabel(mode: RoomMode): string {
  const map: Record<RoomMode, string> = {
    ai_solo: 'AI 剧本督学',
    peer_1v1: '1v1 互督',
    peer_group: '多人围观',
  }
  return map[mode] ?? mode
}

/** 加载广场房间列表 */
async function loadPlaza(): Promise<void> {
  loading.value = true
  loadError.value = ''
  try {
    const res = await roomApi.plaza()
    rooms.value = res.rooms
  } catch {
    loadError.value = '广场数据加载失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

/** 进入房间（占位跳转，实际联调在 M4 完成） */
function enterRoom(room: PlazaRoom): void {
  router.push({ name: room.mode === 'ai_solo' ? 'study' : 'watch', params: { roomId: room.room_id } })
}

/** 退出登录 */
function logout(): void {
  userStore.logout()
  router.push('/login')
}

onMounted(loadPlaza)
</script>

<template>
  <div class="plaza-page">
    <header class="topbar">
      <div class="brand">督学公告广场 <span class="version">M3-A</span></div>
      <nav class="nav-actions">
        <span class="user-chip">{{ userStore.nickname }} · {{ userStore.rankTitle }}</span>
        <button type="button" class="nav-btn" @click="router.push('/profile')">成长档案</button>
        <button type="button" class="nav-btn" @click="logout">退出登录</button>
      </nav>
    </header>

    <main class="plaza-main">
      <div class="section-head">
        <h2>活跃房间</h2>
        <button type="button" class="refresh-btn" @click="loadPlaza">刷新</button>
      </div>

      <p v-if="loading" class="hint">正在扫描广场信号...</p>
      <p v-else-if="loadError" class="error-hint">{{ loadError }}</p>
      <p v-else-if="rooms.length === 0" class="hint">当前暂无公开房间，成为第一个开房的人吧。</p>

      <div v-else class="room-grid">
        <article v-for="room in rooms" :key="room.room_id" class="room-card">
          <div class="room-mode">{{ modeLabel(room.mode) }}</div>
          <h3 class="room-title">{{ room.scene_title }}</h3>
          <p class="room-goal">{{ room.goal_text }}</p>
          <div class="room-meta">
            <span>房主：{{ room.host_nickname }}</span>
            <span>已坚持 {{ room.elapsed_seconds }} 秒</span>
            <span>围观 {{ room.watcher_count }}/{{ room.max_watchers }}</span>
          </div>
          <button type="button" class="enter-btn" @click="enterRoom(room)">进入房间</button>
        </article>
      </div>
    </main>
  </div>
</template>

<style scoped>
.plaza-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #050b14 0%, #0d1f33 100%);
  color: #e6f1ff;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 32px;
  border-bottom: 1px solid rgba(0, 229, 255, 0.2);
  background: rgba(5, 11, 20, 0.7);
}

.brand {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 1px;
  color: #00e5ff;
}

.version {
  font-size: 11px;
  color: #8aa4c0;
  margin-left: 8px;
}

.nav-actions {
  display: flex;
  align-items: center;
  gap: 14px;
}

.user-chip {
  font-size: 13px;
  color: #8aa4c0;
}

.nav-btn,
.refresh-btn,
.enter-btn {
  background: rgba(0, 229, 255, 0.08);
  border: 1px solid rgba(0, 229, 255, 0.35);
  color: #00e5ff;
  border-radius: 6px;
  padding: 6px 14px;
  cursor: pointer;
  font-size: 13px;
}

.nav-btn:hover,
.refresh-btn:hover,
.enter-btn:hover {
  background: rgba(0, 229, 255, 0.18);
}

.plaza-main {
  max-width: 1080px;
  margin: 0 auto;
  padding: 32px 24px;
}

.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.section-head h2 {
  margin: 0;
  font-size: 20px;
  letter-spacing: 1px;
}

.hint,
.error-hint {
  color: #8aa4c0;
  font-size: 14px;
}

.error-hint {
  color: #ff6b8a;
}

.room-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}

.room-card {
  padding: 18px;
  background: rgba(13, 31, 51, 0.8);
  border: 1px solid rgba(0, 229, 255, 0.2);
  border-radius: 10px;
}

.room-mode {
  display: inline-block;
  font-size: 11px;
  color: #00e5ff;
  border: 1px solid rgba(0, 229, 255, 0.4);
  border-radius: 999px;
  padding: 2px 10px;
}

.room-title {
  margin: 12px 0 6px;
  font-size: 17px;
}

.room-goal {
  margin: 0 0 12px;
  font-size: 13px;
  color: #8aa4c0;
  min-height: 36px;
}

.room-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #8aa4c0;
  margin-bottom: 14px;
}

.enter-btn {
  width: 100%;
}
</style>
