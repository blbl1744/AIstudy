<script setup lang="ts">
/**
 * 督学官监督界面（M4-B）。
 * 独立持有房间 WebSocket，监听哨兵预警 / 房间互动 / 分析更新 / 结算与断连信号；
 * 受鉴权快照每 1.5 秒以 blob 轮询并即时回收 ObjectURL，避免内存泄漏；
 * 工具箱提供快捷弹幕、自定义弹幕（1～60 字）与 4 种互动音效，侧栏实时滚动互动记录。
 * 所有后端文本一律 {{ }} 插值渲染，严禁 v-html；注释全部中文。
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import http, { getToken } from '@/api/http'
import { RoomSocket } from '@/api/roomSocket'
import { useUserStore } from '@/stores/user'
import type { SoundEffect } from '@/types'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()
const roomId = computed(() => {
  const value = route.params.roomId
  return Array.isArray(value) ? value[0] ?? '' : value ?? ''
})
const myName = computed(() => userStore.nickname || '我')

// ===== 一、受鉴权画面轮询（blob + ObjectURL，轮询前后即时回收） =====
const snapshotUrl = ref('')
const hasFrame = ref(false)
const waitingHost = ref(true)
let pollTimer: number | null = null

/** 拉取一帧受鉴权快照；请求经 http 客户端自动携带 Bearer Token */
async function pollSnapshot(): Promise<void> {
  try {
    const res = await http.get<Blob>(`/rooms/${encodeURIComponent(roomId.value)}/snapshot.jpg`, {
      responseType: 'blob',
    })
    const blob = res.data
    if (!blob || blob.size === 0 || !blob.type.startsWith('image/')) {
      throw new Error('empty snapshot')
    }
    // 刷新前释放上一帧 URL，防止反复轮询造成内存泄漏
    if (snapshotUrl.value) URL.revokeObjectURL(snapshotUrl.value)
    snapshotUrl.value = URL.createObjectURL(blob)
    hasFrame.value = true
    waitingHost.value = false
  } catch {
    // 404 / 暂无画面：回收旧帧并回到"等待推流"占位
    if (snapshotUrl.value) {
      URL.revokeObjectURL(snapshotUrl.value)
      snapshotUrl.value = ''
    }
    hasFrame.value = false
    waitingHost.value = true
  }
}

// ===== 二、哨兵预警横幅（纯文本插值 + CSS 闪烁） =====
const sentinelMsg = ref('')
const sentinelVisible = ref(false)
let sentinelTimer: number | null = null

// ===== 三、房间结束与断连提示 =====
const sessionEnded = ref(false)
const connectionLost = ref(false)

// ===== 四、互动记录（本地已发送 + 服务端广播，含轻量去重） =====
interface InteractionItem {
  id: number
  kind: 'barrage' | 'sound_effect'
  from: string
  text?: string
  sound?: string
  time: string
}
const interactions = ref<InteractionItem[]>([])
const listEl = ref<HTMLElement | null>(null)
let interactionSeq = 0
let lastSentSig = ''
let lastSentAt = 0

function signature(kind: string, text: string | undefined, sound: string | undefined): string {
  return `${kind}|${text ?? ''}|${sound ?? ''}`
}

async function pushInteraction(item: Omit<InteractionItem, 'id' | 'time'>): Promise<void> {
  interactions.value.push({
    ...item,
    id: ++interactionSeq,
    time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
  })
  await nextTick()
  if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight
}

/** 收到服务端广播；过滤自己刚发出又被回传的同条消息，避免重复展示 */
function onRemoteInteraction(msg: { kind: 'barrage' | 'sound_effect'; from: string; text?: string; sound?: string }): void {
  const sig = signature(msg.kind, msg.text, msg.sound)
  const now = Date.now()
  if (msg.from === myName.value && sig === lastSentSig && now - lastSentAt < 1500) return
  void pushInteraction({ kind: msg.kind, from: msg.from, text: msg.text, sound: msg.sound })
}

// ===== 五、互动工具箱 =====
const QUICK_BARRAGES = ['抬头！别低头', '专心做题，加油', '坐直了！', '坚持住，快到终点']
const SOUND_OPTIONS: { value: SoundEffect; label: string }[] = [
  { value: 'knock_board', label: '敲黑板' },
  { value: 'clear_throat', label: '清嗓子' },
  { value: 'alarm', label: '防空警报' },
  { value: 'cheer', label: '欢呼鼓励' },
]
const SOUND_LABEL: Record<SoundEffect, string> = {
  knock_board: '敲黑板',
  clear_throat: '清嗓子',
  alarm: '防空警报',
  cheer: '欢呼鼓励',
}
const barrageText = ref('')
const toolError = ref('')
const BARRAGE_MAX = 60
let socket: RoomSocket | null = null
let unsubscribers: Array<() => void> = []

function sendBarrage(text: string): void {
  const content = text.trim()
  if (content.length < 1 || content.length > BARRAGE_MAX) {
    toolError.value = `弹幕长度需为 1～${BARRAGE_MAX} 字`
    return
  }
  if (!socket?.send({ type: 'barrage', text: content })) {
    toolError.value = '连接未就绪，发送失败'
    return
  }
  toolError.value = ''
  lastSentSig = signature('barrage', content, undefined)
  lastSentAt = Date.now()
  void pushInteraction({ kind: 'barrage', from: myName.value, text: content })
  barrageText.value = ''
}

function sendSound(sound: SoundEffect): void {
  if (!socket?.send({ type: 'sound_effect', sound })) {
    toolError.value = '连接未就绪，发送失败'
    return
  }
  toolError.value = ''
  lastSentSig = signature('sound_effect', undefined, sound)
  lastSentAt = Date.now()
  void pushInteraction({ kind: 'sound_effect', from: myName.value, sound: SOUND_LABEL[sound] })
}

function backToPlaza(): void {
  router.push('/plaza')
}

onMounted(() => {
  socket = new RoomSocket({
    roomId: roomId.value,
    getToken,
    onOpen: () => { if (!sessionEnded.value) connectionLost.value = false },
    onClose: () => { if (!sessionEnded.value) connectionLost.value = true },
  })
  unsubscribers.push(
    socket.on('sentinel_alert', (msg) => {
      sentinelMsg.value = msg.message
      sentinelVisible.value = true
      if (sentinelTimer !== null) window.clearTimeout(sentinelTimer)
      sentinelTimer = window.setTimeout(() => { sentinelVisible.value = false }, 6000)
    }),
    socket.on('room_interaction', (msg) => onRemoteInteraction(msg)),
    socket.on('analysis_update', () => {
      // 分析更新可用于后续扩展画面标注；当前监督页仅依赖快照轮询
    }),
    socket.on('session_finished', () => {
      sessionEnded.value = true
      connectionLost.value = false
    }),
    socket.on('room_locked', (msg) => {
      sessionEnded.value = true
      connectionLost.value = false
      sentinelMsg.value = msg.msg
      sentinelVisible.value = true
    }),
  )
  socket.connect()
  void pollSnapshot()
  pollTimer = window.setInterval(() => { void pollSnapshot() }, 1500)
})

onBeforeUnmount(() => {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
  if (sentinelTimer !== null) {
    window.clearTimeout(sentinelTimer)
    sentinelTimer = null
  }
  unsubscribers.forEach((u) => u())
  unsubscribers = []
  socket?.close()
  socket = null
  if (snapshotUrl.value) {
    URL.revokeObjectURL(snapshotUrl.value)
    snapshotUrl.value = ''
  }
})
</script>

<template>
  <div class="watch-page">
    <!-- 顶部操作栏 -->
    <header class="watch-topbar">
      <div class="watch-brand">督学官监督台 <span class="room-tag">房间 {{ roomId }}</span></div>
      <button type="button" class="back-btn" @click="backToPlaza">返回督学广场</button>
    </header>

    <!-- 断连提示（信息性，不强制跳转） -->
    <div v-if="connectionLost" class="conn-banner">与房间的连接已断开，正在自动尝试重连…</div>

    <!-- 哨兵预警横幅：高亮闪烁，纯文本插值 -->
    <div v-if="sentinelVisible" class="sentinel-banner">{{ sentinelMsg }}</div>

    <main class="watch-main">
      <!-- 左侧：实时画面 -->
      <section class="screen-panel">
        <div class="panel-head">实时学习画面</div>
        <div class="screen-frame">
          <img v-if="hasFrame" :src="snapshotUrl" alt="学习者实时画面快照" class="screen-img" />
          <div v-if="waitingHost" class="screen-placeholder">等待房主开启摄像头推流…</div>
        </div>
        <p class="screen-hint">画面为受鉴权快照，每 1.5 秒刷新，不在本地留存</p>
      </section>

      <!-- 右侧：互动工具箱 + 互动记录 -->
      <section class="tool-panel">
        <div class="panel-head">互动工具箱</div>

        <div class="tool-block">
          <div class="tool-label">快捷督促弹幕</div>
          <div class="chip-row">
            <button
              v-for="text in QUICK_BARRAGES"
              :key="text"
              type="button"
              class="chip"
              @click="sendBarrage(text)"
            >{{ text }}</button>
          </div>
        </div>

        <div class="tool-block">
          <div class="tool-label">自定义弹幕（1～60 字）</div>
          <div class="custom-row">
            <input
              v-model="barrageText"
              type="text"
              class="barrage-input"
              :maxlength="BARRAGE_MAX"
              placeholder="输入督促内容..."
              @keyup.enter="sendBarrage(barrageText)"
            />
            <button type="button" class="send-btn" @click="sendBarrage(barrageText)">发送</button>
          </div>
        </div>

        <div class="tool-block">
          <div class="tool-label">互动音效</div>
          <div class="chip-row">
            <button
              v-for="opt in SOUND_OPTIONS"
              :key="opt.value"
              type="button"
              class="chip chip--sound"
              @click="sendSound(opt.value)"
            >{{ opt.label }}</button>
          </div>
        </div>

        <p v-if="toolError" class="tool-error">{{ toolError }}</p>

        <div class="tool-block log-block">
          <div class="tool-label">互动记录</div>
          <div ref="listEl" class="log-list">
            <p v-if="interactions.length === 0" class="log-empty">暂无互动，点击下方按钮发送督促</p>
            <div v-for="item in interactions" :key="item.id" class="log-item">
              <span class="log-time">{{ item.time }}</span>
              <span class="log-from">{{ item.from }}</span>
              <span class="log-content">{{ item.kind === 'barrage' ? item.text : ('音效：' + item.sound) }}</span>
            </div>
          </div>
        </div>
      </section>
    </main>

    <!-- 房主结束本局：明确提示并引导返回广场 -->
    <div v-if="sessionEnded" class="ended-mask">
      <div class="ended-card">
        <div class="ended-badge">监督结束</div>
        <h2>房主已结束本局并完成结算</h2>
        <p>感谢你的督学支持，可返回广场继续监督其他学习者。</p>
        <button type="button" class="ended-btn" @click="backToPlaza">返回督学广场</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.watch-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #050b14 0%, #0d1f33 100%);
  color: #e6f1ff;
}
.watch-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 28px;
  border-bottom: 1px solid rgba(0, 229, 255, 0.2);
  background: rgba(5, 11, 20, 0.7);
}
.watch-brand { font-size: 18px; font-weight: 700; letter-spacing: 1px; color: #00e5ff; }
.room-tag { font-size: 12px; font-weight: 400; color: #8aa4c0; margin-left: 10px; }
.back-btn {
  padding: 7px 14px;
  font-size: 13px;
  color: #00e5ff;
  background: transparent;
  border: 1px solid #00e5ff;
  border-radius: 5px;
  cursor: pointer;
}
.conn-banner {
  padding: 8px 20px;
  text-align: center;
  font-size: 13px;
  color: #ffd666;
  background: rgba(255, 214, 102, 0.12);
  border-bottom: 1px solid rgba(255, 214, 102, 0.3);
}
.sentinel-banner {
  padding: 12px 20px;
  text-align: center;
  font-size: 16px;
  font-weight: 700;
  color: #fff;
  background: rgba(255, 42, 85, 0.85);
  animation: sentinel-flash 0.9s ease-in-out infinite;
}
@keyframes sentinel-flash {
  0%, 100% { background: rgba(255, 42, 85, 0.55); }
  50% { background: rgba(255, 42, 85, 1); }
}
.watch-main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
  display: grid;
  grid-template-columns: minmax(360px, 1.3fr) minmax(320px, 1fr);
  gap: 24px;
  align-items: start;
}
@media (max-width: 900px) {
  .watch-main { grid-template-columns: 1fr; }
}
.screen-panel,
.tool-panel {
  padding: 18px;
  background: rgba(13, 31, 51, 0.6);
  border: 1px solid rgba(0, 229, 255, 0.18);
  border-radius: 12px;
}
.panel-head { font-size: 16px; font-weight: 700; color: #00e5ff; margin-bottom: 14px; }
.screen-frame {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #05080f;
  border: 1px solid rgba(0, 229, 255, 0.3);
  border-radius: 6px;
  overflow: hidden;
}
.screen-img { width: 100%; height: 100%; object-fit: cover; }
.screen-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  color: #8aa4c0;
}
.screen-hint { margin: 12px 0 0; font-size: 12px; color: #8aa4c0; }

.tool-block { margin-bottom: 18px; }
.tool-label { font-size: 13px; color: #b9cde2; margin-bottom: 8px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  padding: 6px 10px;
  font-size: 12px;
  color: #00e5ff;
  background: rgba(0, 229, 255, 0.08);
  border: 1px solid rgba(0, 229, 255, 0.35);
  border-radius: 999px;
  cursor: pointer;
}
.chip:hover { background: rgba(0, 229, 255, 0.2); }
.chip--sound { color: #ffd666; border-color: rgba(255, 214, 102, 0.4); background: rgba(255, 214, 102, 0.08); }
.custom-row { display: flex; gap: 8px; }
.barrage-input {
  flex: 1;
  min-width: 0;
  padding: 7px 10px;
  font-size: 13px;
  color: #fff;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(0, 229, 255, 0.3);
  border-radius: 5px;
  outline: none;
}
.send-btn {
  padding: 7px 14px;
  font-size: 13px;
  color: #04121a;
  background: #00e5ff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}
.tool-error { margin: 0 0 12px; font-size: 12px; color: #ff6b8a; }
.log-block { margin-bottom: 0; }
.log-list {
  height: 220px;
  overflow-y: auto;
  padding: 10px;
  background: rgba(5, 20, 34, 0.8);
  border: 1px solid rgba(0, 229, 255, 0.15);
  border-radius: 6px;
}
.log-empty { font-size: 12px; color: #8aa4c0; margin: 0; }
.log-item {
  display: flex;
  gap: 8px;
  font-size: 12px;
  padding: 5px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  line-height: 1.5;
}
.log-time { color: #6f8aa5; white-space: nowrap; }
.log-from { color: #00e5ff; white-space: nowrap; }
.log-content { color: #e6f1ff; word-break: break-all; }

.ended-mask {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(3, 8, 16, 0.88);
  backdrop-filter: blur(4px);
}
.ended-card {
  width: min(420px, 92vw);
  padding: 28px;
  text-align: center;
  background: linear-gradient(160deg, #08202c 0%, #0d1f33 100%);
  border: 1px solid #00e5ff;
  border-radius: 12px;
}
.ended-badge {
  display: inline-block;
  font-size: 12px;
  color: #00e5ff;
  border: 1px solid #00e5ff;
  border-radius: 999px;
  padding: 2px 12px;
}
.ended-card h2 { margin: 16px 0 8px; font-size: 20px; }
.ended-card p { margin: 0 0 20px; font-size: 13px; color: #8aa4c0; line-height: 1.6; }
.ended-btn {
  padding: 10px 22px;
  font-size: 14px;
  font-weight: 700;
  color: #04121a;
  background: #00e5ff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
</style>
