<script setup lang="ts">
/**
 * 学习者沉浸式督学主界面（M4-B）。
 * 页面装配：外层 SceneRenderer 注入场景主题，内部组合 HUD、指挥官对话与惩罚弹窗，
 * 顶部展示本局学习目标与退出/结算操作栏。
 * 摄像头遵循隐私合规：默认关闭、显式知情同意、仅本地内存处理并按 1fps 推帧，绝不落盘。
 * WebSocket 生命周期在此统一创建/连接/销毁；战报弹窗消费 session_finished 完整字段。
 * 所有后端文本一律 {{ }} 插值渲染，严禁 v-html；注释全部中文。
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getToken, roomApi } from '@/api/http'
import { useUserStore } from '@/stores/user'
import { useRoomStore } from '@/stores/room'
import { useSceneStore } from '@/stores/scene'
import SceneRenderer from '@/components/SceneRenderer.vue'
import HudOverlay from '@/components/HudOverlay.vue'
import CommanderChat from '@/components/CommanderChat.vue'
import PunishmentModal from '@/components/PunishmentModal.vue'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()
const roomStore = useRoomStore()
const sceneStore = useSceneStore()

// 从路由参数安全取得房间 ID（可能是字符串或字符串数组）
const currentRoomId = computed(() => {
  const value = route.params.roomId
  return Array.isArray(value) ? value[0] ?? '' : value ?? ''
})

// ===== 一、场景恢复：优先内存态，其次房间详情，最后退回预设首项 =====
const sceneForRender = computed(() => sceneStore.activeScene)
// 目标文本优先取现场恢复值，其次取房间详情，避免刷新/暂离回来后目标丢失
const goalText = computed(() =>
  roomStore.goalText || roomStore.roomDetail?.goal_text || '暂无学习目标（请从督学广场重新进入本局）',
)
const sceneTitle = computed(() => sceneStore.activeScene?.scene_title ?? '督学战场')

/**
 * 挂载时拉取房间现场元信息：恢复目标、剩余机会、锁房、惩罚与场景配置。
 * 房间已被结算/回收（404）时保留空态，交由连接失败提示处理，不阻塞渲染。
 */
async function restoreFromMeta(): Promise<void> {
  try {
    const meta = await roomApi.meta(currentRoomId.value)
    roomStore.applyRoomMeta(meta)
    if (meta.scene_config) sceneStore.setActiveScene(meta.scene_config)
  } catch {
    // 房间不存在或接口异常时忽略，场景回退逻辑由 ensureScene 兜底
  }
}

async function ensureScene(): Promise<void> {
  if (sceneStore.activeScene) return
  const fromDetail = roomStore.roomDetail?.scene_config
  if (fromDetail) {
    sceneStore.setActiveScene(fromDetail)
    return
  }
  try {
    const presets = await sceneStore.loadPresets()
    if (presets.length > 0) sceneStore.setActiveScene(presets[0])
  } catch {
    // 场景加载失败时保留空态，模板显示加载提示，不阻塞房间连接
  }
}

// ===== 二、摄像头隐私合规采集与推流 =====
const consentChecked = ref(false) // 知情同意勾选，未勾选禁用开启按钮
const cameraOn = ref(false)
const cameraError = ref('')
const videoEl = ref<HTMLVideoElement | null>(null)
let cameraStream: MediaStream | null = null
let offscreenCanvas: HTMLCanvasElement | null = null
let frameTimer: number | null = null
const PUSH_WIDTH = 480
const JPEG_QUALITY = 0.7

/** 开启摄像头：显式授权后采集，绑定本地预览并启动 1fps 推流 */
async function enableCamera(): Promise<void> {
  if (!consentChecked.value || cameraOn.value) return
  cameraError.value = ''
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    cameraStream = stream
    cameraOn.value = true
    await nextTick()
    startPush(stream)
  } catch {
    cameraError.value = '摄像头开启失败，请检查浏览器权限或设备占用情况'
    cameraOn.value = false
  }
}

/** 按 480px 等比缩放离屏画布，每秒导出一帧 JPEG Base64 并经 WebSocket 上行 */
function startPush(stream: MediaStream): void {
  const video = videoEl.value
  if (!video) return
  video.srcObject = stream
  const settings = stream.getVideoTracks()[0]?.getSettings()
  const sourceW = settings?.width ?? video.videoWidth ?? 640
  const sourceH = settings?.height ?? video.videoHeight ?? 480
  const scale = PUSH_WIDTH / (sourceW || 640)
  const targetH = Math.round((sourceH || 480) * scale)

  const canvas = document.createElement('canvas')
  canvas.width = PUSH_WIDTH
  canvas.height = targetH
  offscreenCanvas = canvas
  const ctx = canvas.getContext('2d')

  // 压缩帧写入房间帧缓存队列，UI 关闭后即停止推帧
  frameTimer = window.setInterval(() => {
    if (!ctx || video.readyState < 2) return
    ctx.drawImage(video, 0, 0, PUSH_WIDTH, targetH)
    const image = canvas.toDataURL('image/jpeg', JPEG_QUALITY)
    roomStore.send({ type: 'frame', image })
  }, 1000)
}

/** 停止推流并释放摄像头轨道：结束本局、主动退出与组件卸载均调用 */
function stopCamera(): void {
  if (frameTimer !== null) {
    window.clearInterval(frameTimer)
    frameTimer = null
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop())
    cameraStream = null
  }
  if (videoEl.value) videoEl.value.srcObject = null
  offscreenCanvas = null
  cameraOn.value = false
}

// ===== 三、WebSocket 生命周期与战报 =====
interface BattleReport {
  duration_minutes: number
  avg_score: number | null
  total_focus_minutes: number
  rank_title: string
  streak_clean_sessions: number
  shield_count: number
}
const report = ref<BattleReport | null>(null)
const finishError = ref('')
let wsUnsubscribers: Array<() => void> = []

/**
 * 暂时返回广场：只停推、断开连接并清空本地房间态，不结算、不销毁房间。
 * 后端在房主断开时标记暂离并保留现场，稍后可从广场「返回学习室」恢复本局。
 */
function backToPlaza(): void {
  stopCamera()
  roomStore.reset()
  router.push('/plaza')
}

/** 正常结束本局：立即停推（隐私优先），再请求服务端结算，等待 session_finished 弹窗 */
function finishSession(): void {
  stopCamera()
  finishError.value = ''
  const ok = roomStore.send({ type: 'finish_session' })
  if (!ok) finishError.value = '连接未就绪，结算请求未发出，请稍后重试'
}

/** 确认战报：刷新成长档案、清理房间态并返回广场 */
async function confirmReport(): Promise<void> {
  report.value = null
  try {
    await userStore.fetchProfile()
  } catch {
    // 档案刷新失败不阻塞返回，成长数据下次进入个人页会重新拉取
  }
  backToPlaza()
}

onMounted(() => {
  // 先恢复现场（目标/场景/惩罚等），再建立连接；连接建立后 room_state_sync 会二次校准
  void restoreFromMeta().then(() => ensureScene())
  const ws = roomStore.createSocket(() => getToken(), currentRoomId.value)
  // 在连接前订阅：store 内部消费 room_state_sync 恢复进度，此处据其场景配置还原渲染场景
  wsUnsubscribers.push(
    ws.on('session_finished', (msg) => {
      stopCamera()
      report.value = {
        duration_minutes: msg.duration_minutes,
        avg_score: msg.avg_score,
        total_focus_minutes: msg.total_focus_minutes,
        rank_title: msg.rank_title,
        streak_clean_sessions: msg.streak_clean_sessions,
        shield_count: msg.shield_count,
      }
    }),
    ws.on('room_state_sync', (msg) => {
      // 房主重连：以服务端现场为准还原场景（进度与历史对话由 roomStore 恢复）
      if (msg.scene_config) sceneStore.setActiveScene(msg.scene_config)
    }),
  )
  ws.connect()
})

onBeforeUnmount(() => {
  stopCamera()
  wsUnsubscribers.forEach((u) => u())
  wsUnsubscribers = []
  roomStore.reset()
})
</script>

<template>
  <div class="study-page">
    <SceneRenderer v-if="sceneForRender" :scene="sceneForRender">
      <div class="study-layout">
        <!-- 顶部：学习目标与操作栏 -->
        <header class="study-topbar">
          <div class="goal-box">
            <span class="goal-label">本局目标</span>
            <span class="goal-text">{{ goalText }}</span>
          </div>
          <div class="topbar-actions">
            <span class="scene-name">{{ sceneTitle }}</span>
            <button type="button" class="top-btn top-btn--ghost" @click="backToPlaza">暂时返回广场</button>
            <button type="button" class="top-btn top-btn--primary" @click="finishSession">结束本局并结算</button>
          </div>
        </header>

        <!-- 主体：左侧摄像头采集，右侧指挥官对话 -->
        <div class="study-body">
          <section class="camera-panel">
            <div class="panel-title">摄像头 · 隐私合规推流</div>
            <div class="cam-frame">
              <video ref="videoEl" class="cam-preview" autoplay playsinline muted></video>
              <div v-if="!cameraOn" class="cam-placeholder">摄像头未开启</div>
            </div>

            <label class="consent-line">
              <input v-model="consentChecked" type="checkbox" :disabled="cameraOn" />
              我已阅读并同意：画面仅在本地内存中实时处理、按 1fps 压缩推流，不落盘、不留存原始图像
            </label>

            <div class="cam-actions">
              <button
                type="button"
                class="cam-btn"
                :disabled="!consentChecked || cameraOn"
                @click="enableCamera"
              >{{ cameraOn ? '摄像头推流中' : '开启摄像头' }}</button>
              <button type="button" class="cam-btn cam-btn--ghost" :disabled="!cameraOn" @click="stopCamera">
                关闭摄像头
              </button>
            </div>
            <p v-if="cameraError" class="cam-error">{{ cameraError }}</p>
            <p v-if="finishError" class="cam-error">{{ finishError }}</p>
            <p class="cam-status">剩余机会 {{ roomStore.strikesLeft }} · 专注评分 {{ roomStore.focusScore ?? '--' }} · 在线督学 {{ roomStore.watcherCount }}</p>
          </section>

          <section class="chat-wrap">
            <CommanderChat />
          </section>
        </div>

        <!-- HUD 覆盖层与惩罚弹窗 -->
        <HudOverlay />
        <PunishmentModal />
      </div>
    </SceneRenderer>
    <div v-else class="scene-loading">场景加载中，请稍候…</div>

    <!-- 战报摘要弹窗：纯文本插值 -->
    <Teleport to="body">
      <div v-if="report" class="report-mask">
        <div class="report-card">
          <div class="report-badge">本局战报 · 任务结算</div>
          <h2 class="report-title">{{ report.rank_title }}</h2>
          <div class="report-grid">
            <div class="report-item">
              <span class="report-key">本局专注时长</span>
              <span class="report-value">{{ report.duration_minutes }} 分钟</span>
            </div>
            <div class="report-item">
              <span class="report-key">本局平均分</span>
              <span class="report-value">{{ report.avg_score === null ? '暂无' : Math.round(report.avg_score) }}</span>
            </div>
            <div class="report-item">
              <span class="report-key">累计专注时长</span>
              <span class="report-value">{{ report.total_focus_minutes }} 分钟</span>
            </div>
            <div class="report-item">
              <span class="report-key">连续零失误</span>
              <span class="report-value">{{ report.streak_clean_sessions }} 局</span>
            </div>
            <div class="report-item">
              <span class="report-key">自律护盾</span>
              <span class="report-value">× {{ report.shield_count }}</span>
            </div>
          </div>
          <button type="button" class="report-confirm" @click="confirmReport">确认并返回广场</button>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.study-page { height: 100vh; }
.scene-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: #050b14;
  color: #00e5ff;
  font-size: 16px;
}
.study-layout {
  position: relative;
  box-sizing: border-box;
  height: 100%;
  padding: 64px 24px 84px;
  color: var(--scene-primary, #00e5ff);
}
.study-topbar {
  position: relative;
  z-index: 6;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}
.goal-box {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  background: rgba(0, 0, 0, 0.45);
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 6px;
  font-size: 14px;
  max-width: 60%;
}
.goal-label { opacity: 0.7; font-size: 12px; white-space: nowrap; }
.goal-text { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.topbar-actions { display: flex; align-items: center; gap: 10px; }
.scene-name { font-size: 13px; opacity: 0.8; }
.top-btn {
  padding: 7px 14px;
  font-size: 13px;
  border-radius: 5px;
  cursor: pointer;
}
.top-btn--ghost {
  color: var(--scene-primary, #00e5ff);
  background: transparent;
  border: 1px solid var(--scene-primary, #00e5ff);
}
.top-btn--primary {
  color: #04121a;
  background: var(--scene-primary, #00e5ff);
  border: 1px solid var(--scene-primary, #00e5ff);
  font-weight: 700;
}
.study-body {
  position: relative;
  z-index: 4;
  display: grid;
  grid-template-columns: minmax(300px, 380px) minmax(320px, 1fr);
  gap: 20px;
  height: 100%;
  min-height: 0;
  margin-top: 16px;
}
@media (max-width: 860px) {
  .study-body { grid-template-columns: 1fr; grid-template-rows: auto 1fr; overflow-y: auto; }
}
.camera-panel {
  align-self: start;
  padding: 16px;
  background: rgba(0, 0, 0, 0.35);
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 6px;
}
.panel-title { font-size: 14px; font-weight: 700; margin-bottom: 12px; }
.cam-frame {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #05080f;
  border: 1px dashed color-mix(in srgb, var(--scene-primary, #00e5ff) 45%, transparent);
  border-radius: 4px;
  overflow: hidden;
}
.cam-preview { width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); }
.cam-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  opacity: 0.55;
}
.consent-line {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin: 12px 0;
  font-size: 12px;
  line-height: 1.6;
  opacity: 0.9;
}
.consent-line input { margin-top: 2px; }
.cam-actions { display: flex; gap: 10px; }
.cam-btn {
  flex: 1;
  padding: 8px 10px;
  font-size: 13px;
  color: #04121a;
  background: var(--scene-primary, #00e5ff);
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 4px;
  cursor: pointer;
}
.cam-btn--ghost {
  color: var(--scene-primary, #00e5ff);
  background: transparent;
}
.cam-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.cam-error { margin: 10px 0 0; color: var(--scene-alert, #ff2a55); font-size: 12px; }
.cam-status { margin: 12px 0 0; font-size: 12px; opacity: 0.75; }
.chat-wrap { min-height: 0; }

/* 战报弹窗 */
.report-mask {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(3, 8, 16, 0.88);
  backdrop-filter: blur(4px);
}
.report-card {
  width: min(440px, 92vw);
  padding: 28px;
  background: linear-gradient(160deg, #08202c 0%, #0d1f33 100%);
  border: 1px solid #00e5ff;
  border-radius: 12px;
  box-shadow: 0 0 40px rgba(0, 229, 255, 0.25);
  color: #e6f1ff;
}
.report-badge {
  display: inline-block;
  font-size: 12px;
  color: #00e5ff;
  border: 1px solid #00e5ff;
  border-radius: 999px;
  padding: 2px 12px;
}
.report-title { margin: 14px 0 18px; font-size: 24px; color: #7fe9c0; }
.report-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.report-item {
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(0, 229, 255, 0.2);
  border-radius: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.report-key { font-size: 12px; opacity: 0.7; }
.report-value { font-size: 18px; font-weight: 700; }
.report-confirm {
  width: 100%;
  margin-top: 22px;
  padding: 12px;
  font-size: 15px;
  font-weight: 700;
  color: #04121a;
  background: #00e5ff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
</style>
