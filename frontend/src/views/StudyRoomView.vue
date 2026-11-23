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
import type { PublishStreamState } from '@/types'
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

// 房间号紧凑缩略展示，避免长号撑宽顶部信息卡遮挡中央 HUD；完整号通过 title 悬停查看
const shortRoomId = computed(() => {
  const id = currentRoomId.value
  if (!id || id.length <= 13) return id
  return `${id.slice(0, 8)}…${id.slice(-4)}`
})

// ===== 一、场景恢复：优先内存态，其次房间详情，最后退回预设首项 =====
const sceneForRender = computed(() => sceneStore.activeScene)
// 目标文本优先取现场恢复值，其次取房间详情，避免刷新/暂离回来后目标丢失
const goalText = computed(() =>
  roomStore.goalText || roomStore.roomDetail?.goal_text || '暂无学习目标（请从督学广场重新进入本局）',
)
const sceneTitle = computed(() => sceneStore.activeScene?.scene_title ?? '督学战场')

// 实例代次令牌：切房间/卸载后，旧的异步回调不得把旧房数据写进新房 Pinia
let instanceToken = 0

/**
 * 挂载时拉取房间现场元信息：恢复目标、剩余机会、锁房、惩罚与场景配置。
 * 房间已被结算/回收（404）时保留空态，交由连接失败提示处理，不阻塞渲染。
 */
async function restoreFromMeta(): Promise<void> {
  const token = instanceToken
  const roomId = currentRoomId.value
  try {
    const meta = await roomApi.meta(roomId)
    // 仅当本实例仍活跃、房间号未变时才允许写入，避免延迟响应污染其它房间
    if (token !== instanceToken || roomId !== currentRoomId.value) return
    roomStore.applyRoomMeta(meta)
    if (meta.scene_config) sceneStore.setActiveScene(meta.scene_config)
  } catch {
    // 房间不存在或接口异常时忽略，场景回退逻辑由 ensureScene 兜底
  }
}

async function ensureScene(): Promise<void> {
  const token = instanceToken
  if (sceneStore.activeScene) return
  const fromDetail = roomStore.roomDetail?.scene_config
  if (fromDetail) {
    sceneStore.setActiveScene(fromDetail)
    return
  }
  try {
    const presets = await sceneStore.loadPresets()
    // 场景加载完成后再次校验实例活跃，避免切房后覆盖新场景
    if (token !== instanceToken) return
    if (presets.length > 0) sceneStore.setActiveScene(presets[0])
  } catch {
    // 场景加载失败时保留空态，模板显示加载提示，不阻塞房间连接
  }
}

// ===== 二、摄像头隐私合规采集与推流 =====
const consentChecked = ref(false) // 知情同意勾选，未勾选禁用开启按钮
const cameraOn = ref(false)
const cameraError = ref('')
// 发布态状态机：本地开启不等于远端成功，成功态只由后端 frame_ack 驱动
const publishState = ref<PublishStreamState>('CAMERA_CLOSED')
const videoEl = ref<HTMLVideoElement | null>(null)
let cameraStream: MediaStream | null = null
let offscreenCanvas: HTMLCanvasElement | null = null
let frameTimer: number | null = null
let frameSeq = 0                 // 单调递增帧序号，随 frame 上行、由 frame_ack 原样回传
let ackFrameId = -1              // 最近一次被后端确认的帧序号，用于判定是否已真正发布
let consecutiveSendFail = 0      // 连续发送失败计数（WS 未连接或 send 抛错）
const MAX_SEND_FAIL = 3          // 连续失败阈值：超过即进入 STREAM_ERROR
const PUSH_WIDTH = 480
const JPEG_QUALITY = 0.7

/** 是否处于已发布：须同时满足状态为已发布且确有被后端确认过的帧号 */
const isStreamPublished = computed(() => publishState.value === 'STREAM_PUBLISHED' && ackFrameId >= 0)

/** 发布态的中文提示，供 UI 明确区分本地就绪与远端确认 */
const publishStateText = computed(() => {
  switch (publishState.value) {
    case 'CAMERA_CLOSED': return '摄像头未开启'
    case 'CAMERA_OPENING': return '正在开启摄像头…'
    case 'CAMERA_READY': return '本地就绪，等待发布'
    case 'PUBLISHING': return '画面发布中，等待确认…'
    case 'STREAM_PUBLISHED': return '画面已发布，督学官可见'
    case 'STREAM_ERROR': return cameraError.value || '推流失败，正在重试…'
    default: return ''
  }
})

/** 开启摄像头：显式授权后采集，绑定本地预览并启动 1fps 推流 */
async function enableCamera(): Promise<void> {
  if (!consentChecked.value || cameraOn.value) return
  const token = instanceToken
  const roomId = currentRoomId.value
  cameraError.value = ''
  publishState.value = 'CAMERA_OPENING'
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    // 授权回包延迟期间若已切房/卸载，立即释放轨道，杜绝旧房摄像头写入新房
    if (token !== instanceToken || roomId !== currentRoomId.value) {
      stream.getTracks().forEach((track) => track.stop())
      return
    }
    cameraStream = stream
    cameraOn.value = true
    publishState.value = 'CAMERA_READY'
    await nextTick()
    startPush(stream, token, roomId)
  } catch {
    cameraError.value = '摄像头开启失败，请检查浏览器权限或设备占用情况'
    cameraOn.value = false
    publishState.value = 'STREAM_ERROR'
  }
}

/**
 * 按 480px 等比缩放离屏画布，每秒尝试导出一帧 JPEG 并经 WebSocket 上行。
 * 关键纪律：定时器只负责“尝试发送”，绝不把 send 成功当作后端已接收；
 * 成功态仅由 frame_ack(ok=true) 驱动， WS 未连接/发送失败会累积为明确错误态。
 */
function startPush(stream: MediaStream, token: number, roomId: string): void {
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

  frameTimer = window.setInterval(() => {
    // 切房/卸载后旧定时器不得再为新房产帧
    if (token !== instanceToken || roomId !== currentRoomId.value) return
    if (!cameraOn.value) return
    if (!ctx || video.readyState < 2) return
    ctx.drawImage(video, 0, 0, PUSH_WIDTH, targetH)
    const image = canvas.toDataURL('image/jpeg', JPEG_QUALITY)
    const frameId = ++frameSeq
    // 显式检查 WS 是否 OPEN：未连接不伪造成功，累积失败态，待自动重连后继续尝试
    if (!roomStore.isSocketConnected() || !roomStore.send({ type: 'frame', image, frame_id: frameId })) {
      consecutiveSendFail += 1
      if (consecutiveSendFail >= MAX_SEND_FAIL) {
        publishState.value = 'STREAM_ERROR'
        cameraError.value = '连接中断，画面暂停发布，恢复后自动重试'
      } else if (publishState.value !== 'STREAM_PUBLISHED') {
        publishState.value = 'CAMERA_READY'
      }
      return
    }
    consecutiveSendFail = 0
    cameraError.value = ''
    // send 成功仅代表消息进入 WS 缓冲，状态停留 PUBLISHING 等待后端确认
    if (publishState.value !== 'STREAM_PUBLISHED') publishState.value = 'PUBLISHING'
  }, 1000)
}

/** 处理帧回执：仅 ok=true 才推进到 STREAM_PUBLISHED，失败累积为错误态 */
function onFrameAck(msg: { ok: boolean; frame_id?: number; reason?: string }): void {
  if (msg.ok) {
    if (typeof msg.frame_id === 'number') ackFrameId = msg.frame_id
    consecutiveSendFail = 0
    cameraError.value = ''
    publishState.value = 'STREAM_PUBLISHED'
  } else {
    consecutiveSendFail += 1
    if (consecutiveSendFail >= MAX_SEND_FAIL) {
      publishState.value = 'STREAM_ERROR'
      cameraError.value = msg.reason === 'stale_room'
        ? '房间已变更，画面发布已停止'
        : '服务端未接收到有效画面，正在重试'
    }
  }
}

/** 处理连入时下发的发布态：以服务端现场校准本地显示，不凭本地假设成功 */
function onStreamStatus(msg: { published: boolean; ts?: number | null }): void {
  if (!cameraOn.value) return
  if (!msg.published) {
    // 服务端无画面：即便本地已开启也只能停留在等待发布，不能显示已发布
    if (publishState.value === 'STREAM_PUBLISHED') publishState.value = 'PUBLISHING'
  } else if (isStreamPublished.value === false) {
    // 服务端已有画面（如本房先前推过）：确认发布成功
    publishState.value = 'STREAM_PUBLISHED'
  }
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
  publishState.value = 'CAMERA_CLOSED'
  consecutiveSendFail = 0
  ackFrameId = -1
}

// ===== 二·五、房间号与督学链接复制 =====
const copyTip = ref('')

/** 通用剪贴板写入：优先 Clipboard API，失败时降级到临时 textarea + execCommand */
async function writeClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Clipboard API 不可用时走降级路径
  }
  try {
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(textarea)
    return ok
  } catch {
    return false
  }
}

/** 短暂展示复制成功提示，2 秒后自动淡出 */
function showCopyTip(text: string): void {
  copyTip.value = text
  window.setTimeout(() => {
    if (copyTip.value === text) copyTip.value = ''
  }, 2000)
}

/** 复制纯房间号 */
async function copyRoomId(): Promise<void> {
  const ok = await writeClipboard(currentRoomId.value)
  showCopyTip(ok ? '房间号已复制' : '复制失败，请手动复制房间号')
}

/** 复制完整督学链接（/watch/:roomId） */
async function copyWatchLink(): Promise<void> {
  const link = `${window.location.origin}/watch/${currentRoomId.value}`
  const ok = await writeClipboard(link)
  showCopyTip(ok ? '督学链接已复制' : '复制失败，请手动复制链接')
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
const finishing = ref(false)            // 结算进行中：点击即置位，防止短时间重复点击
let finishRequestId = ''                // 本次结算请求标识，用于关联 ACK、忽略旧回执
let finishTimer: number | null = null   // 结算超时兜底，避免"发送成功即无限等待"
let reportShownForRequest = ''          // 已弹窗的请求标识，避免 ACK 与 session_finished 重复弹窗
let wsUnsubscribers: Array<() => void> = []

/** 清除结算超时计时器 */
function clearFinishTimer(): void {
  if (finishTimer !== null) {
    window.clearTimeout(finishTimer)
    finishTimer = null
  }
}

/** 统一弹战报（幂等）：同一请求只弹一次，兼容 ACK 与 session_finished 两条通路 */
function showReport(result: BattleReport, rid: string): void {
  if (reportShownForRequest === rid) return
  reportShownForRequest = rid
  finishing.value = false
  clearFinishTimer()
  stopCamera()
  report.value = result
}

/** 结算失败：明确原因、恢复可点击状态，不吞异常假装成功 */
function failFinish(message: string): void {
  finishing.value = false
  clearFinishTimer()
  finishError.value = message || '结算失败，请重试'
}

/**
 * 正常结束本局：立即停推（隐私优先），再请求服务端幂等结算。
 * 点击即 loading + 防重复；成功/失败均有明确反馈；超时兜底不无限等待。
 * 网络中断/重复点击/旧回执均由服务端按 room_id 幂等兜底，可安全重试。
 */
function finishSession(): void {
  if (finishing.value) return          // 防短时间重复点击
  stopCamera()
  finishError.value = ''
  finishing.value = true
  finishRequestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`
  clearFinishTimer()
  const ok = roomStore.send({ type: 'finish_session', request_id: finishRequestId })
  if (!ok) {
    failFinish('连接未就绪，结算请求未发出，请稍后重试')
    return
  }
  // 超时兜底：不把"网络发送成功"当业务成功，也不无限等待
  finishTimer = window.setTimeout(() => {
    if (finishing.value) {
      failFinish('结算响应超时，可能网络中断；请稍后重试（服务端保证只结算一次）')
    }
  }, 8000)
}

/**
 * 暂时返回广场：只停推、断开连接并清空本地房间态，不结算、不销毁房间。
 * 后端在房主断开时标记暂离并保留现场，稍后可从广场「返回学习室」恢复本局。
 */
function backToPlaza(): void {
  stopCamera()
  roomStore.reset()
  router.push('/plaza')
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
  // 每次挂载递增令牌：旧实例的异步回调（restoreFromMeta/ensureScene）持有旧 token 将被丢弃
  instanceToken += 1
  const token = instanceToken
  const mountedRoomId = currentRoomId.value
  // sceneStore 为全局单例：先清空上间房的场景，避免旧房场景残留到新房渲染
  sceneStore.resetScene()
  // 先恢复现场（目标/场景/惩罚等），再建立连接；连接建立后 room_state_sync 会二次校准
  void restoreFromMeta().then(() => {
    if (token === instanceToken && mountedRoomId === currentRoomId.value) void ensureScene()
  })
  const ws = roomStore.createSocket(() => getToken(), mountedRoomId)
  // 在连接前订阅：store 内部消费 room_state_sync 恢复进度，此处据其场景配置还原渲染场景
  wsUnsubscribers.push(
    // 帧回执：唯一能把本地摄像头推进到“已发布”的依据；旧客户端不认识该 type 即忽略
    ws.on('frame_ack', (msg) => {
      if (token !== instanceToken || mountedRoomId !== currentRoomId.value) return
      onFrameAck(msg)
    }),
    // 连入发布态：以服务端现场校准本地，刷新/重连后不误判“已发布”
    ws.on('stream_status', (msg) => {
      if (token !== instanceToken || mountedRoomId !== currentRoomId.value) return
      onStreamStatus(msg)
    }),
    // ACK：新后端回执，承载终态（成功/重复/失败）；旧客户端不认识该 type，无此订阅即忽略
    ws.on('session_finish_ack', (msg) => {
      if (token !== instanceToken || mountedRoomId !== currentRoomId.value) return
      // 忽略旧请求的迟到回执，避免污染当前会话
      if (msg.request_id && msg.request_id !== finishRequestId) return
      const st = msg.status
      if ((st === 'success' || st === 'duplicate') && msg.result) {
        showReport({
          duration_minutes: msg.result.duration_minutes,
          avg_score: msg.result.avg_score,
          total_focus_minutes: msg.result.total_focus_minutes,
          rank_title: msg.result.rank_title,
          streak_clean_sessions: msg.result.streak_clean_sessions,
          shield_count: msg.result.shield_count,
        }, finishRequestId)
      } else if (st === 'in_progress') {
        // 并发中：本房另有请求正在结算，继续等待终态 ACK/广播，不误判失败
        finishError.value = '正在结算中，请稍候…'
      } else {
        // db_failed / forbidden / not_found / internal_error：明确原因，停止等待
        failFinish(msg.message || '结算失败，请重试')
      }
    }),
    // session_finished：兼容旧后端广播，与新 ACK 共用去重键，避免重复弹窗
    ws.on('session_finished', (msg) => {
      if (token !== instanceToken || mountedRoomId !== currentRoomId.value) return
      showReport({
        duration_minutes: msg.duration_minutes,
        avg_score: msg.avg_score,
        total_focus_minutes: msg.total_focus_minutes,
        rank_title: msg.rank_title,
        streak_clean_sessions: msg.streak_clean_sessions,
        shield_count: msg.shield_count,
      }, finishRequestId)
    }),
    ws.on('room_state_sync', (msg) => {
      if (token !== instanceToken || mountedRoomId !== currentRoomId.value) return
      // 房主重连：以服务端现场为准还原场景（进度与历史对话由 roomStore 恢复）
      if (msg.scene_config) sceneStore.setActiveScene(msg.scene_config)
    }),
  )
  ws.connect()
})

onBeforeUnmount(() => {
  // 先失效本实例所有异步回调，再停推/退订/清状态，杜绝旧房事件回写新房
  instanceToken += 1
  clearFinishTimer()
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
        <!-- HUD 状态条：正常流首行，按内容动态换行，不再覆盖房间信息框 -->
        <HudOverlay />
        <!-- 顶部：房间号、学习目标与操作栏 -->
        <header class="study-topbar">
          <div class="goal-box">
            <div class="room-id-line">
              <span class="room-id-label">房间号</span>
              <span class="room-id-value" :title="currentRoomId">{{ shortRoomId }}</span>
              <button type="button" class="room-copy-btn" @click="copyRoomId">复制房间号</button>
              <button type="button" class="room-copy-btn" @click="copyWatchLink">复制督学链接</button>
              <span v-if="copyTip" class="copy-tip">{{ copyTip }}</span>
            </div>
            <span class="goal-label">本局目标</span>
            <span class="goal-text">{{ goalText }}</span>
          </div>
          <div class="topbar-actions">
            <span class="scene-name">{{ sceneTitle }}</span>
            <button type="button" class="top-btn top-btn--ghost" @click="backToPlaza">暂时返回广场</button>
            <button type="button" class="top-btn top-btn--primary" :disabled="finishing" @click="finishSession">
              {{ finishing ? '结算中…' : '结束本局并结算' }}
            </button>
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
              >{{ cameraOn ? '摄像头已开启' : '开启摄像头' }}</button>
              <button type="button" class="cam-btn cam-btn--ghost" :disabled="!cameraOn" @click="stopCamera">
                关闭摄像头
              </button>
            </div>
            <!-- 发布态明确区分“本地就绪”与“后端已确认”：不用定时器伪造成功 -->
            <p class="cam-publish-state" :class="{ 'cam-publish-state--ok': isStreamPublished }">
              推流状态：{{ publishStateText }}
            </p>
            <p v-if="cameraError" class="cam-error">{{ cameraError }}</p>
            <p v-if="finishError" class="cam-error">{{ finishError }}</p>
            <p class="cam-status">剩余机会 {{ roomStore.strikesLeft }} · 专注评分 {{ roomStore.focusScore ?? '--' }} · 在线督学 {{ roomStore.watcherCount }}</p>
          </section>

          <section class="chat-wrap">
            <CommanderChat />
          </section>
        </div>

        <!-- 惩罚弹窗 -->
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
  display: flex;
  flex-direction: column;
  gap: 12px;
  /* 顶部不再为绝对定位 HUD 预留固定 64px：状态条按内容自然占位 */
  padding: 8px 24px 84px;
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
  flex-wrap: wrap;
  gap: 10px;
  padding: 8px 14px;
  background: rgba(0, 0, 0, 0.45);
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 6px;
  font-size: 14px;
  max-width: 60%;
}
.room-id-line {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-basis: 100%;
  padding-bottom: 6px;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.15);
}
.room-id-label { opacity: 0.7; font-size: 12px; white-space: nowrap; }
.room-id-value {
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--scene-primary, #00e5ff);
  user-select: all;
}
.room-copy-btn {
  padding: 2px 8px;
  font-size: 11px;
  color: var(--scene-primary, #00e5ff);
  background: transparent;
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 4px;
  cursor: pointer;
}
.copy-tip { font-size: 11px; color: #6ee7a8; }
.goal-label { opacity: 0.7; font-size: 12px; white-space: nowrap; }
.goal-text { font-weight: 600; min-width: 0; overflow-wrap: anywhere; white-space: normal; }
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
  flex: 1 1 auto;
  min-height: 0;
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
.cam-publish-state { margin: 10px 0 0; font-size: 12px; color: var(--scene-text-dim, #9aa4b2); }
.cam-publish-state--ok { color: #4ade80; }
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
