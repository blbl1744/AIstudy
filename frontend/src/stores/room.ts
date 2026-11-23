/**
 * 房间状态仓库：当前房间、剩余机会、专注评分、锁房状态与指挥官字幕流。
 * 通过 bindSocket 订阅 RoomSocket 的强类型下行消息，集中维护所有房间态。
 */
import { computed, ref, shallowRef } from 'vue'
import { defineStore } from 'pinia'
import { roomApi } from '@/api/http'
import { RoomSocket } from '@/api/roomSocket'
import type {
  CreateRoomRequest,
  DialogueHistoryItem,
  DialogueTrigger,
  RoomDetail,
  RoomMeta,
  WsIncomingMsg,
  WsOutgoingMsg,
  WsRoomStateSyncMsg,
} from '@/types'

/** 一条指挥官字幕 */
export interface DialogueLine {
  id: number
  text: string
  trigger: DialogueTrigger
}

/** 最近一条房间互动（弹幕 / 音效） */
export interface RoomInteraction {
  kind: 'barrage' | 'sound_effect'
  from: string
  text?: string
  sound?: string
  /** 服务端/客户端生成的唯一标识，供幂等去重；旧消息可能缺省 */
  message_id?: string
  /** 服务端认证身份，不依赖可重名的昵称 */
  user_id?: number
  /** 单调递增序号，供晚挂载组件按序补放 */
  seq: number
}

export const useRoomStore = defineStore('room', () => {
  // ===== 状态 =====
  const roomId = ref('')
  const roomDetail = ref<RoomDetail | null>(null)
  // 学习目标：暂离重返/刷新后由 meta 或 room_state_sync 恢复，独立于 roomDetail
  const goalText = ref('')
  const strikesLeft = ref(3)
  const focusScore = ref<number | null>(null)
  const distractedStreakSec = ref(0)
  // 连续走神阈值（秒）由后端惩罚引擎权威下发；默认值仅用于首帧到达前占位，
  // 不参与任何业务裁决（是否走神/是否扣血完全由后端决定）。
  const strikeSeconds = ref(10)
  const isLocked = ref(false)
  const activePunishment = ref('')
  const watcherCount = ref(0)
  const watchers = ref<{ user_id: number; nickname: string }[]>([])
  const dialogueLines = ref<DialogueLine[]>([])
  const dialogueActive = ref(false)
  const lastInteraction = ref<RoomInteraction | null>(null)
  const sentinelMessage = ref('')
  // 互动音效开关（督学官弹幕音效），由左下控制条与 HUD 共享，避免悬浮按钮遮挡聊天栏
  const interactionSoundOn = ref(true)
  // WebSocket 实例非响应式对象，用 shallowRef 避免深层代理
  const socket = shallowRef<RoomSocket | null>(null)

  let dialogueSeq = 0

  // ===== 弹幕可靠去重（第五阶段⑦）=====
  // 单一权威数据流：所有 room_interaction 先在此按 message_id 幂等去重并写入有界缓存，
  // HudOverlay 只从缓存按游标消费，从而避免“直接订阅 socket 受挂载/重连时序影响丢消息”。
  const INTERACTION_CACHE_MAX = 128
  const interactionCache = ref<RoomInteraction[]>([])
  let interactionSeq = 0
  let hudCursorSeq = 0
  /** HUD 监听用的单调水位：每次新增互动自增，避免缓存满后 length 不变漏触发 */
  const interactionHeadSeq = ref(0)
  const seenInteractionIds = new Set<string>()
  const seenIdOrder: string[] = []

  /** 记忆 message_id 并判断是否首次见到；无 id 的旧消息用自动 id 兜底，缓存有界 FIFO 淘汰。 */
  function rememberInteraction(id: string): boolean {
    if (seenInteractionIds.has(id)) return false
    seenInteractionIds.add(id)
    seenIdOrder.push(id)
    if (seenIdOrder.length > INTERACTION_CACHE_MAX) {
      const evicted = seenIdOrder.shift()
      if (evicted !== undefined) seenInteractionIds.delete(evicted)
    }
    return true
  }

  /** 处理一条服务端互动广播：去重后写入缓存；返回是否新增。 */
  function noteInteraction(msg: {
    kind: 'barrage' | 'sound_effect'
    from: string
    text?: string
    sound?: string
    message_id?: string
    user_id?: number
  }): boolean {
    const id = (msg.message_id && msg.message_id.trim())
      ? msg.message_id.trim()
      : `auto:${++interactionSeq}`
    const item: RoomInteraction = {
      kind: msg.kind,
      from: msg.from,
      text: msg.text,
      sound: msg.sound,
      message_id: id,
      user_id: msg.user_id,
      seq: ++interactionSeq,
    }
    lastInteraction.value = item
    if (!rememberInteraction(id)) return false
    interactionCache.value.push(item)
    if (interactionCache.value.length > INTERACTION_CACHE_MAX) {
      interactionCache.value.shift()
    }
    interactionHeadSeq.value = item.seq
    return true
  }

  /** HUD 按游标取走尚未展示的互动，推进游标；晚挂载与重连均不重复、不丢。 */
  function drainHudInteractions(): RoomInteraction[] {
    const pending = interactionCache.value.filter((item) => item.seq > hudCursorSeq)
    if (pending.length > 0) {
      hudCursorSeq = pending[pending.length - 1].seq
    }
    return pending
  }

  // ===== 派生 =====
  const strikesUsed = computed(() => Math.max(0, 3 - strikesLeft.value))
  const subtitleText = computed(() => dialogueLines.value.map((line) => line.text).join(''))

  /** 创建房间并写入房间态 */
  async function createRoom(payload: CreateRoomRequest): Promise<RoomDetail> {
    const room = await roomApi.create(payload)
    applyRoomDetail(room)
    return room
  }

  /** 用房间详情初始化状态 */
  function applyRoomDetail(room: RoomDetail): void {
    roomId.value = room.room_id
    roomDetail.value = room
    goalText.value = room.goal_text ?? ''
    strikesLeft.value = room.strikes_left ?? 3
    isLocked.value = Boolean(room.is_locked)
    activePunishment.value = room.active_punishment ?? ''
  }

  /**
   * 把后端历史对话条目映射为字幕行。
   * trigger 收紧到合法联合成员，非法值回退 chat；id 用负数避免与实时 ++dialogueSeq 冲突。
   */
  function applyDialogueHistory(history: DialogueHistoryItem[]): void {
    if (!Array.isArray(history) || history.length === 0) {
      return
    }
    const validTriggers: DialogueTrigger[] = ['proactive', 'distraction', 'chat']
    dialogueLines.value = history.map((item, index) => ({
      id: -(index + 1),
      text: String(item?.text ?? ''),
      trigger: validTriggers.includes(item?.trigger as DialogueTrigger)
        ? (item.trigger as DialogueTrigger)
        : 'chat',
    }))
  }

  /** 用房间元信息（GET /api/rooms/:id）恢复现场，供暂离重返/刷新后校验与还原 */
  function applyRoomMeta(meta: RoomMeta): void {
    roomId.value = meta.room_id
    goalText.value = meta.goal_text ?? ''
    strikesLeft.value = meta.strikes_left ?? 3
    isLocked.value = Boolean(meta.is_locked)
    activePunishment.value = meta.active_punishment ?? ''
  }

  /** 房间级现场字段：meta 与 room_state_sync 共有的进度与惩罚状态 */
  type SceneSnapshot = Pick<WsRoomStateSyncMsg,
    'goal_text' | 'strikes_left' | 'is_locked' | 'active_punishment' | 'scene_config'>

  /** 应用现场快照（仅状态字段，场景配置由视图层交给 sceneStore），供重连恢复进度 */
  function applySceneSnapshot(snapshot: SceneSnapshot): void {
    goalText.value = snapshot.goal_text ?? ''
    strikesLeft.value = snapshot.strikes_left ?? 3
    isLocked.value = Boolean(snapshot.is_locked)
    activePunishment.value = snapshot.active_punishment ?? ''
  }

  /** 构造 WebSocket 管理器实例（不自动连接，由调用方决定时机） */
  function createSocket(getToken: () => string | null, roomIdValue = roomId.value): RoomSocket {
    const ws = new RoomSocket({ roomId: roomIdValue, getToken })
    bindSocket(ws)
    socket.value = ws
    return ws
  }

  /** 绑定所有下行消息到状态更新逻辑 */
  function bindSocket(ws: RoomSocket): void {
    ws.on('analysis_update', (msg) => {
      if (msg.skipped) {
        return
      }
      if (typeof msg.focus_score === 'number' || msg.focus_score === null) {
        focusScore.value = msg.focus_score
      }
      if (typeof msg.strikes_left === 'number') {
        strikesLeft.value = msg.strikes_left
      }
      if (typeof msg.distracted_streak_sec === 'number') {
        distractedStreakSec.value = msg.distracted_streak_sec
      }
      // 阈值以服务端实际惩罚阈值为准，前端不硬编码、不自行推算业务状态
      if (typeof msg.strike_seconds === 'number' && msg.strike_seconds > 0) {
        strikeSeconds.value = msg.strike_seconds
      }
    })

    ws.on('ai_dialogue_start', (msg) => {
      dialogueActive.value = true
      dialogueLines.value.push({ id: ++dialogueSeq, text: '', trigger: msg.trigger })
    })

    ws.on('ai_dialogue_chunk', (msg: Extract<WsIncomingMsg, { type: 'ai_dialogue_chunk' }>) => {
      const last = dialogueLines.value[dialogueLines.value.length - 1]
      if (last) {
        last.text += msg.text
      } else {
        dialogueLines.value.push({ id: ++dialogueSeq, text: msg.text, trigger: 'chat' })
      }
    })

    ws.on('ai_dialogue_end', () => {
      dialogueActive.value = false
    })

    ws.on('punishment_required', (msg) => {
      isLocked.value = true
      activePunishment.value = msg.punishment_type
    })

    ws.on('punishment_cleared', (msg) => {
      // 解锁（或护盾抵消）后同步清空惩罚现场与走神累计，防止前端立即复弹进度条
      applyPunishmentCleared(msg.strikes_left)
    })

    ws.on('room_locked', () => {
      isLocked.value = true
    })

    ws.on('sentinel_alert', (msg) => {
      sentinelMessage.value = msg.message
    })

    ws.on('room_interaction', (msg) => {
      // 单一权威入口：按 message_id 幂等去重后写入缓存，供 HUD 按游标消费；
      // 房主/学员/跨房均由后端广播与房间隔离约束，前端只负责可靠渲染。
      noteInteraction({
        kind: msg.kind,
        from: msg.from,
        text: msg.text,
        sound: msg.sound,
        message_id: msg.message_id,
        user_id: msg.user_id,
      })
    })

    ws.on('watcher_update', (msg) => {
      watcherCount.value = msg.watcher_count
      watchers.value = msg.watchers
    })

    // 结算完成后刷新成长能力由调用方（视图层）结合 userStore 处理
    ws.on('session_finished', () => {
      isLocked.value = false
    })

    // 房主重连后的现场快照：恢复目标/机会/锁房/惩罚与历史对话（场景由视图层还原）
    ws.on('room_state_sync', (msg) => {
      applySceneSnapshot(msg)
      applyDialogueHistory(msg.dialogue_history)
    })
  }

  /**
   * 应用惩罚解锁后的统一现场：清空惩罚、恢复机会、清零走神累计。
   * WS 回执与 HTTP 兜底共用，避免两处状态更新逻辑漂移。
   */
  function applyPunishmentCleared(strikesLeftValue?: number): void {
    isLocked.value = false
    activePunishment.value = ''
    strikesLeft.value = typeof strikesLeftValue === 'number' ? strikesLeftValue : 3
    distractedStreakSec.value = 0
  }

  /**
   * 惩罚提交 HTTP 兜底：WS 未连接或未收到回执时调用。
   * 后端复用同一套校验并广播 punishment_cleared；若此刻 WS 已断开收不到广播，
   * 则依据 HTTP 返回的 passed 在本地同步解锁，彻底避免按钮卡死。
   */
  async function submitPunishmentHttp(
    punishmentType: string,
    content: string,
  ): Promise<{ passed: boolean; reason: string }> {
    const res = await roomApi.punishmentSubmit(roomId.value, {
      punishment_type: punishmentType,
      content,
    })
    if (res.passed) {
      applyPunishmentCleared(res.strikes_left)
    }
    return { passed: res.passed, reason: res.reason ?? '' }
  }

  /** 上行发送包装 */
  function send(msg: WsOutgoingMsg): boolean {
    return socket.value?.send(msg) ?? false
  }

  /** WS 是否处于已连接状态，供提交惩罚时选择 WS 或 HTTP 通道 */
  function isSocketConnected(): boolean {
    return socket.value?.isConnected ?? false
  }

  /** 清空房间态（离开房间时调用） */
  function reset(): void {
    socket.value?.close()
    socket.value = null
    roomId.value = ''
    roomDetail.value = null
    goalText.value = ''
    strikesLeft.value = 3
    focusScore.value = null
    distractedStreakSec.value = 0
    strikeSeconds.value = 10
    isLocked.value = false
    activePunishment.value = ''
    watcherCount.value = 0
    watchers.value = []
    dialogueLines.value = []
    dialogueActive.value = false
    lastInteraction.value = null
    sentinelMessage.value = ''
    interactionSoundOn.value = true
    dialogueSeq = 0
    // 清空弹幕去重态：换房/重连后不与旧房间的消息 ID 冲突，缓存与游标归零
    interactionCache.value = []
    interactionSeq = 0
    hudCursorSeq = 0
    interactionHeadSeq.value = 0
    seenInteractionIds.clear()
    seenIdOrder.length = 0
  }

  return {
    roomId,
    roomDetail,
    goalText,
    strikesLeft,
    focusScore,
    distractedStreakSec,
    strikeSeconds,
    isLocked,
    activePunishment,
    watcherCount,
    watchers,
    dialogueLines,
    dialogueActive,
    lastInteraction,
    interactionCache,
    interactionHeadSeq,
    sentinelMessage,
    interactionSoundOn,
    socket,
    strikesUsed,
    subtitleText,
    createRoom,
    applyRoomDetail,
    applyRoomMeta,
    applySceneSnapshot,
    applyDialogueHistory,
    createSocket,
    send,
    drainHudInteractions,
    isSocketConnected,
    submitPunishmentHttp,
    reset,
  }
})
