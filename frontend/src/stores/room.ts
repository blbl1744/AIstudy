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
  const isLocked = ref(false)
  const activePunishment = ref('')
  const watcherCount = ref(0)
  const watchers = ref<{ user_id: number; nickname: string }[]>([])
  const dialogueLines = ref<DialogueLine[]>([])
  const dialogueActive = ref(false)
  const lastInteraction = ref<RoomInteraction | null>(null)
  const sentinelMessage = ref('')
  // WebSocket 实例非响应式对象，用 shallowRef 避免深层代理
  const socket = shallowRef<RoomSocket | null>(null)

  let dialogueSeq = 0

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
      isLocked.value = false
      activePunishment.value = ''
      strikesLeft.value = typeof msg.strikes_left === 'number' ? msg.strikes_left : 3
      distractedStreakSec.value = 0
    })

    ws.on('room_locked', () => {
      isLocked.value = true
    })

    ws.on('sentinel_alert', (msg) => {
      sentinelMessage.value = msg.message
    })

    ws.on('room_interaction', (msg) => {
      lastInteraction.value = {
        kind: msg.kind,
        from: msg.from,
        text: msg.text,
        sound: msg.sound,
      }
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

  /** 上行发送包装 */
  function send(msg: WsOutgoingMsg): boolean {
    return socket.value?.send(msg) ?? false
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
    isLocked.value = false
    activePunishment.value = ''
    watcherCount.value = 0
    watchers.value = []
    dialogueLines.value = []
    dialogueActive.value = false
    lastInteraction.value = null
    sentinelMessage.value = ''
    dialogueSeq = 0
  }

  return {
    roomId,
    roomDetail,
    goalText,
    strikesLeft,
    focusScore,
    distractedStreakSec,
    isLocked,
    activePunishment,
    watcherCount,
    watchers,
    dialogueLines,
    dialogueActive,
    lastInteraction,
    sentinelMessage,
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
    reset,
  }
})
