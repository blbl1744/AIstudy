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
  DialogueTrigger,
  RoomDetail,
  WsIncomingMsg,
  WsOutgoingMsg,
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
    strikesLeft.value = room.strikes_left ?? 3
    isLocked.value = Boolean(room.is_locked)
    activePunishment.value = room.active_punishment ?? ''
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
      isLocked.value = false
      activePunishment.value = ''
      if (typeof msg.strikes_left === 'number') {
        strikesLeft.value = msg.strikes_left
      }
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
    createSocket,
    send,
    reset,
  }
})
