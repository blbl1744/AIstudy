/**
 * 房间 WebSocket 管理类。
 * 职责：连接 /ws/room/{roomId}?token=、强类型消息分发事件总线、心跳保活、断线指数退避重连。
 * 后端下行消息全部经 dispatch 按 type 精确派发，业务侧只订阅自己关心的类型。
 */
import type {
  WsIncomingMsg,
  WsIncomingType,
  WsOutgoingMsg,
} from '@/types'

/** 订阅回调：按消息 type 精确推断出具体的下行消息类型 */
type Handler<K extends WsIncomingType> = (msg: Extract<WsIncomingMsg, { type: K }>) => void

/** 内部统一回调签名 */
type AnyHandler = (msg: WsIncomingMsg) => void

export interface RoomSocketOptions {
  /** 房间 ID */
  roomId: string
  /** Token 读取函数（从 userStore / localStorage 取，始终使用最新值） */
  getToken: () => string | null
  /** 心跳间隔，默认 15000ms */
  heartbeatInterval?: number
  /** 重连最大退避时间，默认 30000ms */
  maxReconnectDelay?: number
  /** 连接建立回调 */
  onOpen?: () => void
  /** 连接关闭回调 */
  onClose?: () => void
  /** 连接异常回调 */
  onError?: (event: Event) => void
}

export class RoomSocket {
  private readonly roomId: string
  private readonly getToken: () => string | null
  private readonly heartbeatInterval: number
  private readonly maxReconnectDelay: number
  private readonly onOpen?: () => void
  private readonly onClose?: () => void
  private readonly onError?: (event: Event) => void

  private ws: WebSocket | null = null
  private readonly handlers = new Map<WsIncomingType, Set<AnyHandler>>()
  private heartbeatTimer: number | null = null
  private reconnectTimer: number | null = null
  private reconnectAttempts = 0
  /** 主动关闭标记：主动 close 时不触发自动重连 */
  private manualClosed = false

  constructor(options: RoomSocketOptions) {
    this.roomId = options.roomId
    this.getToken = options.getToken
    this.heartbeatInterval = options.heartbeatInterval ?? 15000
    this.maxReconnectDelay = options.maxReconnectDelay ?? 30000
    this.onOpen = options.onOpen
    this.onClose = options.onClose
    this.onError = options.onError
  }

  /** 当前是否处于已连接状态 */
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  /** 订阅某类下行消息，返回取消订阅函数 */
  on<K extends WsIncomingType>(type: K, handler: Handler<K>): () => void {
    let bucket = this.handlers.get(type)
    if (!bucket) {
      bucket = new Set<AnyHandler>()
      this.handlers.set(type, bucket)
    }
    bucket.add(handler as AnyHandler)
    return () => this.off(type, handler)
  }

  /** 取消订阅某类下行消息 */
  off<K extends WsIncomingType>(type: K, handler: Handler<K>): void {
    this.handlers.get(type)?.delete(handler as AnyHandler)
  }

  /** 按 type 精确派发消息到订阅者 */
  private dispatch(msg: WsIncomingMsg): void {
    this.handlers.get(msg.type)?.forEach((handler) => handler(msg))
  }

  /** 建立连接；重复调用会先关闭旧连接 */
  connect(): void {
    this.manualClosed = false
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return
    }
    this.clearReconnectTimer()

    const token = this.getToken() ?? ''
    const scheme = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss://' : 'ws://'
    const host = typeof window !== 'undefined' ? window.location.host : '127.0.0.1:5173'
    const url = `${scheme}${host}/ws/room/${encodeURIComponent(this.roomId)}?token=${encodeURIComponent(token)}`

    const ws = new WebSocket(url)
    this.ws = ws

    ws.onopen = () => {
      this.reconnectAttempts = 0
      this.startHeartbeat()
      this.onOpen?.()
    }

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as WsIncomingMsg
        if (msg && typeof msg === 'object' && 'type' in msg) {
          this.dispatch(msg)
        }
      } catch {
        // 忽略了无法解析或非预期结构的消息，避免单条脏消息中断整个长连接
      }
    }

    ws.onerror = (event: Event) => {
      this.onError?.(event)
    }

    ws.onclose = () => {
      this.stopHeartbeat()
      this.onClose?.()
      if (!this.manualClosed) {
        this.scheduleReconnect()
      }
    }
  }

  /** 发送强类型上行消息；未连接时返回 false */
  send(msg: WsOutgoingMsg): boolean {
    if (!this.isConnected) {
      return false
    }
    this.ws?.send(JSON.stringify(msg))
    return true
  }

  /** 主动关闭连接并停止一切重连/心跳计时器 */
  close(): void {
    this.manualClosed = true
    this.stopHeartbeat()
    this.clearReconnectTimer()
    if (this.ws) {
      this.ws.onclose = null
      this.ws.close()
      this.ws = null
    }
  }

  /** 启动心跳保活 */
  private startHeartbeat(): void {
    this.stopHeartbeat()
    this.heartbeatTimer = window.setInterval(() => {
      this.send({ type: 'ping' })
    }, this.heartbeatInterval)
  }

  /** 停止心跳保活 */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      window.clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  /** 指数退避重连：delay = min(基础值 * 2^尝试次数, 上限) */
  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) {
      return
    }
    const base = 1000
    const delay = Math.min(base * (2 ** this.reconnectAttempts), this.maxReconnectDelay)
    this.reconnectAttempts += 1
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null
      this.connect()
    }, delay)
  }

  /** 清除重连计时器 */
  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }
}

export default RoomSocket
