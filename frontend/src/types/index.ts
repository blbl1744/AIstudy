/**
 * 全局类型契约层。
 * 所有字段严格对齐后端 M1/M2（见 app/routers/* 与 DEVELOPMENT_PLAN.md 第三节），
 * 前端一律以这些接口约束 WebSocket 消息与场景配置，避免联调字段错配。
 */

// ===== 通用枚举：与后端渲染白名单 / 业务枚举保持一致 =====

/** 房间模式：AI 单机 / 1v1 互督 / 多人围观 */
export type RoomMode = 'ai_solo' | 'peer_1v1' | 'peer_group'

/** 建设性惩罚类型 */
export type PunishmentType = 'feynman_recall' | 'emergency_sprint' | 'goal_breakdown'

/** 场景 HUD 风格白名单 */
export type HudStyle = 'scifi' | 'military' | 'academy' | 'cyber'

/** 粒子特效预设白名单 */
export type ParticlePreset = 'stars' | 'embers' | 'rain' | 'matrix' | 'dust'

/** 环境白噪音预设白名单 */
export type AmbientSound = 'spaceship_hum' | 'radar_tick' | 'library_rain' | 'fireplace' | 'none'

/** 督学官互动音效白名单（与后端 SOUND_WHITELIST 一致） */
export type SoundEffect = 'knock_board' | 'clear_throat' | 'alarm' | 'cheer'

/** 指挥官对话触发来源 */
export type DialogueTrigger = 'proactive' | 'distraction' | 'chat'

// ===== AI 场景配置契约（后端 SceneConfig 标准结构） =====

/** 场景主题：主色 / 警戒色 / 背景渐变 / HUD 风格 */
export interface SceneTheme {
  primary_color: string
  alert_color: string
  bg_gradient: string
  hud_style: HudStyle
}

/** 场景氛围：粒子预设 / 环境音 / 音色标识 */
export interface SceneAtmosphere {
  particle_preset: ParticlePreset
  ambient_sound: AmbientSound
  voice_id: string
}

/** 三次机会对应的警告台词 */
export interface SceneStrikes {
  strike_1: string
  strike_2: string
  strike_3: string
}

/** 完整场景包：内置剧本与大模型编译结果统一输出此结构供渲染器消费 */
export interface SceneConfig {
  scene_id: string
  source_work: string
  scene_title: string
  commander_title: string
  commander_persona: string
  opening_line: string
  theme: SceneTheme
  atmosphere: SceneAtmosphere
  strikes: SceneStrikes
  victory_line: string
}

// ===== 用户与战报契约 =====

/** 单局督学战报（GET /api/users/me/profile 内 sessions 元素） */
export interface StudySession {
  id: number
  user_id: number
  mode: RoomMode
  scene_title: string
  goal_text: string
  duration_minutes: number
  avg_score: number | null
  strikes_used: number
  punishment_type: string
  status: 'completed' | 'failed'
  created_at: string
}

/** 个人成长档案（GET /api/users/me/profile 返回体） */
export interface UserProfile {
  id: number
  username: string
  nickname: string
  total_focus_minutes: number
  rank_title: string
  shield_count: number
  streak_clean_sessions: number
  created_at: string
  sessions: StudySession[]
}

// ===== 认证接口契约 =====

/** 注册请求：用户名 3~32、密码 6~64、昵称 1~32（后端做字符白名单校验） */
export interface RegisterRequest {
  username: string
  password: string
  nickname: string
}

/** 登录请求 */
export interface LoginRequest {
  username: string
  password: string
}

/** 登录响应（JWT 无角色载荷） */
export interface LoginResponse {
  access_token: string
  token_type: string
  user_id: number
  username: string
  nickname: string
}

/** 注册响应 */
export interface RegisterResponse {
  message: string
  user_id: number
}

// ===== 房间与广场契约 =====

/** 创建房间请求（POST /api/rooms） */
export interface CreateRoomRequest {
  mode: RoomMode
  is_public: boolean
  scene_id: string
  work_name: string
  scene_title: string
  goal_text: string
  max_watchers: number
  punishment_types: PunishmentType[]
}

/** 创建房间响应（后端回显房间字典，剔除下划线内部字段） */
export interface RoomDetail {
  room_id: string
  host_id: number
  host_nickname: string
  mode: RoomMode
  is_public: boolean
  scene_id: string
  work_name: string
  scene_title: string
  goal_text: string
  max_watchers: number
  punishment_types: PunishmentType[]
  created_at: number
  strikes_left: number
  distracted_streak_sec: number
  scene_config: SceneConfig
  dialogue_history: unknown[]
  is_locked: boolean
  strikes_used: number
  active_punishment: string
  last_frame_ts: number | null
  last_frame_b64: string | null
  score_history: (number | null)[]
  watchers: Record<string, string>
}

/** 公告广场房间条目（GET /api/rooms/plaza） */
export interface PlazaRoom {
  room_id: string
  host_id: number
  host_nickname: string
  mode: RoomMode
  scene_title: string
  goal_text: string
  elapsed_seconds: number
  watcher_count: number
  max_watchers: number
  /** 房主是否在线：false 表示房主暂离但房间仍保留，供本人「返回学习室」 */
  host_online: boolean
}

/** 房间元信息（GET /api/rooms/{room_id}），供督学官校验归属，也供房主暂离后恢复现场 */
export interface RoomMeta {
  room_id: string
  host_id: number
  host_nickname: string
  scene_title: string
  goal_text: string
  mode: RoomMode
  is_public: boolean
  /** 房主在线标记：暂离为 false，重连恢复为 true */
  host_online: boolean
  /** 当前挂载剧本，供重返恢复场景 */
  scene_config: SceneConfig
  strikes_left: number
  is_locked: boolean
  active_punishment: string
}

// ===== WebSocket 消息契约（按 type 字段做可辨识联合，实现强类型分发） =====

/** 客户端上行：学习者推帧 */
export interface WsFrameMsg {
  type: 'frame'
  image: string
  scene?: string
}

/** 客户端上行：向指挥官求助 / 互动 */
export interface WsUserMessageMsg {
  type: 'user_message'
  text: string
  voice_enabled?: boolean
}

/** 客户端上行：提交惩罚解锁内容 */
export interface WsPunishmentSubmitMsg {
  type: 'punishment_submit'
  punishment_type: PunishmentType | string
  content: string
}

/** 客户端上行：督学官弹幕 */
export interface WsBarrageMsg {
  type: 'barrage'
  text: string
}

/** 客户端上行：督学官音效 */
export interface WsSoundEffectMsg {
  type: 'sound_effect'
  sound: SoundEffect
}

/** 客户端上行：正常结束本局 */
export interface WsFinishSessionMsg {
  type: 'finish_session'
}

/** 客户端上行：应用层心跳保活（后端对未知 type 静默忽略，仅用于维持连接活跃） */
export interface WsPingMsg {
  type: 'ping'
}

/** 客户端上行：房主主动离开房间（后端据此销毁房间与画面缓存） */
export interface WsLeaveRoomMsg {
  type: 'leave_room'
}

/** 客户端上行消息联合类型 */
export type WsOutgoingMsg =
  | WsFrameMsg
  | WsUserMessageMsg
  | WsPunishmentSubmitMsg
  | WsBarrageMsg
  | WsSoundEffectMsg
  | WsFinishSessionMsg
  | WsPingMsg
  | WsLeaveRoomMsg

/** 服务端下行：姿态分析结果 */
export interface WsAnalysisUpdateMsg {
  type: 'analysis_update'
  focus_score?: number | null
  action_label?: string | null
  annotated_image?: string | null
  strikes_left?: number
  distracted_streak_sec?: number
  skipped?: boolean
}

/** 服务端下行：指挥官对话开始 */
export interface WsDialogueStartMsg {
  type: 'ai_dialogue_start'
  trigger: DialogueTrigger
}

/** 服务端下行：指挥官字幕分片 */
export interface WsDialogueChunkMsg {
  type: 'ai_dialogue_chunk'
  text: string
}

/** 服务端下行：指挥官语音分片（base64 音频） */
export interface WsAudioChunkMsg {
  type: 'ai_audio_chunk'
  audio_b64: string
}

/** 服务端下行：指挥官对话结束 */
export interface WsDialogueEndMsg {
  type: 'ai_dialogue_end'
  trigger: DialogueTrigger
}

/** 服务端下行：要求提交惩罚解锁 */
export interface WsPunishmentRequiredMsg {
  type: 'punishment_required'
  punishment_type: PunishmentType | string
  prompt: string
}

/** 服务端下行：惩罚解锁成功 */
export interface WsPunishmentClearedMsg {
  type: 'punishment_cleared'
  shield_saved: boolean
  strikes_left: number
}

/** 服务端下行：惩罚提交校验不通过 */
export interface WsPunishmentRejectedMsg {
  type: 'punishment_rejected'
  reason: string
}

/** 服务端下行：AI 哨兵预警（仅推督学官） */
export interface WsSentinelAlertMsg {
  type: 'sentinel_alert'
  message: string
  distracted_streak_sec: number
}

/** 服务端下行：房间互动广播（弹幕 / 音效） */
export interface WsRoomInteractionMsg {
  type: 'room_interaction'
  kind: 'barrage' | 'sound_effect'
  from: string
  text?: string
  sound?: string
}

/** 服务端下行：督学官名单更新（推房主） */
export interface WsWatcherUpdateMsg {
  type: 'watcher_update'
  watcher_count: number
  watchers: { user_id: number; nickname: string }[]
}

/** 服务端下行：锁房提示 */
export interface WsRoomLockedMsg {
  type: 'room_locked'
  msg: string
}

/** 服务端下行：本局结算完成 */
export interface WsSessionFinishedMsg {
  type: 'session_finished'
  duration_minutes: number
  avg_score: number | null
  total_focus_minutes: number
  rank_title: string
  streak_clean_sessions: number
  shield_count: number
}

/** 服务端下行：房主离开导致房间关闭（保留类型兼容，暂离改用 host_status） */
export interface WsRoomClosedMsg {
  type: 'room_closed'
  msg: string
}

/** 服务端下行：房主在线状态变化（暂离 / 返回），推送给在线督学官 */
export interface WsHostStatusMsg {
  type: 'host_status'
  online: boolean
  msg?: string
}

/** 指挥官历史对话条目，结构对齐后端 ai_commander 的 dialogue_history */
export interface DialogueHistoryItem {
  trigger: string
  user_input: string
  text: string
}

/** 服务端下行：房主重连后的现场快照，恢复目标 / 机会 / 锁房 / 惩罚 / 场景与历史对话 */
export interface WsRoomStateSyncMsg {
  type: 'room_state_sync'
  goal_text: string
  strikes_left: number
  is_locked: boolean
  active_punishment: string
  scene_config: SceneConfig
  dialogue_history: DialogueHistoryItem[]
}

/** 服务端下行消息联合类型 */
export type WsIncomingMsg =
  | WsAnalysisUpdateMsg
  | WsDialogueStartMsg
  | WsDialogueChunkMsg
  | WsAudioChunkMsg
  | WsDialogueEndMsg
  | WsPunishmentRequiredMsg
  | WsPunishmentClearedMsg
  | WsPunishmentRejectedMsg
  | WsSentinelAlertMsg
  | WsRoomInteractionMsg
  | WsWatcherUpdateMsg
  | WsRoomLockedMsg
  | WsSessionFinishedMsg
  | WsRoomClosedMsg
  | WsHostStatusMsg
  | WsRoomStateSyncMsg

/** 服务端下行消息的 type 字段联合，供事件总线键类型约束 */
export type WsIncomingType = WsIncomingMsg['type']


// ===== M5 战报明细契约（GET /api/stats/sessions/{id}） =====

/** 单局专注度采样点 */
export interface FocusSample {
  created_at: string
  focus_score: number | null
}

/** 单局战报摘要（与后端 summary 字段一一对应） */
export interface SessionSummary {
  id: number
  mode: RoomMode
  scene_title: string
  goal_text: string
  duration_minutes: number
  avg_score: number | null
  strikes_used: number
  punishment_type: string
  status: 'completed' | 'failed'
  created_at: string
}

/** 单局战报详情响应：摘要 + 采样点列表 */
export interface SessionDetail {
  summary: SessionSummary
  points: FocusSample[]
}
