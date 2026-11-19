<script setup lang="ts">
// 指挥官实时对话面板：流式字幕打字机 + 双模语音播报队列 + 文字/语音双向互动
// 严格对接 roomStore 的 ai_dialogue_start / ai_dialogue_chunk / ai_dialogue_end 状态流，
// ai_audio_chunk 未在 store 订阅，此处在组件层直接订阅 socket 并解码播放。
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoomStore } from '@/stores/room'
import { systemApi } from '@/api/http'
import type { RoomSocket } from '@/api/roomSocket'

const room = useRoomStore()

// 大模型在线状态：默认按离线处理，挂载后由 /api/health 校准
const llmEnabled = ref(false)
onMounted(() => {
  systemApi.health()
    .then((data) => { llmEnabled.value = Boolean(data.llm_enabled) })
    .catch(() => { llmEnabled.value = false })
})

// ===== 一、流式字幕打字机 =====
interface Bubble {
  id: number
  trigger: string
  fullText: string
  shownText: string
}
const bubbles = ref<Bubble[]>([])
const listEl = ref<HTMLElement | null>(null)

// 将 store 的对话行同步为气泡，逐字补齐由定时器驱动，实现打字机效果
watch(
  () => room.dialogueLines,
  (lines) => {
    if (lines.length === 0) {
      bubbles.value = []
      return
    }
    for (const line of lines) {
      const found = bubbles.value.find((b) => b.id === line.id)
      if (!found) {
        bubbles.value.push({ id: line.id, trigger: line.trigger, fullText: line.text, shownText: '' })
      } else {
        found.fullText = line.text
      }
    }
  },
  { deep: true, immediate: true },
)

function scrollToBottom(): void {
  void nextTick(() => {
    if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight
  })
}

// 触发类型中文标签
function triggerLabel(trigger: string): string {
  if (trigger === 'distraction') return '纠偏指令'
  if (trigger === 'proactive') return '主动指令'
  return '对话'
}

let typeTimer: number | null = null
onMounted(() => {
  // 每 24ms 为每个未显示完的气泡补一个字，形成平滑打字机效果
  typeTimer = window.setInterval(() => {
    let changed = false
    for (const bubble of bubbles.value) {
      if (bubble.shownText.length < bubble.fullText.length) {
        bubble.shownText = bubble.fullText.slice(0, bubble.shownText.length + 1)
        changed = true
      }
    }
    if (changed) scrollToBottom()
  }, 24)
})

// 当前正在播放的字幕（最后一个气泡的已显示文本）
const currentSubtitle = ref('')
watch(bubbles, (list) => {
  const last = list[list.length - 1]
  currentSubtitle.value = last ? last.shownText : ''
}, { deep: true })

// ===== 二、双模语音播报队列 =====
const voiceEnabled = ref(true) // 顶部一键开关

// 后端 TTS 音频（Base64）顺序播放队列
const audioQueue: string[] = []
let audioPlaying = false
let hasBackendAudio = false // 本轮对话是否收到后端音频，用于决定是否降级浏览器 TTS

function playNextAudio(): void {
  if (audioPlaying || audioQueue.length === 0) return
  const b64 = audioQueue.shift()
  if (!b64) return
  audioPlaying = true
  const audio = new Audio('data:audio/mpeg;base64,' + b64)
  const finish = () => {
    audioPlaying = false
    playNextAudio()
  }
  audio.onended = finish
  audio.onerror = finish
  void audio.play().catch(finish)
}

// 浏览器 Web Speech 降级朗读（中文）
function speakByBrowser(text: string): void {
  if (!('speechSynthesis' in window)) return
  try {
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text.slice(0, 500))
    utterance.lang = 'zh-CN'
    utterance.rate = 1.05
    utterance.pitch = 1
    window.speechSynthesis.speak(utterance)
  } catch {
    // 浏览器语音合成不可用时静默降级为纯字幕
  }
}

// 对话结束：若本轮未收到后端音频且语音开启，则用浏览器 TTS 逐句朗读完整文本
watch(
  () => room.dialogueActive,
  (active, prev) => {
    if (prev === true && active === false && voiceEnabled.value && !hasBackendAudio) {
      const last = room.dialogueLines[room.dialogueLines.length - 1]
      if (last?.text) speakByBrowser(last.text)
    }
  },
)

// 关闭语音时立即停止朗读并清空待播音频
watch(voiceEnabled, (on) => {
  if (!on) {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel()
    audioQueue.length = 0
  }
})

// socket 可能在组件挂载后才创建，故监听其变化来绑定音频订阅
let unsubscribers: Array<() => void> = []
watch(
  () => room.socket,
  (ws: RoomSocket | null) => {
    unsubscribers.forEach((u) => u())
    unsubscribers = []
    if (!ws) return
    unsubscribers.push(ws.on('ai_dialogue_start', () => { hasBackendAudio = false }))
    unsubscribers.push(
      ws.on('ai_audio_chunk', (msg) => {
        if (!msg.audio_b64) return
        hasBackendAudio = true
        if (!voiceEnabled.value) return
        audioQueue.push(msg.audio_b64)
        playNextAudio()
      }),
    )
  },
  { immediate: true },
)

// ===== 三、学习者双向互动 =====
const inputText = ref('')

/** 发送文字求助：上行 user_message 契约 */
function sendText(raw: string): void {
  const text = raw.trim()
  if (!text) return
  room.send({ type: 'user_message', text, voice_enabled: voiceEnabled.value })
  inputText.value = ''
}

// 语音识别（按住说话）：不支持的浏览器自动隐藏按钮并保留文字输入
interface SpeechResultEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }>>
}
interface SpeechRecognitionLike {
  lang: string
  continuous: boolean
  interimResults: boolean
  onresult: ((event: SpeechResultEvent) => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
  start(): void
  stop(): void
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike

const speechWindow = window as unknown as {
  SpeechRecognition?: SpeechRecognitionCtor
  webkitSpeechRecognition?: SpeechRecognitionCtor
}
const RecognitionCtor = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition
const speechSupported = ref(Boolean(RecognitionCtor))
const isListening = ref(false)
let recognizer: SpeechRecognitionLike | null = null

function startListen(): void {
  if (!RecognitionCtor || isListening.value) return
  const instance = new RecognitionCtor()
  instance.lang = 'zh-CN'
  instance.continuous = false
  instance.interimResults = false
  instance.onresult = (event) => {
    const transcript = event.results[0]?.[0]?.transcript?.trim() ?? ''
    if (transcript) sendText(transcript)
  }
  instance.onend = () => { isListening.value = false; recognizer = null }
  instance.onerror = () => { isListening.value = false; recognizer = null }
  recognizer = instance
  isListening.value = true
  try {
    instance.start()
  } catch {
    isListening.value = false
    recognizer = null
  }
}

function stopListen(): void {
  try {
    recognizer?.stop()
  } catch {
    // 识别已结束时忽略
  }
}

onBeforeUnmount(() => {
  if (typeTimer !== null) window.clearInterval(typeTimer)
  unsubscribers.forEach((u) => u())
  if ('speechSynthesis' in window) window.speechSynthesis.cancel()
  audioQueue.length = 0
})
</script>

<template>
  <div class="chat-panel">
    <!-- 顶部频道栏：标题 + 语音播报一键开关 -->
    <div class="chat-header">
      <div class="chat-header__title">
        <span class="chat-header__dot"></span>
        指挥官频道
        <!-- 离线剧本模式提示：未配置 LLM Key 时温和告知，不影响正常使用 -->
        <span v-if="!llmEnabled" class="chat-offline-badge">当前为离线剧本模式，配置 LLM Key 可开启自由问答</span>
      </div>
      <button
        type="button"
        class="voice-toggle"
        :class="{ 'voice-toggle--on': voiceEnabled }"
        @click="voiceEnabled = !voiceEnabled"
      >
        语音播报 {{ voiceEnabled ? '开' : '关' }}
      </button>
    </div>

    <!-- 实时字幕条：当前指令逐字显示 -->
    <div v-if="currentSubtitle" class="live-subtitle">{{ currentSubtitle }}</div>

    <!-- 对话气泡列表 -->
    <div ref="listEl" class="chat-list">
      <div
        v-for="bubble in bubbles"
        :key="bubble.id"
        class="chat-bubble"
        :class="`chat-bubble--${bubble.trigger}`"
      >
        <div class="chat-bubble__meta">
          <span class="chat-bubble__role">指挥官</span>
          <span class="chat-bubble__tag">{{ triggerLabel(bubble.trigger) }}</span>
        </div>
        <div class="chat-bubble__text">
          {{ bubble.shownText }}<span class="chat-bubble__caret" v-if="bubble.shownText.length < bubble.fullText.length">▌</span>
        </div>
      </div>
      <div v-if="bubbles.length === 0" class="chat-empty">等待指挥官下达指令…</div>
    </div>

    <!-- 底部双向互动区：文字快捷输入 + 按住说话 -->
    <div class="chat-compose">
      <input
        v-model="inputText"
        class="chat-input"
        type="text"
        maxlength="200"
        placeholder="向指挥官求助，回车发送…"
        @keyup.enter="sendText(inputText)"
      />
      <button type="button" class="chat-send" @click="sendText(inputText)">发送</button>
      <button
        v-if="speechSupported"
        type="button"
        class="chat-ptt"
        :class="{ 'chat-ptt--active': isListening }"
        @pointerdown.prevent="startListen"
        @pointerup="stopListen"
        @pointerleave="stopListen"
      >
        {{ isListening ? '松开结束' : '按住说话' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 260px;
  background: rgba(0, 0, 0, 0.35);
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 4px;
  color: var(--scene-primary, #00e5ff);
  box-sizing: border-box;
  overflow: hidden;
}

/* 顶部频道栏 */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid color-mix(in srgb, var(--scene-primary, #00e5ff) 40%, transparent);
  font-size: 13px;
}
.chat-header__title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  letter-spacing: 1px;
}
.chat-header__dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--scene-primary, #00e5ff);
  box-shadow: 0 0 8px var(--scene-primary, #00e5ff);
}
.chat-offline-badge {
  font-size: 11px;
  font-weight: 400;
  letter-spacing: 0;
  color: #b9a674;
  background: rgba(185, 166, 116, 0.12);
  border: 1px solid rgba(185, 166, 116, 0.35);
  border-radius: 999px;
  padding: 2px 9px;
  white-space: nowrap;
}
.voice-toggle {
  padding: 3px 10px;
  font-size: 12px;
  color: var(--scene-primary, #00e5ff);
  background: transparent;
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 2px;
  cursor: pointer;
}
.voice-toggle--on {
  background: color-mix(in srgb, var(--scene-primary, #00e5ff) 22%, transparent);
}

/* 实时字幕条 */
.live-subtitle {
  padding: 8px 12px;
  font-size: 15px;
  font-weight: 600;
  line-height: 1.5;
  background: color-mix(in srgb, var(--scene-primary, #00e5ff) 10%, transparent);
  border-bottom: 1px solid color-mix(in srgb, var(--scene-primary, #00e5ff) 25%, transparent);
  min-height: 34px;
}

/* 对话气泡列表 */
.chat-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.chat-bubble {
  max-width: 88%;
  align-self: flex-start;
  padding: 8px 12px;
  border-left: 3px solid var(--scene-primary, #00e5ff);
  background: rgba(255, 255, 255, 0.05);
  border-radius: 2px 6px 6px 2px;
}
.chat-bubble--distraction {
  border-left-color: var(--scene-alert, #ff2a55);
  background: color-mix(in srgb, var(--scene-alert, #ff2a55) 10%, transparent);
}
.chat-bubble__meta {
  display: flex;
  gap: 8px;
  font-size: 11px;
  opacity: 0.75;
  margin-bottom: 4px;
}
.chat-bubble__role { font-weight: 700; }
.chat-bubble__tag {
  padding: 0 6px;
  border: 1px solid currentColor;
  border-radius: 2px;
}
.chat-bubble__text {
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
  white-space: pre-wrap;
}
.chat-bubble__caret {
  animation: blink 0.8s steps(1) infinite;
}
.chat-empty {
  margin: auto;
  font-size: 13px;
  opacity: 0.5;
}
@keyframes blink {
  50% { opacity: 0; }
}

/* 底部互动区 */
.chat-compose {
  display: flex;
  gap: 8px;
  padding: 10px 12px;
  border-top: 1px solid color-mix(in srgb, var(--scene-primary, #00e5ff) 40%, transparent);
}
.chat-input {
  flex: 1;
  min-width: 0;
  padding: 7px 10px;
  font-size: 13px;
  color: var(--scene-primary, #00e5ff);
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid color-mix(in srgb, var(--scene-primary, #00e5ff) 50%, transparent);
  border-radius: 3px;
  outline: none;
}
.chat-input::placeholder { color: color-mix(in srgb, var(--scene-primary, #00e5ff) 55%, transparent); }
.chat-send,
.chat-ptt {
  padding: 7px 12px;
  font-size: 13px;
  color: var(--scene-primary, #00e5ff);
  background: transparent;
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 3px;
  cursor: pointer;
  white-space: nowrap;
  user-select: none;
  touch-action: none;
}
.chat-send:hover,
.chat-ptt:hover { background: color-mix(in srgb, var(--scene-primary, #00e5ff) 18%, transparent); }
.chat-ptt--active {
  background: color-mix(in srgb, var(--scene-alert, #ff2a55) 35%, transparent);
  border-color: var(--scene-alert, #ff2a55);
}
</style>
