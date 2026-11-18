<script setup lang="ts">
/**
 * 沉浸式 HUD 覆盖层：三次机会生命值、实时专注评分、走神进度、专注计时、
 * 在线督学官角标，以及房间互动弹幕飘过与程序化互动音效。
 * 所有数据来自 roomStore；弹幕与音效在组件层直接订阅 socket，避免快捷广播被状态覆盖丢失。
 * 全程零外部资源，音效由 Web Audio API 实时合成。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoomStore } from '@/stores/room'
import type { RoomSocket } from '@/api/roomSocket'
import type { SoundEffect } from '@/types'

const room = useRoomStore()

// ===== 一、三次机会生命值（扣血震动） =====
const hearts = computed(() => [0, 1, 2].map((i) => i < room.strikesLeft))
const shaking = ref(false)
let shakeTimer: number | null = null

// 生命值减少时触发屏幕红色震动动效
watch(
  () => room.strikesLeft,
  (now, before) => {
    if (now < before) {
      shaking.value = true
      if (shakeTimer !== null) window.clearTimeout(shakeTimer)
      shakeTimer = window.setTimeout(() => { shaking.value = false }, 520)
    }
  },
)

// ===== 二、专注评分与走神进度 =====
const scoreText = computed(() => room.focusScore === null ? '--' : String(Math.round(room.focusScore)))
// 走神判定阈值 10 秒，与后端防误判缓冲一致；进度封顶 100%
const distractedPercent = computed(() => Math.min(100, Math.round((room.distractedStreakSec / 10) * 100)))

// ===== 三、本局专注计时器 =====
const elapsedSeconds = ref(0)
let clockTimer: number | null = null
const clockText = computed(() => {
  const m = Math.floor(elapsedSeconds.value / 60).toString().padStart(2, '0')
  const s = (elapsedSeconds.value % 60).toString().padStart(2, '0')
  return `${m}:${s}`
})

// ===== 四、实时弹幕飘过（纯插值渲染，严禁 v-html） =====
interface FloatingBarrage { id: number; text: string }
const barrages = ref<FloatingBarrage[]>([])
let barrageSeq = 0

function pushBarrage(text: string): void {
  const item: FloatingBarrage = { id: ++barrageSeq, text }
  barrages.value.push(item)
  // 动画 6 秒后移除该条弹幕，防止节点无限累积
  window.setTimeout(() => {
    barrages.value = barrages.value.filter((b) => b.id !== item.id)
  }, 6000)
}

// ===== 五、程序化互动音效（Web Audio 合成，零外部资源） =====
type AudioCtxCtor = typeof AudioContext
const audioWindow = window as unknown as { webkitAudioContext?: AudioCtxCtor }
const AudioCtx: AudioCtxCtor | undefined = window.AudioContext ?? audioWindow.webkitAudioContext
let audioCtx: AudioContext | null = null
const soundEnabled = ref(true)

function getCtx(): AudioContext | null {
  if (!AudioCtx || !soundEnabled.value) return null
  if (!audioCtx) audioCtx = new AudioCtx()
  if (audioCtx.state === 'suspended') void audioCtx.resume()
  return audioCtx
}

/** 合成一个振荡器音调，offset 为相对当前时间的延迟秒数 */
function playTone(
  ctx: AudioContext,
  freq: number,
  duration: number,
  type: OscillatorType,
  volume: number,
  offset = 0,
  endFreq?: number,
): void {
  const osc = ctx.createOscillator()
  osc.type = type
  const start = ctx.currentTime + offset
  osc.frequency.setValueAtTime(freq, start)
  if (endFreq !== undefined) osc.frequency.linearRampToValueAtTime(endFreq, start + duration)
  const gain = ctx.createGain()
  gain.gain.setValueAtTime(volume, start)
  gain.gain.exponentialRampToValueAtTime(0.001, start + duration)
  osc.connect(gain).connect(ctx.destination)
  osc.start(start)
  osc.stop(start + duration)
}

/** 合成一段经滤波的噪声，用于敲击/清嗓/欢呼质感 */
function playNoise(
  ctx: AudioContext,
  duration: number,
  filterType: BiquadFilterType,
  freq: number,
  volume: number,
): void {
  const length = Math.floor(ctx.sampleRate * duration)
  const buffer = ctx.createBuffer(1, length, ctx.sampleRate)
  const data = buffer.getChannelData(0)
  for (let i = 0; i < length; i += 1) data[i] = Math.random() * 2 - 1
  const source = ctx.createBufferSource()
  source.buffer = buffer
  const filter = ctx.createBiquadFilter()
  filter.type = filterType
  filter.frequency.value = freq
  const gain = ctx.createGain()
  const start = ctx.currentTime
  gain.gain.setValueAtTime(volume, start)
  gain.gain.exponentialRampToValueAtTime(0.001, start + duration)
  source.connect(filter).connect(gain).connect(ctx.destination)
  source.start(start)
  source.stop(start + duration)
}

/** 按互动音效类型分派到不同合成方案 */
function playSound(sound: string): void {
  const ctx = getCtx()
  if (!ctx) return
  if (sound === 'knock_board') {
    // 敲黑板：低频方波快速下坠 + 低通噪声短促敲击
    playTone(ctx, 180, 0.16, 'square', 0.25, 0, 90)
    playNoise(ctx, 0.09, 'lowpass', 700, 0.18)
  } else if (sound === 'clear_throat') {
    // 清嗓子：带通噪声摩擦 + 低频锯齿短鸣
    playNoise(ctx, 0.26, 'bandpass', 1200, 0.22)
    playTone(ctx, 130, 0.2, 'sawtooth', 0.08, 0, 90)
  } else if (sound === 'alarm') {
    // 防空警报：高低双音交替三轮
    for (let i = 0; i < 4; i += 1) {
      playTone(ctx, i % 2 === 0 ? 720 : 980, 0.28, 'sine', 0.16, i * 0.3)
    }
  } else if (sound === 'cheer') {
    // 欢呼：上升扫频 + 宽带噪声烘托
    playTone(ctx, 380, 0.9, 'triangle', 0.12, 0, 900)
    playNoise(ctx, 0.9, 'bandpass', 1800, 0.12)
  }
}

// socket 可能晚于组件挂载创建，故监听其变化来绑定互动订阅
let unsubscribers: Array<() => void> = []
watch(
  () => room.socket,
  (ws: RoomSocket | null) => {
    unsubscribers.forEach((u) => u())
    unsubscribers = []
    if (!ws) return
    unsubscribers.push(
      ws.on('room_interaction', (msg) => {
        if (msg.kind === 'barrage' && msg.text) {
          pushBarrage(msg.text)
        } else if (msg.kind === 'sound_effect' && msg.sound) {
          playSound(msg.sound as SoundEffect)
        }
      }),
    )
  },
  { immediate: true },
)

onMounted(() => {
  clockTimer = window.setInterval(() => { elapsedSeconds.value += 1 }, 1000)
})

onBeforeUnmount(() => {
  if (clockTimer !== null) window.clearInterval(clockTimer)
  if (shakeTimer !== null) window.clearTimeout(shakeTimer)
  unsubscribers.forEach((u) => u())
  if (audioCtx) {
    void audioCtx.close()
    audioCtx = null
  }
})
</script>

<template>
  <div class="hud" :class="{ 'hud--shake': shaking }">
    <!-- 左上：三次机会生命值 + 扣血震动 -->
    <div class="hud-block hud-lives">
      <span class="hud-label">生命值</span>
      <span
        v-for="(alive, i) in hearts"
        :key="i"
        class="hud-heart"
        :class="{ 'hud-heart--lost': !alive }"
      >{{ alive ? '◆' : '◇' }}</span>
    </div>

    <!-- 右上：在线督学官角标 -->
    <div class="hud-block hud-watchers">
      在线督学 <b>{{ room.watcherCount }}</b>
    </div>

    <!-- 顶部中央：专注评分与计时 -->
    <div class="hud-center">
      <div class="hud-score">专注 <b>{{ scoreText }}</b></div>
      <div class="hud-timer">{{ clockText }}</div>
    </div>

    <!-- 走神进度条 -->
    <div class="hud-distract">
      <span class="hud-label">走神预警</span>
      <div class="hud-bar">
        <div class="hud-bar__fill" :style="{ width: distractedPercent + '%' }"></div>
      </div>
    </div>

    <!-- 音效开关 -->
    <button
      type="button"
      class="hud-sound-toggle"
      :class="{ 'hud-sound-toggle--on': soundEnabled }"
      @click="soundEnabled = !soundEnabled"
    >音效 {{ soundEnabled ? '开' : '关' }}</button>

    <!-- 横向飘过弹幕层：纯文本插值 -->
    <div class="hud-barrage-layer">
      <div v-for="b in barrages" :key="b.id" class="hud-barrage">{{ b.text }}</div>
    </div>
  </div>
</template>

<style scoped>
.hud {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 5;
  color: var(--scene-primary, #00e5ff);
  font-family: inherit;
}
.hud--shake {
  animation: hud-shake 0.5s ease-in-out;
}
@keyframes hud-shake {
  0%, 100% { transform: translate(0, 0); }
  20% { transform: translate(-6px, 3px); }
  40% { transform: translate(6px, -3px); }
  60% { transform: translate(-4px, -2px); }
  80% { transform: translate(4px, 2px); }
}

.hud-block {
  position: absolute;
  top: 14px;
  padding: 6px 12px;
  background: rgba(0, 0, 0, 0.45);
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 3px;
  font-size: 13px;
}
.hud-lives { left: 14px; }
.hud-watchers { right: 14px; }
.hud-label {
  opacity: 0.7;
  margin-right: 8px;
  font-size: 12px;
}
.hud-heart {
  font-size: 18px;
  color: var(--scene-alert, #ff2a55);
  margin-right: 4px;
  text-shadow: 0 0 8px var(--scene-alert, #ff2a55);
}
.hud-heart--lost {
  color: rgba(255, 255, 255, 0.25);
  text-shadow: none;
}

.hud-center {
  position: absolute;
  top: 12px;
  left: 50%;
  transform: translateX(-50%);
  text-align: center;
  text-shadow: 0 0 10px var(--scene-primary, #00e5ff);
}
.hud-score { font-size: 14px; }
.hud-score b { font-size: 20px; }
.hud-timer {
  font-size: 26px;
  font-variant-numeric: tabular-nums;
  letter-spacing: 2px;
}

.hud-distract {
  position: absolute;
  top: 70px;
  left: 50%;
  transform: translateX(-50%);
  width: 220px;
  text-align: center;
}
.hud-bar {
  margin-top: 4px;
  height: 6px;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 3px;
  overflow: hidden;
}
.hud-bar__fill {
  height: 100%;
  background: var(--scene-alert, #ff2a55);
  transition: width 0.25s linear;
}

.hud-sound-toggle {
  position: absolute;
  right: 14px;
  bottom: 14px;
  pointer-events: auto;
  padding: 4px 10px;
  font-size: 12px;
  color: var(--scene-primary, #00e5ff);
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid var(--scene-primary, #00e5ff);
  border-radius: 3px;
  cursor: pointer;
}
.hud-sound-toggle--on {
  background: color-mix(in srgb, var(--scene-primary, #00e5ff) 20%, transparent);
}

.hud-barrage-layer {
  position: absolute;
  top: 30%;
  left: 0;
  width: 100%;
  height: 160px;
  overflow: hidden;
}
.hud-barrage {
  position: absolute;
  right: 0;
  white-space: nowrap;
  font-size: 20px;
  font-weight: 700;
  color: #fff;
  text-shadow: 0 0 6px var(--scene-alert, #ff2a55);
  animation: barrage-fly 6s linear forwards;
}
.hud-barrage:nth-child(even) { top: 40px; }
.hud-barrage:nth-child(3n) { top: 90px; }
@keyframes barrage-fly {
  from { transform: translateX(100%); }
  to { transform: translateX(-120%); }
}
</style>
