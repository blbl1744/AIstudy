<script setup lang="ts">
// 动态场景渲染引擎：CSS 变量换肤 + tsParticles 粒子特效 + Web Audio 程序化环境音
// 严格消费后端 SceneConfig 契约，零外部音源依赖（不加载任何 mp3，避免 404）
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { ISourceOptions } from '@tsparticles/engine'
import type { SceneConfig } from '../types'
import { useRoomStore } from '../stores/room'

// 互动音效开关与其他房间态共享 store，供左下控制条直接切换（不再用右下悬浮按钮）
const room = useRoomStore()

const props = defineProps<{
  scene: SceneConfig
}>()

// ===== 一、CSS 变量换肤：把主题字段实时注入根容器 =====
const rootStyle = computed(() => ({
  '--scene-primary': props.scene.theme.primary_color,
  '--scene-alert': props.scene.theme.alert_color,
  '--scene-bg': props.scene.theme.bg_gradient,
}))
// HUD 风格类：scifi / military / academy / cyber，驱动边框光效、角标与字体
const hudClass = computed(() => `scene-hud--${props.scene.theme.hud_style}`)

// ===== 二、粒子特效：按 preset 生成配置，可开关 =====
const particlesEnabled = ref(true)
const particleKey = computed(() => props.scene.atmosphere.particle_preset)

// 生成 5 种粒子预设配置（全部使用内置 shape，零外部图片资源）
function buildParticleOptions(preset: string): ISourceOptions {
  const base: Record<string, unknown> = {
    fullScreen: { enable: false },
    detectRetina: true,
    fpsLimit: 60,
    background: { color: 'transparent' },
    interactivity: {
      detectsOn: 'canvas',
      events: { resize: { enable: true } },
    },
  }

  let particles: Record<string, unknown>
  switch (preset) {
    // 星辰：缓慢漂浮、明暗闪烁的冷色星点
    case 'stars':
      particles = {
        number: { value: 90, density: { enable: true, width: 1920, height: 1080 } },
        color: { value: ['#ffffff', '#bfe9ff', '#8fd3ff'] },
        shape: { type: 'circle' },
        opacity: { value: 0.6, animation: { enable: true, speed: 0.6, minimumValue: 0.1 } },
        size: { value: { min: 1, max: 3 } },
        move: { enable: true, speed: 0.5, direction: 'none', outModes: { default: 'out' } },
      }
      break
    // 余烬：橙红色火星向上飘升并逐渐消散
    case 'embers':
      particles = {
        number: { value: 70, density: { enable: true, width: 1920, height: 1080 } },
        color: { value: ['#ff7a18', '#ffb703', '#ff3d00'] },
        shape: { type: 'circle' },
        opacity: { value: 0.8, animation: { enable: true, speed: 0.5, minimumValue: 0 } },
        size: { value: { min: 2, max: 5 }, animation: { enable: true, speed: 1, minimumValue: 0.5 } },
        move: { enable: true, speed: 1.6, direction: 'top', outModes: { top: 'out', default: 'none' } },
      }
      break
    // 雨丝：冷蓝色快速下坠的细密粒子
    case 'rain':
      particles = {
        number: { value: 160, density: { enable: true, width: 1920, height: 1080 } },
        color: { value: ['#4fc3f7', '#81d4fa', '#b3e5fc'] },
        shape: { type: 'circle' },
        opacity: { value: 0.35 },
        size: { value: { min: 1, max: 2 } },
        move: { enable: true, speed: 7, direction: 'bottom', outModes: { bottom: 'out', default: 'none' } },
      }
      break
    // 矩阵：绿色方块如代码雨般下落
    case 'matrix':
      particles = {
        number: { value: 100, density: { enable: true, width: 1920, height: 1080 } },
        color: { value: ['#00ff9c', '#00c875', '#00ff6a'] },
        shape: { type: 'square' },
        opacity: { value: 0.5, animation: { enable: true, speed: 0.8, minimumValue: 0.1 } },
        size: { value: { min: 2, max: 4 } },
        move: { enable: true, speed: 2.4, direction: 'bottom', outModes: { bottom: 'out', default: 'none' } },
      }
      break
    // 尘埃：暖灰色微粒缓慢无序漂浮
    case 'dust':
    default:
      particles = {
        number: { value: 80, density: { enable: true, width: 1920, height: 1080 } },
        color: { value: ['#cfc3a8', '#a8997a', '#8d7f63'] },
        shape: { type: 'circle' },
        opacity: { value: 0.4, animation: { enable: true, speed: 0.4, minimumValue: 0.1 } },
        size: { value: { min: 1, max: 4 } },
        move: { enable: true, speed: 0.35, direction: 'none', outModes: { default: 'out' } },
      }
      break
  }
  return { ...base, particles } as ISourceOptions
}
const particleOptions = computed<ISourceOptions>(() => buildParticleOptions(particleKey.value))

// ===== 三、Web Audio 程序化环境白噪音合成器 =====
// 所有声音均由振荡器 / 噪声缓冲实时合成，不引用任何外部音频文件
const ambientSound = computed(() => props.scene.atmosphere.ambient_sound)
const soundOn = ref(false) // 浏览器自动播放策略要求用户手势后才能出声，默认静音
const volume = ref(0.35)

let audioCtx: AudioContext | null = null
let masterGain: GainNode | null = null
let activeHandle: (() => void) | null = null
const unlockAudio = () => {
  if (!audioCtx) return
  if (audioCtx.state === 'suspended') void audioCtx.resume()
}

// 惰性创建 AudioContext（仅用户交互后调用，规避自动播放限制）
function ensureContext(): AudioContext | null {
  if (typeof window === 'undefined') return null
  if (!audioCtx) {
    const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!Ctor) return null
    audioCtx = new Ctor()
    masterGain = audioCtx.createGain()
    masterGain.gain.value = soundOn.value ? volume.value : 0
    masterGain.connect(audioCtx.destination)
  }
  unlockAudio()
  return audioCtx
}

// 生成 3 秒粉红噪声缓冲（Paul Kellet 近似算法），供雨声 / 壁炉复用
function createPinkNoiseBuffer(ctx: AudioContext): AudioBuffer {
  const length = ctx.sampleRate * 3
  const buffer = ctx.createBuffer(1, length, ctx.sampleRate)
  const data = buffer.getChannelData(0)
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0
  for (let i = 0; i < length; i++) {
    const white = Math.random() * 2 - 1
    b0 = 0.99886 * b0 + white * 0.0555179
    b1 = 0.99332 * b1 + white * 0.0750759
    b2 = 0.969 * b2 + white * 0.153852
    b3 = 0.8665 * b3 + white * 0.3104856
    b4 = 0.55 * b4 + white * 0.5329522
    b5 = -0.7616 * b5 - white * 0.016898
    data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11
    b6 = white * 0.115926
  }
  return buffer
}

// 安全停止一个音频节点
function stopNode(node: AudioScheduledSourceNode): void {
  try { node.stop() } catch { /* 已停止则忽略 */ }
}

// 飞船舱室低频嗡鸣：双振荡器低通滤波 + 缓慢呼吸调制
function buildSpaceshipHum(ctx: AudioContext, dest: GainNode): () => void {
  const gain = ctx.createGain()
  gain.gain.value = 0.2
  gain.connect(dest)
  const lowpass = ctx.createBiquadFilter()
  lowpass.type = 'lowpass'
  lowpass.frequency.value = 220
  lowpass.connect(gain)
  const osc1 = ctx.createOscillator()
  osc1.type = 'sawtooth'
  osc1.frequency.value = 52
  osc1.connect(lowpass)
  osc1.start()
  const osc2 = ctx.createOscillator()
  osc2.type = 'sine'
  osc2.frequency.value = 104
  const osc2Gain = ctx.createGain()
  osc2Gain.gain.value = 0.5
  osc2.connect(osc2Gain)
  osc2Gain.connect(lowpass)
  osc2.start()
  const lfo = ctx.createOscillator()
  lfo.frequency.value = 0.15
  const lfoGain = ctx.createGain()
  lfoGain.gain.value = 0.06
  lfo.connect(lfoGain)
  lfoGain.connect(gain.gain)
  lfo.start()
  return () => {
    stopNode(osc1); stopNode(osc2); stopNode(lfo)
    gain.disconnect(); lowpass.disconnect(); osc2Gain.disconnect(); lfoGain.disconnect()
  }
}

// 雷达滴答：周期性短促扫频脉冲
function buildRadarTick(ctx: AudioContext, dest: GainNode): () => void {
  const tick = () => {
    const t = ctx.currentTime
    const osc = ctx.createOscillator()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(1400, t)
    osc.frequency.exponentialRampToValueAtTime(680, t + 0.12)
    const g = ctx.createGain()
    g.gain.setValueAtTime(0.0001, t)
    g.gain.exponentialRampToValueAtTime(0.28, t + 0.01)
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.14)
    osc.connect(g)
    g.connect(dest)
    osc.start(t)
    osc.stop(t + 0.16)
  }
  tick()
  const timer = window.setInterval(tick, 2600)
  return () => window.clearInterval(timer)
}

// 图书馆雨声：粉红噪声经带通滤波后的沙沙底噪
function buildLibraryRain(ctx: AudioContext, dest: GainNode): () => void {
  const source = ctx.createBufferSource()
  source.buffer = createPinkNoiseBuffer(ctx)
  source.loop = true
  const highpass = ctx.createBiquadFilter()
  highpass.type = 'highpass'
  highpass.frequency.value = 140
  const lowpass = ctx.createBiquadFilter()
  lowpass.type = 'lowpass'
  lowpass.frequency.value = 1700
  const gain = ctx.createGain()
  gain.gain.value = 0.28
  source.connect(highpass)
  highpass.connect(lowpass)
  lowpass.connect(gain)
  gain.connect(dest)
  source.start()
  return () => {
    stopNode(source)
    source.disconnect(); highpass.disconnect(); lowpass.disconnect(); gain.disconnect()
  }
}

// 壁炉：低频暖噪声底 + 随机带通噼啪声
function buildFireplace(ctx: AudioContext, dest: GainNode): () => void {
  const buffer = createPinkNoiseBuffer(ctx)
  const source = ctx.createBufferSource()
  source.buffer = buffer
  source.loop = true
  const lowpass = ctx.createBiquadFilter()
  lowpass.type = 'lowpass'
  lowpass.frequency.value = 420
  const gain = ctx.createGain()
  gain.gain.value = 0.18
  source.connect(lowpass)
  lowpass.connect(gain)
  gain.connect(dest)
  source.start()
  const crackle = () => {
    if (Math.random() < 0.5) {
      const t = ctx.currentTime
      const burst = ctx.createBufferSource()
      burst.buffer = buffer
      const bandpass = ctx.createBiquadFilter()
      bandpass.type = 'bandpass'
      bandpass.frequency.value = 1200 + Math.random() * 1800
      bandpass.Q.value = 1.2
      const g = ctx.createGain()
      g.gain.setValueAtTime(0.0001, t)
      g.gain.exponentialRampToValueAtTime(0.32, t + 0.005)
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.07)
      burst.connect(bandpass)
      bandpass.connect(g)
      g.connect(dest)
      burst.start(t, Math.random() * 2)
      burst.stop(t + 0.09)
    }
  }
  const timer = window.setInterval(crackle, 220)
  return () => {
    stopNode(source)
    window.clearInterval(timer)
    source.disconnect(); lowpass.disconnect(); gain.disconnect()
  }
}

// 停止当前环境音
function stopAmbient(): void {
  if (activeHandle) {
    activeHandle()
    activeHandle = null
  }
}

// 按预设启动环境音
function startAmbient(): void {
  stopAmbient()
  if (!soundOn.value || ambientSound.value === 'none') return
  const ctx = ensureContext()
  if (!ctx || !masterGain) return
  switch (ambientSound.value) {
    case 'spaceship_hum': activeHandle = buildSpaceshipHum(ctx, masterGain); break
    case 'radar_tick': activeHandle = buildRadarTick(ctx, masterGain); break
    case 'library_rain': activeHandle = buildLibraryRain(ctx, masterGain); break
    case 'fireplace': activeHandle = buildFireplace(ctx, masterGain); break
    default: break
  }
}

// 场景切换时重建粒子（key 触发重挂载）与环境音
watch(ambientSound, () => { if (soundOn.value) startAmbient() })
// 静音开关：关闭立即静音并停播，开启需用户手势后重建
watch(soundOn, (on) => {
  if (on) startAmbient()
  else stopAmbient()
  if (masterGain && audioCtx) {
    masterGain.gain.setTargetAtTime(on ? volume.value : 0, audioCtx.currentTime, 0.05)
  }
})
// 音量滑块：平滑调节总增益
watch(volume, (v) => {
  if (masterGain && audioCtx && soundOn.value) {
    masterGain.gain.setTargetAtTime(v, audioCtx.currentTime, 0.05)
  }
})

onMounted(() => {
  // 首次用户点击/触摸时解锁音频上下文，之后开关环境音即可正常出声
  window.addEventListener('pointerdown', unlockAudio, { once: true })
})
onBeforeUnmount(() => {
  stopAmbient()
  window.removeEventListener('pointerdown', unlockAudio)
  if (audioCtx) void audioCtx.close()
  audioCtx = null
  masterGain = null
})
</script>

<template>
  <div class="scene-root" :class="hudClass" :style="rootStyle">
    <!-- 粒子层：跟随 particle_preset 切换，开关 controlled by particlesEnabled -->
    <VueParticles
      v-if="particlesEnabled"
      :key="particleKey"
      id="scene-particles"
      class="scene-particles"
      :options="particleOptions"
    />

    <!-- 四角装饰角标（HUD 风格通过 CSS 变量着色） -->
    <i class="scene-corner scene-corner--tl" aria-hidden="true"></i>
    <i class="scene-corner scene-corner--tr" aria-hidden="true"></i>
    <i class="scene-corner scene-corner--bl" aria-hidden="true"></i>
    <i class="scene-corner scene-corner--br" aria-hidden="true"></i>

    <!-- 场景标题条 -->
    <div class="scene-header">
      <span class="scene-header__dot"></span>
      <span class="scene-header__title">{{ scene.scene_title }}</span>
      <span class="scene-header__tag">{{ scene.source_work }}</span>
    </div>

    <!-- 控制面板：粒子开关 + 环境音静音开关 + 音量滑块 -->
    <div class="scene-controls">
      <button
        type="button"
        class="scene-btn"
        :class="{ 'scene-btn--active': particlesEnabled }"
        @click="particlesEnabled = !particlesEnabled"
      >
        粒子 {{ particlesEnabled ? '开' : '关' }}
      </button>
      <button
        type="button"
        class="scene-btn"
        :class="{ 'scene-btn--active': soundOn }"
        @click="soundOn = !soundOn"
      >
        {{ soundOn ? '环境音 开' : '环境音 关' }}
      </button>
      <button
        type="button"
        class="scene-btn"
        :class="{ 'scene-btn--active': room.interactionSoundOn }"
        @click="room.interactionSoundOn = !room.interactionSoundOn"
      >
        音效 {{ room.interactionSoundOn ? '开' : '关' }}
      </button>
      <label class="scene-volume">
        <span>音量</span>
        <input v-model.number="volume" type="range" min="0" max="1" step="0.01" />
      </label>
    </div>

    <!-- 主内容插槽：交由上层页面注入 HUD、视频与对话面板 -->
    <div class="scene-slot">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.scene-root {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 320px;
  overflow: hidden;
  background: var(--scene-bg);
  color: var(--scene-primary);
  border: 1px solid var(--scene-primary);
  box-sizing: border-box;
  transition: background 0.6s ease, border-color 0.4s ease, color 0.4s ease;
}

.scene-particles {
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
}

.scene-slot {
  position: relative;
  z-index: 2;
  width: 100%;
  height: 100%;
}

/* 顶部标题条 */
.scene-header {
  position: absolute;
  top: 14px;
  left: 20px;
  z-index: 3;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  letter-spacing: 1px;
}
.scene-header__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--scene-primary);
  box-shadow: 0 0 10px var(--scene-primary);
}
.scene-header__title {
  font-weight: 700;
}
.scene-header__tag {
  opacity: 0.6;
  font-size: 12px;
}

/* 控制面板 */
.scene-controls {
  position: absolute;
  bottom: 14px;
  left: 20px;
  z-index: 3;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.scene-btn {
  padding: 4px 12px;
  font-size: 12px;
  color: var(--scene-primary);
  background: transparent;
  border: 1px solid var(--scene-primary);
  border-radius: 2px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.scene-btn:hover { background: color-mix(in srgb, var(--scene-primary) 18%, transparent); }
.scene-btn--active {
  background: color-mix(in srgb, var(--scene-primary) 24%, transparent);
  box-shadow: 0 0 8px color-mix(in srgb, var(--scene-primary) 50%, transparent);
}
.scene-volume {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}
.scene-volume input[type='range'] {
  width: 96px;
  accent-color: var(--scene-primary);
  cursor: pointer;
}

/* 四角装饰角标 */
.scene-corner {
  position: absolute;
  z-index: 3;
  width: 22px;
  height: 22px;
  border-color: var(--scene-primary);
  border-style: solid;
  pointer-events: none;
}
.scene-corner--tl { top: 6px; left: 6px; border-width: 2px 0 0 2px; }
.scene-corner--tr { top: 6px; right: 6px; border-width: 2px 2px 0 0; }
.scene-corner--bl { bottom: 6px; left: 6px; border-width: 0 0 2px 2px; }
.scene-corner--br { bottom: 6px; right: 6px; border-width: 0 2px 2px 0; }

/* ===== HUD 风格：科幻（科幻框线 + 等宽字体 + 青色辉光） ===== */
.scene-hud--scifi {
  font-family: 'Consolas', 'Monaco', monospace;
  box-shadow: inset 0 0 60px color-mix(in srgb, var(--scene-primary) 12%, transparent);
}
.scene-hud--scifi .scene-header { text-transform: uppercase; }

/* ===== HUD 风格：战地迷彩（粗犷无衬线 + 虚线边框 + 警示色角标） ===== */
.scene-hud--military {
  font-family: 'Impact', 'Arial Narrow', sans-serif;
  border-style: dashed;
  border-width: 3px;
}
.scene-hud--military .scene-corner { border-color: var(--scene-alert); width: 30px; height: 30px; }

/* ===== HUD 风格：学院羊皮纸（衬线字体 + 双线边框 + 暖色标题） ===== */
.scene-hud--academy {
  font-family: 'Georgia', 'Times New Roman', serif;
  border-width: 4px;
  outline: 1px solid var(--scene-primary);
  outline-offset: -9px;
}
.scene-hud--academy .scene-header__title { color: var(--scene-primary); }

/* ===== HUD 风格：赛博霓虹（霓虹辉光 + 品红/青撞色 + 强闪烁感） ===== */
.scene-hud--cyber {
  font-family: 'Courier New', monospace;
  box-shadow: inset 0 0 90px color-mix(in srgb, var(--scene-alert) 22%, transparent),
              0 0 18px color-mix(in srgb, var(--scene-primary) 40%, transparent);
}
.scene-hud--cyber .scene-corner { border-color: var(--scene-alert); box-shadow: 0 0 12px var(--scene-alert); }
.scene-hud--cyber .scene-header__dot { background: var(--scene-alert); box-shadow: 0 0 12px var(--scene-alert); }
</style>
