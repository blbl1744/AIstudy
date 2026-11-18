<script setup lang="ts">
/**
 * 督学公告广场 + 快速开房配置台（M4-A）。
 * 顶部为成长状态栏；主体左侧开房配置（模式 / 场景 / 目标 / 惩罚），右侧为实时公告广场。
 * 所有后端文本一律用 {{ }} 插值渲染，严禁 v-html；注释全部中文。
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { roomApi } from '@/api/http'
import { useUserStore } from '@/stores/user'
import { useSceneStore } from '@/stores/scene'
import { useRoomStore } from '@/stores/room'
import type { PlazaRoom, PunishmentType, RoomMode, SceneConfig } from '@/types'

const router = useRouter()
const userStore = useUserStore()
const sceneStore = useSceneStore()
const roomStore = useRoomStore()

// ===== 一、模式选择 =====
interface ModeOption { value: RoomMode; label: string; desc: string }
const MODE_OPTIONS: ModeOption[] = [
  { value: 'ai_solo', label: 'AI 剧本督学', desc: '单机沉浸式沉浸局' },
  { value: 'peer_1v1', label: '1v1 互督', desc: '公开房间一对一监督' },
  { value: 'peer_group', label: '多人围观', desc: '公开房间多人督学' },
]
const selectedMode = ref<RoomMode>('ai_solo')
const maxWatchers = ref(6)
const isPublic = ref(true)

function selectMode(mode: RoomMode): void {
  selectedMode.value = mode
  // 只有公开房间才会在公告广场展示，互督/多人模式默认公开
  isPublic.value = mode !== 'ai_solo'
}

// ===== 二、场景选择（预设 + 自定义作品编译） =====
const customWork = ref('')
const sceneError = ref('')

/** 加载内置预设，并在尚无激活场景时默认选中第一个 */
async function loadScenes(): Promise<void> {
  try {
    const presets = await sceneStore.loadPresets()
    if (!sceneStore.activeScene && presets.length > 0) {
      sceneStore.setActiveScene(presets[0])
    }
  } catch {
    sceneError.value = '场景预设加载失败，请稍后重试'
  }
}

/** 点击预设卡片选中该场景 */
function selectScene(scene: SceneConfig): void {
  sceneStore.setActiveScene(scene)
  sceneError.value = ''
}

/** 编译自定义作品名并设为激活场景（带加载态） */
async function compileCustom(): Promise<void> {
  const name = customWork.value.trim()
  if (!name) {
    sceneError.value = '请输入作品名称'
    return
  }
  sceneError.value = ''
  try {
    await sceneStore.compileScene(name)
  } catch {
    sceneError.value = '作品编译失败，请更换作品名重试'
  }
}

// ===== 三、学习目标（必填 1~100 字） =====
const goalText = ref('')
const GOAL_MAX = 100
const goalLength = computed(() => goalText.value.trim().length)
const goalValid = computed(() => goalLength.value >= 1 && goalLength.value <= GOAL_MAX)

// ===== 四、惩罚多选 =====
interface PunishmentOption { value: PunishmentType; label: string; desc: string }
const PUNISHMENT_OPTIONS: PunishmentOption[] = [
  { value: 'feynman_recall', label: '费曼输出', desc: '总结核心概念解锁' },
  { value: 'emergency_sprint', label: '战损冲刺', desc: '5 分钟急救冲刺' },
  { value: 'goal_breakdown', label: '军令状拆解', desc: '拆解微行动继续' },
]
const selectedPunishments = ref<PunishmentType[]>([])
const punishmentError = ref('')

function togglePunishment(type: PunishmentType): void {
  const i = selectedPunishments.value.indexOf(type)
  if (i >= 0) {
    selectedPunishments.value.splice(i, 1)
  } else {
    selectedPunishments.value.push(type)
  }
}

// ===== 五、开房 =====
const creating = ref(false)
const createError = ref('')
const canCreate = computed(() =>
  sceneStore.activeScene !== null && goalValid.value && !creating.value,
)

async function startRoom(): Promise<void> {
  if (!canCreate.value || !sceneStore.activeScene) return
  if (selectedPunishments.value.length === 0) {
    punishmentError.value = '请至少选择一种建设性惩罚'
    return
  }
  punishmentError.value = ''
  createError.value = ''
  creating.value = true
  const scene = sceneStore.activeScene
  try {
    const room = await roomStore.createRoom({
      mode: selectedMode.value,
      is_public: isPublic.value,
      scene_id: scene.scene_id,
      work_name: scene.source_work,
      scene_title: scene.scene_title,
      goal_text: goalText.value.trim(),
      max_watchers: selectedMode.value === 'peer_group' ? maxWatchers.value : 1,
      punishment_types: [...selectedPunishments.value],
    })
    // 同步激活场景到场景仓库，学习页直接消费
    sceneStore.setActiveScene(scene)
    await router.push({ name: 'study', params: { roomId: room.room_id } })
  } catch {
    createError.value = '开房失败，请检查配置后重试'
  } finally {
    creating.value = false
  }
}

// ===== 六、公告广场（手动刷新 + 10 秒轮询） =====
const rooms = ref<PlazaRoom[]>([])
const loading = ref(true)
const loadError = ref('')
let pollTimer: number | null = null

/** 房间模式中文 Label */
function modeLabel(mode: RoomMode): string {
  const map: Record<RoomMode, string> = {
    ai_solo: 'AI 剧本督学',
    peer_1v1: '1v1 互督',
    peer_group: '多人围观',
  }
  return map[mode] ?? mode
}

/**
 * 加载广场房间列表：后端已不过滤本人房间，前端同样保留，供房主暂离后返回。
 */
async function loadPlaza(): Promise<void> {
  try {
    const res = await roomApi.plaza()
    rooms.value = res.rooms
    loadError.value = ''
  } catch {
    loadError.value = '广场数据加载失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

/** 是否为本人房间（房主本人暂离后仍显示在广场，标注为可返回） */
function isMyRoom(room: PlazaRoom): boolean {
  return room.host_id === userStore.profile?.id
}

/**
 * 进入房间：本人房间跳学习室（继续/恢复本局），他人房间跳督学视角。
 */
function enterWatch(room: PlazaRoom): void {
  if (isMyRoom(room)) {
    router.push({ name: 'study', params: { roomId: room.room_id } })
  } else {
    router.push({ name: 'watch', params: { roomId: room.room_id } })
  }
}

/** 退出登录 */
function logout(): void {
  userStore.logout()
  router.push('/login')
}

onMounted(() => {
  void loadScenes()
  void loadPlaza()
  pollTimer = window.setInterval(() => { void loadPlaza() }, 10000)
})

onBeforeUnmount(() => {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
})
</script>

<template>
  <div class="plaza-page">
    <!-- 顶部成长状态栏 -->
    <header class="topbar">
      <div class="brand">督学公告广场 <span class="version">M4-A</span></div>
      <nav class="nav-actions">
        <span class="user-chip">{{ userStore.nickname || '学习者' }}</span>
        <span class="badge badge--rank">{{ userStore.rankTitle }}</span>
        <span class="badge badge--shield">护盾 × {{ userStore.shieldCount }}</span>
        <button type="button" class="nav-btn" @click="router.push('/profile')">成长档案</button>
        <button type="button" class="nav-btn" @click="logout">退出登录</button>
      </nav>
    </header>

    <main class="plaza-main">
      <!-- 左侧：开房配置台 -->
      <section class="config-panel">
        <h2 class="panel-title">快速开房</h2>

        <!-- 模式选择 -->
        <div class="field">
          <div class="field-label">督学模式</div>
          <div class="option-row">
            <button
              v-for="opt in MODE_OPTIONS"
              :key="opt.value"
              type="button"
              class="option-card"
              :class="{ 'option-card--active': selectedMode === opt.value }"
              @click="selectMode(opt.value)"
            >
              <div class="option-title">{{ opt.label }}</div>
              <div class="option-desc">{{ opt.desc }}</div>
            </button>
          </div>
          <div v-if="selectedMode === 'peer_group'" class="sub-field">
            <label>最大督学人数：
              <input v-model.number="maxWatchers" type="number" min="1" max="50" class="num-input" />
            </label>
          </div>
          <label class="check-line">
            <input v-model="isPublic" type="checkbox" :disabled="selectedMode === 'ai_solo'" />
            公开到督学广场
          </label>
        </div>

        <!-- 场景选择 -->
        <div class="field">
          <div class="field-label">战斗场景</div>
          <div class="scene-grid">
            <button
              v-for="scene in sceneStore.presets"
              :key="scene.scene_id"
              type="button"
              class="scene-card"
              :class="{ 'scene-card--active': sceneStore.activeScene?.scene_id === scene.scene_id }"
              @click="selectScene(scene)"
            >
              <div class="scene-title">{{ scene.scene_title }}</div>
              <div class="scene-source">原型：{{ scene.source_work }}</div>
              <div class="scene-tag">{{ scene.theme.hud_style }}</div>
            </button>
          </div>
          <div class="compile-row">
            <input
              v-model="customWork"
              type="text"
              class="text-input"
              maxlength="40"
              placeholder="输入电影/游戏/小说名移植为场景"
              @keyup.enter="compileCustom"
            />
            <button type="button" class="compile-btn" :disabled="sceneStore.compiling" @click="compileCustom">
              {{ sceneStore.compiling ? '编译中...' : '移植场景' }}
            </button>
          </div>
          <div v-if="sceneStore.activeScene" class="active-scene">
            当前场景：{{ sceneStore.activeScene.scene_title }} · {{ sceneStore.activeScene.commander_title }}
          </div>
          <p v-if="sceneError" class="error-hint">{{ sceneError }}</p>
        </div>

        <!-- 学习目标 -->
        <div class="field">
          <div class="field-label">本局学习目标 <span class="required">必填</span></div>
          <textarea
            v-model="goalText"
            class="goal-input"
            rows="2"
            :maxlength="GOAL_MAX"
            placeholder="例如：45 分钟复习高数极限，完成 3 道证明题"
          ></textarea>
          <div class="goal-count" :class="{ 'goal-count--bad': !goalValid && goalText.length > 0 }">
            {{ goalLength }} / {{ GOAL_MAX }} 字
          </div>
        </div>

        <!-- 惩罚多选 -->
        <div class="field">
          <div class="field-label">战败惩罚（可多选，至少 1 项）</div>
          <div class="option-row">
            <button
              v-for="opt in PUNISHMENT_OPTIONS"
              :key="opt.value"
              type="button"
              class="punish-card"
              :class="{ 'punish-card--active': selectedPunishments.includes(opt.value) }"
              @click="togglePunishment(opt.value)"
            >
              <div class="option-title">{{ opt.label }}</div>
              <div class="option-desc">{{ opt.desc }}</div>
            </button>
          </div>
          <p v-if="punishmentError" class="error-hint">{{ punishmentError }}</p>
        </div>

        <p v-if="createError" class="error-hint">{{ createError }}</p>
        <button type="button" class="start-btn" :disabled="!canCreate" @click="startRoom">
          {{ creating ? '正在开辟战场...' : '开始专注' }}
        </button>
      </section>

      <!-- 右侧：实时公告广场 -->
      <section class="plaza-panel">
        <div class="section-head">
          <h2 class="panel-title">活跃房间</h2>
          <button type="button" class="refresh-btn" @click="loadPlaza">刷新</button>
        </div>

        <p v-if="loading" class="hint">正在扫描广场信号...</p>
        <p v-else-if="loadError" class="error-hint">{{ loadError }}</p>
        <p v-else-if="rooms.length === 0" class="hint">当前暂无公开房间，成为第一个开房的人吧。</p>

        <div v-else class="room-grid">
          <article
            v-for="room in rooms"
            :key="room.room_id"
            class="room-card"
            :class="{ 'room-card--mine': isMyRoom(room) }"
          >
            <div class="room-mode">
              {{ isMyRoom(room) ? '我的进行中房间' : modeLabel(room.mode) }}
              <span v-if="isMyRoom(room) && !room.host_online" class="room-away">暂离中</span>
            </div>
            <h3 class="room-title">{{ room.scene_title }}</h3>
            <p class="room-goal">{{ room.goal_text }}</p>
            <div class="room-meta">
              <span>房主：{{ isMyRoom(room) ? '我' : room.host_nickname }}</span>
              <span>已坚持 {{ room.elapsed_seconds }} 秒</span>
              <span>在线督学 {{ room.watcher_count }}/{{ room.max_watchers }}</span>
            </div>
            <button type="button" class="enter-btn" @click="enterWatch(room)">
              {{ isMyRoom(room) ? '返回学习室' : '进入督学' }}
            </button>
          </article>
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.plaza-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #050b14 0%, #0d1f33 100%);
  color: #e6f1ff;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 28px;
  border-bottom: 1px solid rgba(0, 229, 255, 0.2);
  background: rgba(5, 11, 20, 0.7);
}
.brand {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 1px;
  color: #00e5ff;
}
.version {
  font-size: 11px;
  color: #8aa4c0;
  margin-left: 8px;
}
.nav-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.user-chip {
  font-size: 13px;
  color: #c7d8ea;
}
.badge {
  font-size: 12px;
  border-radius: 999px;
  padding: 2px 10px;
  border: 1px solid rgba(0, 229, 255, 0.4);
  color: #00e5ff;
}
.badge--shield {
  border-color: rgba(255, 214, 102, 0.5);
  color: #ffd666;
}
.nav-btn,
.refresh-btn,
.compile-btn {
  background: rgba(0, 229, 255, 0.08);
  border: 1px solid rgba(0, 229, 255, 0.35);
  color: #00e5ff;
  border-radius: 6px;
  padding: 6px 14px;
  cursor: pointer;
  font-size: 13px;
}
.nav-btn:hover,
.refresh-btn:hover,
.compile-btn:hover {
  background: rgba(0, 229, 255, 0.18);
}

.plaza-main {
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px 24px;
  display: grid;
  grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
  gap: 24px;
  align-items: start;
}
@media (max-width: 900px) {
  .plaza-main { grid-template-columns: 1fr; }
}

.config-panel,
.plaza-panel {
  background: rgba(13, 31, 51, 0.6);
  border: 1px solid rgba(0, 229, 255, 0.18);
  border-radius: 12px;
  padding: 20px;
}
.panel-title {
  margin: 0 0 18px;
  font-size: 18px;
  letter-spacing: 1px;
  color: #00e5ff;
}

.field {
  margin-bottom: 20px;
}
.field-label {
  font-size: 13px;
  color: #b9cde2;
  margin-bottom: 10px;
}
.required {
  color: #ff2a55;
  font-size: 11px;
  margin-left: 4px;
}
.option-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.option-card,
.punish-card {
  flex: 1 1 120px;
  text-align: left;
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(0, 229, 255, 0.2);
  border-radius: 8px;
  color: #e6f1ff;
  cursor: pointer;
}
.option-card--active,
.punish-card--active {
  border-color: #00e5ff;
  background: rgba(0, 229, 255, 0.14);
}
.option-title {
  font-size: 14px;
  font-weight: 600;
}
.option-desc {
  font-size: 11px;
  color: #8aa4c0;
  margin-top: 2px;
}
.sub-field,
.check-line {
  margin-top: 10px;
  font-size: 13px;
  color: #b9cde2;
}
.num-input {
  width: 64px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(0, 229, 255, 0.3);
  border-radius: 4px;
  color: #fff;
  padding: 4px 6px;
}

.scene-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px;
}
.scene-card {
  text-align: left;
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(0, 229, 255, 0.2);
  border-radius: 8px;
  color: #e6f1ff;
  cursor: pointer;
}
.scene-card--active {
  border-color: #00e5ff;
  background: rgba(0, 229, 255, 0.14);
}
.scene-title {
  font-size: 14px;
  font-weight: 600;
}
.scene-source {
  font-size: 11px;
  color: #8aa4c0;
  margin-top: 3px;
}
.scene-tag {
  display: inline-block;
  margin-top: 6px;
  font-size: 10px;
  color: #00e5ff;
  border: 1px solid rgba(0, 229, 255, 0.4);
  border-radius: 999px;
  padding: 0 8px;
}
.compile-row {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}
.text-input {
  flex: 1;
  min-width: 0;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(0, 229, 255, 0.3);
  border-radius: 6px;
  color: #fff;
  padding: 8px 10px;
  font-size: 13px;
}
.compile-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.active-scene {
  margin-top: 10px;
  font-size: 12px;
  color: #7fe9c0;
}

.goal-input {
  width: 100%;
  resize: vertical;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(0, 229, 255, 0.3);
  border-radius: 6px;
  color: #fff;
  padding: 8px 10px;
  font-size: 13px;
  line-height: 1.6;
  outline: none;
}
.goal-count {
  margin-top: 4px;
  font-size: 11px;
  color: #8aa4c0;
  text-align: right;
}
.goal-count--bad {
  color: #ff6b8a;
}

.start-btn {
  width: 100%;
  padding: 12px;
  font-size: 16px;
  font-weight: 700;
  color: #04121a;
  background: #00e5ff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}
.start-btn:disabled {
  background: #2a4a5a;
  color: #8aa4c0;
  cursor: not-allowed;
}

.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.hint,
.error-hint {
  color: #8aa4c0;
  font-size: 14px;
}
.error-hint {
  color: #ff6b8a;
  margin: 8px 0 0;
}
.room-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 14px;
}
.room-card {
  padding: 16px;
  background: rgba(5, 20, 34, 0.8);
  border: 1px solid rgba(0, 229, 255, 0.2);
  border-radius: 10px;
}
.room-card--mine {
  border-color: rgba(255, 214, 102, 0.6);
  background: rgba(40, 32, 10, 0.7);
}
.room-away {
  margin-left: 6px;
  font-size: 10px;
  color: #ffb020;
  border: 1px solid rgba(255, 176, 32, 0.5);
  border-radius: 999px;
  padding: 1px 8px;
}
.room-mode {
  display: inline-block;
  font-size: 11px;
  color: #00e5ff;
  border: 1px solid rgba(0, 229, 255, 0.4);
  border-radius: 999px;
  padding: 2px 10px;
}
.room-title {
  margin: 10px 0 6px;
  font-size: 16px;
}
.room-goal {
  margin: 0 0 12px;
  font-size: 13px;
  color: #8aa4c0;
  min-height: 36px;
}
.room-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #8aa4c0;
  margin-bottom: 14px;
}
.enter-btn {
  width: 100%;
  background: rgba(0, 229, 255, 0.08);
  border: 1px solid rgba(0, 229, 255, 0.35);
  color: #00e5ff;
  border-radius: 6px;
  padding: 8px;
  cursor: pointer;
  font-size: 13px;
}
.enter-btn:hover {
  background: rgba(0, 229, 255, 0.18);
}
</style>
