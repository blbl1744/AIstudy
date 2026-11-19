<script setup lang="ts">
/**
 * 建设性惩罚弹窗：监听 roomStore.isLocked，锁房时弹出不可关闭遮罩。
 * 按 activePunishment 切换三种界面：费曼输出 / 目标拆解 / 5 分钟战损冲刺。
 * 提交经 WebSocket 上行 punishment_submit；punishment_cleared 由 store 自动解锁，
 * punishment_rejected 在组件层订阅并展示退回原因（store 未订阅该消息）。
 * 全部后端文本以 {{ }} 插值渲染，严禁 v-html。
 */
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useRoomStore } from '@/stores/room'
import type { RoomSocket } from '@/api/roomSocket'
import type { PunishmentType } from '@/types'

const room = useRoomStore()

// ===== 一、惩罚类型与标题 =====
const PUNISHMENT_LABEL: Record<PunishmentType, string> = {
  feynman_recall: '费曼输出解锁',
  emergency_sprint: '5 分钟战损急救冲刺',
  goal_breakdown: '军令状目标拆解',
}
const currentType = computed<PunishmentType>(() => {
  const t = room.activePunishment
  return t === 'feynman_recall' || t === 'emergency_sprint' || t === 'goal_breakdown'
    ? t
    : 'feynman_recall'
})
const titleText = computed(() => PUNISHMENT_LABEL[currentType.value])

// ===== 二、费曼输出：不少于 15 字 =====
const feynmanText = ref('')
const FEYMAN_MIN = 15

// ===== 三、目标拆解：至少 2 条（换行或分号分隔） =====
const breakdownText = ref('')
const breakdownItems = computed(() =>
  breakdownText.value.split(/[\n;；]/).map((s) => s.trim()).filter((s) => s.length > 0),
)
const BREAKDOWN_MIN = 2

// ===== 四、战损冲刺：5 分钟倒计时 =====
const SPRINT_SECONDS = 5 * 60
const sprintRemain = ref(SPRINT_SECONDS)
let sprintTimer: number | null = null
const sprintClock = computed(() => {
  const m = Math.floor(sprintRemain.value / 60).toString().padStart(2, '0')
  const s = (sprintRemain.value % 60).toString().padStart(2, '0')
  return `${m}:${s}`
})

function startSprint(): void {
  stopSprint()
  sprintRemain.value = SPRINT_SECONDS
  sprintTimer = window.setInterval(() => {
    if (sprintRemain.value > 0) {
      sprintRemain.value -= 1
    } else {
      stopSprint()
    }
  }, 1000)
}
function stopSprint(): void {
  if (sprintTimer !== null) {
    window.clearInterval(sprintTimer)
    sprintTimer = null
  }
}

// ===== 五、提示语、退回原因与提交 =====
const promptText = ref('')
const rejectReason = ref('')
const formError = ref('')
const submitting = ref(false)

/** 锁房开启时初始化：清空表单、重置倒计时与提示 */
watch(
  () => room.isLocked,
  (locked) => {
    if (locked) {
      feynmanText.value = ''
      breakdownText.value = ''
      rejectReason.value = ''
      formError.value = ''
      submitting.value = false
      if (currentType.value === 'emergency_sprint') startSprint()
    } else {
      stopSprint()
    }
  },
  { immediate: true },
)

// 惩罚类型可能在锁房后变化，监听切换以重置对应状态
watch(currentType, () => {
  formError.value = ''
  rejectReason.value = ''
  if (currentType.value === 'emergency_sprint') {
    startSprint()
  } else {
    stopSprint()
  }
})

// socket 可能晚于组件挂载创建，监听其变化绑定退回原因与提示语订阅
let unsubscribers: Array<() => void> = []
watch(
  () => room.socket,
  (ws: RoomSocket | null) => {
    unsubscribers.forEach((u) => u())
    unsubscribers = []
    if (!ws) return
    unsubscribers.push(
      ws.on('punishment_required', (msg) => {
        promptText.value = msg.prompt
        rejectReason.value = ''
      }),
      ws.on('punishment_rejected', (msg) => {
        rejectReason.value = msg.reason
        submitting.value = false
      }),
      ws.on('punishment_cleared', () => {
        // 解锁成功：清空输入并停止计时，store 已将 isLocked 置 false，弹窗随之关闭
        feynmanText.value = ''
        breakdownText.value = ''
        rejectReason.value = ''
        formError.value = ''
        submitting.value = false
        stopSprint()
      }),
    )
  },
  { immediate: true },
)

/** 校验并提交当前惩罚的解锁内容 */
function submit(): void {
  if (submitting.value) return
  let content = ''
  if (currentType.value === 'feynman_recall') {
    content = feynmanText.value.trim()
    if (content.length < FEYMAN_MIN) {
      formError.value = `费曼总结需不少于 ${FEYMAN_MIN} 字，当前 ${content.length} 字`
      return
    }
  } else if (currentType.value === 'goal_breakdown') {
    if (breakdownItems.value.length < BREAKDOWN_MIN) {
      formError.value = `请至少拆解 ${BREAKDOWN_MIN} 条微行动（用换行或分号分隔）`
      return
    }
    content = breakdownItems.value.map((s, i) => `${i + 1}. ${s}`).join('\n')
  } else if (currentType.value === 'emergency_sprint') {
    content = '已完成 5 分钟战损冲刺，保持端正坐姿回到战场。'
  }

  formError.value = ''
  const ok = room.send({
    type: 'punishment_submit',
    punishment_type: currentType.value,
    content,
  })
  if (!ok) {
    formError.value = '连接未就绪，提交失败，请稍后重试'
    return
  }
  submitting.value = true
}

onBeforeUnmount(() => {
  stopSprint()
  unsubscribers.forEach((u) => u())
})
</script>

<template>
  <!-- 不可关闭遮罩：无关闭按钮，点击遮罩不触发任何关闭逻辑 -->
  <Teleport to="body">
    <div v-if="room.isLocked" class="punish-mask">
      <div class="punish-card">
        <div class="punish-badge">任务失守 · 建设性惩罚</div>
        <h2 class="punish-title">{{ titleText }}</h2>
        <p v-if="promptText" class="punish-prompt">{{ promptText }}</p>

        <!-- 费曼输出 -->
        <div v-if="currentType === 'feynman_recall'" class="punish-body">
          <p class="punish-tip">请用自己的话总结刚学到的 1 个核心概念，主动回忆可切断分心回路。</p>
          <textarea
            v-model="feynmanText"
            class="punish-textarea"
            rows="6"
            placeholder="在此输入你的费曼总结（不少于 15 字）..."
          ></textarea>
          <div class="punish-count">{{ feynmanText.trim().length }} / {{ FEYMAN_MIN }} 字</div>
        </div>

        <!-- 目标拆解 -->
        <div v-else-if="currentType === 'goal_breakdown'" class="punish-body">
          <p class="punish-tip">把大目标拆成眼前 10 分钟可完成的最小动作，重拾掌控感。</p>
          <textarea
            v-model="breakdownText"
            class="punish-textarea"
            rows="6"
            placeholder="每行或以分号分隔填写一条微行动，至少 2 条..."
          ></textarea>
          <div class="punish-count">已拆解 {{ breakdownItems.length }} / {{ BREAKDOWN_MIN }} 条</div>
        </div>

        <!-- 战损冲刺 -->
        <div v-else class="punish-body sprint-body">
          <p class="punish-tip">保持端正坐姿专注冲刺 5 分钟，即可挽回本局扣除的专注经验。</p>
          <div class="sprint-clock">{{ sprintClock }}</div>
          <p class="sprint-hint">倒计时结束后提交战报即可解锁</p>
        </div>

        <p v-if="formError" class="punish-error">{{ formError }}</p>
        <p v-if="rejectReason" class="punish-error">指挥官退回：{{ rejectReason }}</p>

        <button
          type="button"
          class="punish-submit"
          :disabled="submitting || (currentType === 'emergency_sprint' && sprintRemain > 0)"
          @click="submit"
        >{{ submitting ? '等待校验...' : (currentType === 'emergency_sprint' && sprintRemain > 0 ? '冲刺中...' : '提交解锁') }}</button>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.punish-mask {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(3, 6, 12, 0.92);
  backdrop-filter: grayscale(1);
}
.punish-card {
  width: min(520px, 92vw);
  padding: 28px;
  background: linear-gradient(160deg, #140a12 0%, #1a1420 100%);
  border: 1px solid #ff2a55;
  border-radius: 12px;
  box-shadow: 0 0 40px rgba(255, 42, 85, 0.35);
  color: #ffe9ef;
}
.punish-badge {
  display: inline-block;
  font-size: 12px;
  color: #ff2a55;
  border: 1px solid #ff2a55;
  border-radius: 999px;
  padding: 2px 12px;
}
.punish-title {
  margin: 14px 0 8px;
  font-size: 22px;
}
.punish-prompt {
  margin: 0 0 12px;
  color: #ff9ab0;
  font-size: 14px;
  line-height: 1.6;
}
.punish-body { margin-top: 10px; }
.punish-tip {
  font-size: 13px;
  line-height: 1.6;
  color: #c9a8b2;
  margin: 0 0 10px;
}
.punish-textarea {
  width: 100%;
  resize: vertical;
  padding: 10px 12px;
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid rgba(255, 42, 85, 0.5);
  border-radius: 6px;
  color: #fff;
  font-size: 14px;
  line-height: 1.6;
  outline: none;
}
.punish-textarea:focus {
  border-color: #ff2a55;
}
.punish-count {
  margin-top: 6px;
  font-size: 12px;
  color: #8f6f79;
  text-align: right;
}
.sprint-body { text-align: center; }
.sprint-clock {
  font-size: 64px;
  font-weight: 700;
  letter-spacing: 4px;
  font-variant-numeric: tabular-nums;
  color: #ff2a55;
  text-shadow: 0 0 20px rgba(255, 42, 85, 0.6);
  margin: 12px 0;
}
.sprint-hint {
  font-size: 13px;
  color: #c9a8b2;
}
.punish-error {
  margin: 10px 0 0;
  font-size: 13px;
  color: #ff8fa3;
}
.punish-submit {
  width: 100%;
  margin-top: 18px;
  padding: 12px;
  font-size: 15px;
  font-weight: 700;
  color: #140a12;
  background: #ff2a55;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
.punish-submit:disabled {
  background: #6b3242;
  color: #c9a8b2;
  cursor: not-allowed;
}
</style>
