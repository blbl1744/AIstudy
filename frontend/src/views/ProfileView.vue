<script setup lang="ts">
/**
 * 荣誉档案页（M5 收官）：展示军衔成长指标、历史战报列表与单局专注度曲线，
 * 并提供战报 CSV 导出与清空个人数据。所有内容使用 Vue 文本插值渲染，严禁 v-html。
 */
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'
import { statsApi } from '@/api/http'
import type { SessionDetail } from '@/types'
import FocusChart from '@/components/FocusChart.vue'

const router = useRouter()
const userStore = useUserStore()

const sessions = computed(() => userStore.profile?.sessions ?? [])
const detail = ref<SessionDetail | null>(null)
const activeSessionId = ref<number | null>(null)
const detailLoading = ref(false)
const exporting = ref(false)
const notice = ref('')

onMounted(() => {
  // 进入页面拉取最新档案，确保军衔/护盾/战报列表为当前值
  userStore.fetchProfile().catch(() => {
    // 401 由响应拦截器统一跳登录，这里只保留提示
    notice.value = '档案加载失败，请重新登录'
  })
})

/** 展开/收起某局战报：展开时按需拉取单局明细 */
async function toggleSession(sessionId: number): Promise<void> {
  if (activeSessionId.value === sessionId) {
    activeSessionId.value = null
    detail.value = null
    return
  }
  detailLoading.value = true
  notice.value = ''
  try {
    const data = await statsApi.sessionDetail(sessionId)
    detail.value = data
    activeSessionId.value = sessionId
  } catch (err) {
    notice.value = '战报明细加载失败'
    detail.value = null
    activeSessionId.value = null
  } finally {
    detailLoading.value = false
  }
}

/** 导出战报 CSV：Bearer Token 由 http 拦截器附加；转 Blob 下载后及时释放 URL */
async function exportCsv(): Promise<void> {
  exporting.value = true
  notice.value = ''
  try {
    const blob = await statsApi.exportCsv()
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'AIstudy-战报.csv'
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
    notice.value = '战报已导出'
  } catch (err) {
    notice.value = '战报导出失败'
  } finally {
    exporting.value = false
  }
}

/** 清空个人学习数据：二次确认后调用删除接口并刷新档案 */
async function clearData(): Promise<void> {
  if (!window.confirm('确定清空全部个人学习数据吗？此操作不可恢复。')) return
  try {
    await userStore.clearMyData()
    detail.value = null
    activeSessionId.value = null
    notice.value = '个人学习数据已清空'
  } catch (err) {
    notice.value = '清空数据失败'
  }
}

/** 惩罚类型中文映射 */
function punishmentLabel(value: string): string {
  const map: Record<string, string> = {
    feynman_recall: '费曼输出',
    emergency_sprint: '战损冲刺',
    goal_breakdown: '军令状拆解',
  }
  return map[value] || value || '无'
}

/** 模式中文映射 */
function modeLabel(value: string): string {
  const map: Record<string, string> = {
    ai_solo: 'AI 剧本督学',
    peer_1v1: '双人互督',
    peer_group: '多人围观',
  }
  return map[value] || value
}
</script>

<template>
  <div class="profile-page">
    <div class="page-head">
      <button type="button" class="back-btn" @click="router.push('/plaza')">← 返回广场</button>
      <h1>荣誉档案</h1>
    </div>

    <section class="heroes">
      <div class="hero-card rank">
        <span class="hero-label">当前军衔</span>
        <strong class="hero-value">{{ userStore.rankTitle }}</strong>
        <span class="hero-sub">{{ userStore.nickname }}</span>
      </div>
      <div class="hero-card">
        <span class="hero-label">累计专注时长</span>
        <strong class="hero-value">{{ userStore.totalFocusMinutes }}<small> 分钟</small></strong>
      </div>
      <div class="hero-card">
        <span class="hero-label">自律护盾</span>
        <strong class="hero-value">{{ userStore.shieldCount }}<small> 枚</small></strong>
      </div>
      <div class="hero-card">
        <span class="hero-label">连续零失误局数</span>
        <strong class="hero-value">{{ userStore.streakCleanSessions }}<small> 局</small></strong>
      </div>
    </section>

    <section class="toolbar">
      <button type="button" class="tool-btn" :disabled="exporting" @click="exportCsv">
        {{ exporting ? '导出中…' : '导出战报 CSV' }}
      </button>
      <button type="button" class="tool-btn danger" @click="clearData">清空个人学习数据</button>
    </section>

    <p v-if="notice" class="notice">{{ notice }}</p>

    <section class="history">
      <h2>历史战报</h2>
      <p v-if="sessions.length === 0" class="empty">还没有战报记录，完成一局专注后即可查看曲线。</p>
      <ul v-else class="session-list">
        <li v-for="session in sessions" :key="session.id" class="session-item">
          <button type="button" class="session-head" @click="toggleSession(session.id)">
            <span class="session-title">{{ session.scene_title }}</span>
            <span class="session-tags">
              <em class="tag">{{ modeLabel(session.mode) }}</em>
              <em class="tag" :class="session.status">{{ session.status === 'completed' ? '已完成' : '任务失守' }}</em>
            </span>
            <span class="session-meta">
              {{ session.duration_minutes }} 分钟 · 均分 {{ session.avg_score ?? '—' }} ·
              消耗机会 {{ session.strikes_used }} · {{ punishmentLabel(session.punishment_type) }}
            </span>
            <span class="session-time">{{ session.created_at }}</span>
            <span class="session-toggle">{{ activeSessionId === session.id ? '收起 ▲' : '曲线 ▼' }}</span>
          </button>
          <div v-if="activeSessionId === session.id" class="chart-panel">
            <p v-if="detailLoading" class="empty">曲线加载中…</p>
            <template v-else-if="detail && detail.summary.id === session.id">
              <FocusChart :samples="detail.points" />
              <p class="chart-goal">本局目标：{{ detail.summary.goal_text }}</p>
            </template>
          </div>
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.profile-page {
  min-height: 100vh;
  padding: 32px clamp(16px, 5vw, 64px) 64px;
  background: linear-gradient(135deg, #050b14 0%, #0d1f33 100%);
  color: #e6f1ff;
}
.page-head {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 24px;
}
.page-head h1 {
  margin: 0;
  font-size: 26px;
  letter-spacing: 2px;
}
.back-btn {
  background: transparent;
  border: 1px solid rgba(0, 229, 255, 0.4);
  color: #00e5ff;
  border-radius: 6px;
  padding: 6px 14px;
  cursor: pointer;
}
.back-btn:hover { background: rgba(0, 229, 255, 0.1); }
.heroes {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.hero-card {
  padding: 20px;
  background: rgba(13, 31, 51, 0.8);
  border: 1px solid rgba(0, 229, 255, 0.25);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.hero-card.rank { border-color: rgba(255, 181, 71, 0.5); }
.hero-label { color: #8fa8c0; font-size: 13px; }
.hero-value { font-size: 28px; color: #00e5ff; }
.hero-card.rank .hero-value { color: #ffb547; }
.hero-value small { font-size: 14px; color: #8fa8c0; }
.hero-sub { color: #7f93a8; font-size: 12px; }
.toolbar { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.tool-btn {
  padding: 9px 18px;
  border-radius: 6px;
  border: 1px solid rgba(0, 229, 255, 0.5);
  background: rgba(0, 229, 255, 0.08);
  color: #00e5ff;
  cursor: pointer;
}
.tool-btn:hover:not(:disabled) { background: rgba(0, 229, 255, 0.18); }
.tool-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.tool-btn.danger { border-color: rgba(255, 42, 85, 0.5); color: #ff5d7a; background: rgba(255, 42, 85, 0.08); }
.tool-btn.danger:hover { background: rgba(255, 42, 85, 0.18); }
.notice { color: #7fe0a0; font-size: 14px; }
.history h2 { font-size: 18px; margin: 24px 0 14px; letter-spacing: 1px; }
.empty { color: #7f93a8; padding: 24px 0; }
.session-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 12px; }
.session-item {
  background: rgba(13, 31, 51, 0.7);
  border: 1px solid rgba(0, 229, 255, 0.18);
  border-radius: 10px;
  overflow: hidden;
}
.session-head {
  width: 100%;
  text-align: left;
  background: transparent;
  border: 0;
  color: inherit;
  padding: 16px 18px;
  cursor: pointer;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 6px 12px;
}
.session-head:hover { background: rgba(0, 229, 255, 0.05); }
.session-title { font-size: 16px; font-weight: 600; }
.session-tags { display: flex; gap: 6px; justify-content: flex-end; }
.tag {
  font-style: normal;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: rgba(0, 229, 255, 0.12);
  color: #00e5ff;
}
.tag.failed { background: rgba(255, 42, 85, 0.15); color: #ff5d7a; }
.session-meta { grid-column: 1; color: #8fa8c0; font-size: 13px; }
.session-time { grid-column: 1; color: #5f7690; font-size: 11px; }
.session-toggle { grid-column: 2; grid-row: 2 / span 2; align-self: center; color: #00e5ff; font-size: 13px; }
.chart-panel { padding: 8px 18px 18px; border-top: 1px solid rgba(0, 229, 255, 0.12); }
.chart-goal { color: #8fa8c0; font-size: 13px; margin: 10px 0 0; }
</style>
