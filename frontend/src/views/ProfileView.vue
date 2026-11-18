<script setup lang="ts">
/**
 * 个人成长档案页（M3-A 占位）。
 * 展示军衔、护盾、专注分钟数与连续零失误局数；ECharts 战报曲线将在 M5 接入。
 */
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'

const router = useRouter()
const userStore = useUserStore()
</script>

<template>
  <div class="placeholder-page">
    <button type="button" class="back-btn" @click="router.push('/plaza')">← 返回广场</button>
    <h1>成长档案</h1>
    <div class="profile-card">
      <p>昵称：{{ userStore.nickname }}</p>
      <p>军衔：{{ userStore.rankTitle }}</p>
      <p>自律护盾：{{ userStore.shieldCount }} 枚</p>
      <p>累计专注：{{ userStore.totalFocusMinutes }} 分钟</p>
      <p>连续零失误局数：{{ userStore.streakCleanSessions }}</p>
    </div>
    <p class="note">战报曲线与 CSV 导出将在 M5 阶段接入。</p>
  </div>
</template>

<style scoped>
.placeholder-page {
  min-height: 100vh;
  padding: 40px;
  background: linear-gradient(135deg, #050b14 0%, #0d1f33 100%);
  color: #e6f1ff;
}
.back-btn {
  background: transparent;
  border: 1px solid rgba(0, 229, 255, 0.4);
  color: #00e5ff;
  border-radius: 6px;
  padding: 6px 14px;
  cursor: pointer;
  margin-bottom: 24px;
}
.profile-card {
  max-width: 420px;
  padding: 20px;
  background: rgba(13, 31, 51, 0.8);
  border: 1px solid rgba(0, 229, 255, 0.25);
  border-radius: 10px;
  line-height: 2;
}
.note { color: #ffb547; margin-top: 20px; }
</style>
