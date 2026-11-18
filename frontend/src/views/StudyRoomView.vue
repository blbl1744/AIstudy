<script setup lang="ts">
/**
 * 学习者沉浸式督学主界面（M3-A 占位）。
 * 仅展示路由房间 ID 与房间仓库关键状态，验证通信层与状态层接线；
 * 摄像头采集、HUD、指挥官对话与惩罚弹窗将在 M4 实现。
 */
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useRoomStore } from '@/stores/room'

const route = useRoute()
const roomStore = useRoomStore()

// 从路由参数安全取得房间 ID（可能是字符串或字符串数组）
const currentRoomId = computed(() => {
  const value = route.params.roomId
  return Array.isArray(value) ? value[0] ?? '' : value ?? ''
})
</script>

<template>
  <div class="placeholder-page">
    <h1>学习者督学房间</h1>
    <p class="room-id">房间 ID：{{ currentRoomId }}</p>
    <ul class="state-list">
      <li>剩余机会：{{ roomStore.strikesLeft }}</li>
      <li>专注评分：{{ roomStore.focusScore ?? '暂无' }}</li>
      <li>房间锁定：{{ roomStore.isLocked ? '是' : '否' }}</li>
      <li>字幕流：{{ roomStore.subtitleText || '暂无' }}</li>
    </ul>
    <p class="note">该页面为基础占位，完整交互将在 M4 阶段接入。</p>
  </div>
</template>

<style scoped>
.placeholder-page {
  min-height: 100vh;
  padding: 40px;
  background: linear-gradient(135deg, #050b14 0%, #0d1f33 100%);
  color: #e6f1ff;
}
.room-id { color: #00e5ff; }
.state-list { line-height: 2; color: #8aa4c0; }
.note { color: #ffb547; margin-top: 20px; }
</style>
