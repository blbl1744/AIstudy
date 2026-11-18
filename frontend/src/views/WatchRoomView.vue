<script setup lang="ts">
/**
 * 督学官监督界面（M3-A 占位）。
 * 仅展示路由房间 ID、哨兵提示与互动广播；实时画面与工具箱将在 M4 实现。
 */
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useRoomStore } from '@/stores/room'

const route = useRoute()
const roomStore = useRoomStore()

const currentRoomId = computed(() => {
  const value = route.params.roomId
  return Array.isArray(value) ? value[0] ?? '' : value ?? ''
})
</script>

<template>
  <div class="placeholder-page">
    <h1>督学官监督房间</h1>
    <p class="room-id">房间 ID：{{ currentRoomId }}</p>
    <ul class="state-list">
      <li>围观人数：{{ roomStore.watcherCount }}</li>
      <li>哨兵提示：{{ roomStore.sentinelMessage || '暂无' }}</li>
      <li>最近互动：{{ roomStore.lastInteraction?.text || roomStore.lastInteraction?.sound || '暂无' }}</li>
    </ul>
    <p class="note">该页面为基础占位，实时画面与互动工具箱将在 M4 阶段接入。</p>
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
