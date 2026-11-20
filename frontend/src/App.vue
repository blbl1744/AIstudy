<script setup lang="ts">
// 应用根组件：Naive UI 暗色主题 + 消息/对话框 Provider 包裹路由视图
import { NConfigProvider, NDialogProvider, NMessageProvider, darkTheme } from 'naive-ui'
</script>

<template>
  <n-config-provider :theme="darkTheme">
    <n-message-provider>
      <n-dialog-provider>
        <router-view />
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>
