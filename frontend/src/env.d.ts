/// <reference types="vite/client" />

// 声明 .vue 单文件组件模块，供 TS 识别 SFC 导入
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<Record<string, unknown>, Record<string, unknown>, unknown>
  export default component
}
