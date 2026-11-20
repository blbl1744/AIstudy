/**
 * 场景状态仓库：预设剧本列表、当前激活的 SceneConfig 与作品名编译。
 */
import { ref } from 'vue'
import { defineStore } from 'pinia'
import { sceneApi } from '@/api/http'
import type { SceneConfig } from '@/types'

export const useSceneStore = defineStore('scene', () => {
  // ===== 状态 =====
  const presets = ref<SceneConfig[]>([])
  const activeScene = ref<SceneConfig | null>(null)
  const compiling = ref(false)

  /** 加载预设列表（默认命中缓存，force=true 强制刷新） */
  async function loadPresets(force = false): Promise<SceneConfig[]> {
    if (presets.value.length > 0 && !force) {
      return presets.value
    }
    const res = await sceneApi.presets()
    presets.value = res.presets
    return presets.value
  }

  /** 直接激活一份场景配置 */
  function setActiveScene(scene: SceneConfig): void {
    activeScene.value = scene
  }

  /** 编译任意作品名并设为激活场景 */
  async function compileScene(workName: string): Promise<SceneConfig> {
    compiling.value = true
    try {
      const scene = await sceneApi.compile(workName)
      activeScene.value = scene
      return scene
    } finally {
      compiling.value = false
    }
  }

  /** 清空激活场景 */
  function resetScene(): void {
    activeScene.value = null
  }

  return {
    presets,
    activeScene,
    compiling,
    loadPresets,
    setActiveScene,
    compileScene,
    resetScene,
  }
})
