<script setup lang="ts">
/**
 * 专注度折线图组件：ECharts 绘制单局采样点曲线 + 60 分及格参考线。
 * 生命周期：onMounted 初始化并挂 resize 监听；watch 采样点变化重绘；
 *           onBeforeUnmount 移除监听并 dispose 释放实例，避免内存泄漏。
 * 无采样数据时展示友好空状态；全程文本插值，不使用 v-html。
 */
import { onBeforeUnmount, onMounted, ref, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import type { FocusSample } from '@/types'

const props = defineProps<{
  samples: FocusSample[]
}>()

const chartEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

/** 把 UTC ISO 时间截取为 HH:mm:ss 作为横轴标签（仅展示，不做时区换算） */
function axisLabel(iso: string): string {
  const parts = String(iso || '').split('T')
  const time = parts[1] ? parts[1].replace('Z', '').slice(0, 8) : parts[0]
  return time
}

/** 依据采样点渲染折线图：x 轴为采样时刻，y 轴为专注分（0~100），60 分为及格参考线 */
function render(): void {
  if (!chartEl.value) return
  if (!chart) chart = echarts.init(chartEl.value)
  const source = props.samples || []
  const labels = source.map((s) => axisLabel(s.created_at))
  const values = source.map((s) => s.focus_score)
  chart.setOption(
    {
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(13,31,51,0.92)', borderColor: '#00e5ff', textStyle: { color: '#e6f1ff' } },
      grid: { left: 44, right: 18, top: 24, bottom: 34 },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        axisLine: { lineStyle: { color: 'rgba(0,229,255,0.4)' } },
        axisLabel: { color: '#8fa8c0', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLine: { lineStyle: { color: 'rgba(0,229,255,0.4)' } },
        axisLabel: { color: '#8fa8c0' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
      },
      series: [
        {
          name: '专注分',
          type: 'line',
          smooth: true,
          showSymbol: values.length <= 40,
          lineStyle: { color: '#00e5ff', width: 2 },
          itemStyle: { color: '#00e5ff' },
          areaStyle: { color: 'rgba(0,229,255,0.14)' },
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: '#ffb547', type: 'dashed', width: 1 },
            label: { formatter: '及格线 60', color: '#ffb547', position: 'insideEndTop' },
            data: [{ yAxis: 60 }],
          },
        },
      ],
    },
    true,
  )
}

/** 窗口尺寸变化时重绘，保持图表自适应 */
function handleResize(): void {
  chart?.resize()
}

onMounted(() => {
  render()
  window.addEventListener('resize', handleResize)
})

// 采样点变化（切换战报）时按新数据重绘
watch(
  () => props.samples,
  () => nextTick(render),
  { deep: true },
)

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  if (chart) {
    chart.dispose()
    chart = null
  }
})
</script>

<template>
  <div class="focus-chart">
    <div v-if="!samples || samples.length === 0" class="chart-empty">
      本局暂无专注度采样数据
    </div>
    <div v-else ref="chartEl" class="chart-canvas"></div>
  </div>
</template>

<style scoped>
.focus-chart {
  width: 100%;
}
.chart-canvas {
  width: 100%;
  height: 260px;
}
.chart-empty {
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #7f93a8;
  font-size: 14px;
  border: 1px dashed rgba(0, 229, 255, 0.25);
  border-radius: 8px;
  background: rgba(13, 31, 51, 0.4);
}
</style>
