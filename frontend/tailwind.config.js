/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        hud: {
          bg: '#050B14',
          panel: '#0D1F33',
          cyan: '#00E5FF',
          alert: '#FF2A55',
        },
      },
    },
  },
  // Naive UI 自带完整样式重置，关闭 Tailwind preflight 避免二者样式互相污染
  corePlugins: {
    preflight: false,
  },
  plugins: [],
}
