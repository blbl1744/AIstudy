# 沉浸式游戏化双模督学平台 — 工程开发基准文档 (v1.0)

> **文档用途**：作为项目演进的唯一技术与产品基准。  
> **技术底座**：后端全面继承已完成安全加固（P0）、契约纠偏（P1）、并发优化（P2）与模块化拆分（P3）的 `FastAPI + MediaPipe + SQLite + TTLCache` 架构；前端剥离单页模板，升级为 `Vue 3 + TypeScript` 数据驱动动态渲染工程。

---

## 一、 产品定位与核心业务机制

本平台将传统“单向课堂监控”重构为**前后端分离的沉浸式游戏化督学系统**，通过**大模型场景移植 + 实时语音对话教练 + 社交化互督广场**，帮助学生在低阻力、强反馈的环境中建立学习自驱力。

### 1. 统一账号与双重身份体系
取消原有的“教师 / 学生”硬性角色隔离，所有注册用户均为平等的**普通学习者**，支持在任意时刻自由切换身份：
- **学习者（房主 / 被监督者）**：开启摄像头，选择 **AI 剧本督学** 或 **公开互督房间** 开启专注局。
- **督学官（监督者）**：浏览“督学公告广场”，一键进入他人公开房间观看实时学习状态、接收 AI 哨兵提示并发送互动弹幕与音效提醒。

### 2. 模式一：AI 沉浸式剧本督学

#### 2.1 全自动场景移植引擎
- 用户在开局时可选择内置精选场景，或输入**任意电影、游戏、小说名称**（如《流浪地球》《星际穿越》《亮剑》《霍格沃茨》《黑神话：悟空》）。
- 后端 **AI 剧本编译器**调用大模型 Structured Output，在 2 秒内将其编译为标准 `SceneConfig` 场景包并落库缓存；前端动态渲染引擎实时注入主题色、粒子特效、环境白噪音与角色音色，无需为单个作品硬编码页面。

#### 2.2 实时对话督学指挥官（主动带节奏 + 双向互动）
AI 督学官不再局限于播放固定警告音，而是作为**全程在线的实时任务教练**：
1. **开局领命与目标拆解**：用户输入本局目标（如“45 分钟复习高数极限”），AI 结合角色人设现场拆解为 2～3 个阶段微目标，并下达第一条具体行动指令。
2. **走神即时战术纠偏**：当视觉引擎判定连续走神时，系统将 `[当前目标 + 已坚持时长 + 走神类型 + 剩余机会]` 注入上下文，AI 现场生成带具体动作指引的纠正指令（例如：*“检测到视线离开战场！立刻把手机反扣在桌角，深呼吸两秒，回到第 3 题的推导上！”*）。
3. **卡壳随时求助（双向语音/文字对话）**：学习者遇到难题烦躁时，可通过按住说话（Push-to-Talk）或快捷输入框向指挥官求助，AI 现场拆解切入点或下达阶段调整指令，在刷手机走神发生前把注意力拉回学习。
4. **低延迟流式链路与记忆窗口**：后端维护房间级滑动窗口记忆 `dialogue_history`（保留最近 **8 轮**关键交互与状态事件）；LLM 流式输出按句末标点切分，前端字幕打字机与语音合成（`edge-tts` / 浏览器 Web Speech API）队列同步播放，将首字响应延迟控制在 1.5 秒内。

#### 2.3 防误判缓冲与三次机会状态机
- **防误判缓冲**：沿用单帧时间差截断累计算法。当连续低头或离座达到阈值（默认 `10s`）才判定为一次有效走神，避免正常低头记笔记误触；坐姿恢复端正后，当前走神累计进度清零。
- **递进式警告**：单局共 **3 次机会（生命值）**：
  - **第 1 次失误**：扣除 1 点生命值，触发指挥官口头警告与语音播报；
  - **第 2 次失误**：扣除第 2 点生命值，触发严厉训话与屏幕红框震动；
  - **第 3 次耗尽**：触发“任务失守”，锁定学习操作界面进入建设性惩罚结算。

#### 2.4 激发自驱力的建设性惩罚矩阵与正向激励
摒弃纯羞辱式惩罚，将“战败惩罚”转化为**主动学习的启动器（Trigger）**，开局支持多选组合：

| 惩罚模式 | 触发与解锁机制 | 自驱力心理学原理 |
| :--- | :--- | :--- |
| **1. 费曼输出解锁**（默认推荐） | 三次机会耗尽后锁定屏幕，指挥官要求汇报战果；学习者需语音或输入刚学到的 1 个核心概念 / 3 个单词，AI 校验非敷衍乱码后方可“戴罪立功”解锁。 | **主动回忆（Active Recall）**：强制提取记忆可瞬间切断刷手机的多巴胺回路，把大脑拉回深度思考。 |
| **2. 5 分钟战损急救冲刺** | 画面转为黑白战损滤镜，强制开启 5 分钟极简冲刺倒计时；期间保持端正坐姿不走神即可修复状态并挽回本局扣除的专注经验值。 | **微习惯破冰法则**：用极低时间门槛消除重新投入学习的心理阻力。 |
| **3. 军令状目标拆解** | 暂停总倒计时，弹出任务颗粒度拆解卡；引导学习者将大目标拆分为眼前 10 分钟可完成的最小动作，重新签字确认后继续。 | **破解畏难拖延**：多数走神源于任务过难产生的逃避心理，微拆解可恢复掌控感。 |

- **正向激励闭环**：无失误完成 25 / 45 分钟专注局可累计**专注时长与军衔/段位**；连续达成 3 局零失误奖励一枚**“自律护盾”**（下次耗尽三次机会时自动抵消一次战败结算）。

### 3. 模式二：双人 / 多人督学与公告广场（方案 A）
- **督学公告广场**：用户开启“公开互督（1v1）”或“多人围观监督（1vN）”房间时，房间状态写入 `TTLCache` 并实时展示在广场大厅（展示房主昵称、学习目标、场景标签、已坚持时长、在线督学人数）。
- **轻量帧转发 + AI 哨兵协同（零额外流媒体服务器）**：
  - 学习者端按 `1~2 fps` 上传压缩后的 `480px` JPEG 帧，后端通过房间 WebSocket 广播给在场所有督学官。
  - 后端 MediaPipe 同步在后台静默分析，一旦检测到房主连续走神，**AI 哨兵**立即向督学官端推送高优提示（*“⚠️ 哨兵提示：房主已连续低头 8 秒，请立即发指令督促！”*）。
  - **督学官工具箱**：支持发送全屏滚动大字弹幕、触发互动音效（敲黑板、清嗓子、防空警报）或发起纪律警告。

---

## 二、 系统总体架构与技术栈选型

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                       Vue 3 + TypeScript 前端工程                       │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│  │   督学公告广场   │  │  动态场景渲染引擎   │  │ 专注战报与成长中心 │  │
│  │ (公开房间/筛选)  │  │ CSS变量/粒子/流式TTS│  │ ECharts/军衔/护盾  │  │
│  └────────┬─────────┘  └──────────┬──────────┘  └─────────┬──────────┘  │
└───────────┼───────────────────────┼───────────────────────┼─────────────┘
            │ REST API (JWT)        │ WebSocket (/ws/room/{room_id})
┌───────────▼───────────────────────▼───────────────────────▼─────────────┐
│                       FastAPI + Python 后端服务                         │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│  │ AI剧本与对话引擎 │  │   房间与广场网关    │  │ MediaPipe 推理池   │  │
│  │ (LLM流式+记忆池) │  │ (TTLCache+房间广播) │  │ (线程隔离+背压丢弃)│  │
│  └──────────────────┘  └─────────────────────┘  └────────────────────┘  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
                      SQLite (WAL 模式 + 复合索引)
```

| 架构层级 | 技术选型 | 核心职责与选型理由 |
| :--- | :--- | :--- |
| **前端核心** | **Vue 3 (`<script setup>`) + TypeScript + Vite** | 强类型约束多态 WebSocket 消息与 `SceneConfig` 契约，避免联调字段错误。 |
| **状态与路由** | **Pinia + Vue Router 4** | 集中管理 JWT 登录态、当前房间状态、剩余生命值、护盾数与断线重连长连接。 |
| **动态换肤与 UI** | **Tailwind CSS + CSS Variables + Naive UI** | Naive UI 负责广场卡片、表单与弹窗；Tailwind + CSS 变量实现毫秒级全局主题换肤。 |
| **氛围与语音** | **`@tsparticles/vue3` + `Howler.js` + Web Speech / `edge-tts`** | 预置 5 种粒子特效与白噪音音轨；配合流式分句 TTS 实现指挥官实时语音对话。 |
| **战报可视化** | **ECharts** | 绘制单局专注度时间轴曲线、走神类型分布及周/月累计专注时长统计。 |
| **后端服务** | **FastAPI + MediaPipe + SQLite + OpenAI SDK** | 复用线程安全推理池、TTL 缓存与安全鉴权底座，扩展剧本编译与流式对话服务。 |

---

## 三、 核心数据契约与接口规范

### 1. AI 场景配置标准契约（`SceneConfig`）
无论是内置离线剧本还是大模型实时生成的电影/游戏场景，统一输出以下 JSON 结构供前端渲染器直接消费：

```json
{
  "scene_id": "wandering_earth_moss",
  "source_work": "流浪地球",
  "scene_title": "领航员空间站·中控室",
  "commander_title": "智能量子体 MOSS",
  "commander_persona": "冷静、理性、略带紧迫感的空间站人工智能指挥官，称呼用户为领航员。",
  "opening_line": "领航员，行星发动机已进入推进轨道。请汇报本航段学习目标，准备投入战斗。",
  "theme": {
    "primary_color": "#00E5FF",
    "alert_color": "#FF2A55",
    "bg_gradient": "linear-gradient(135deg, #050B14 0%, #0D1F33 100%)",
    "hud_style": "scifi"
  },
  "atmosphere": {
    "particle_preset": "stars",
    "ambient_sound": "spaceship_hum",
    "voice_id": "zh-CN-YunxiNeural"
  },
  "strikes": {
    "strike_1": "警告：检测到领航员注意力偏移，请立即回归学习岗位。",
    "strike_2": "严重警告：生命维持与专注指数持续下降，这是最后一次口头通告！",
    "strike_3": "任务失守：领航员已脱离战位，启动紧急战损汇报程序。"
  },
  "victory_line": "任务圆满完成，领航员的专注数据已载入空间站日志。"
}
```

- **渲染枚举白名单**（约束 LLM 输出，确保前端 100% 可渲染）：
  - `hud_style`: `"scifi"`（科幻框线） | `"military"`（战地迷彩） | `"academy"`（学院羊皮纸） | `"cyber"`（赛博霓虹）
  - `particle_preset`: `"stars"` | `"embers"` | `"rain"` | `"matrix"` | `"dust"`
  - `ambient_sound`: `"spaceship_hum"` | `"radar_tick"` | `"library_rain"` | `"fireplace"` | `"none"`

### 2. 数据库表结构演进（SQLite 增量迁移）
- **`users`（统一用户表）**：`id`, `username` (唯一), `nickname` (XSS 校验), `password_hash`, `total_focus_minutes` (默认 0), `rank_title` (默认 `"见习学员"`), `shield_count` (默认 0), `streak_clean_sessions` (默认 0), `created_at`。
- **`study_sessions`（督学战报记录表）**：`id`, `user_id`, `mode` (`ai_solo` / `peer_1v1` / `peer_group`), `scene_title`, `goal_text`, `duration_minutes`, `avg_score`, `strikes_used`, `punishment_type`, `status` (`completed` / `failed`), `created_at`。
- **`scene_cache`（AI 剧本缓存表）**：`id`, `work_name` (规范化唯一索引), `scene_json` (TEXT), `created_at`（相同作品名二次请求直接命中缓存）。
- **`behavior_records`（行为采样明细表）**：保留原有字段与 `idx_behavior_student_time` 复合索引，新增 `session_id` 外键关联单局战报。

### 3. 核心 REST API 与 WebSocket 协议

#### 3.1 REST 接口清单
- **认证与成长**：
  - `POST /api/auth/register`、`POST /api/auth/login`、`GET /api/users/me/profile`（返回用户信息、军衔、护盾数与近期战报）
- **AI 剧本编译**：
  - `GET /api/scenes/presets`（获取内置离线剧本列表）
  - `POST /api/scenes/compile`（入参 `{"work_name": "..."}`，优先查缓存，未命中则调用 LLM 生成并落库；未配 API Key 时平滑匹配内置离线剧本）
- **公告广场与房间**：
  - `POST /api/rooms`（创建房间：指定模式、场景 ID、学习目标、最大监督人数、惩罚类型组合）
  - `GET /api/rooms/plaza`（获取当前所有公开活跃房间列表）
  - `GET /api/rooms/{room_id}/snapshot.jpg`（受鉴权保护的房主实时画面快照接口）
- **战报与隐私**：
  - `GET /api/stats/sessions/{session_id}`（单局专注度折线明细）、`GET /api/stats/export.csv`（个人战报 CSV 导出）
  - `DELETE /api/users/me/data`（一键清空个人学习行为记录与在线缓存）

#### 3.2 统一房间 WebSocket 协议（`/ws/room/{room_id}?token=...`）
- **客户端上行消息**：
  - `{"type": "frame", "image": "data:image/jpeg;base64,..."}`（学习者推流帧）
  - `{"type": "user_message", "text": "长官，这题没思路", "voice_enabled": true}`（向 AI 指挥官对话求助）
  - `{"type": "punishment_submit", "punishment_type": "feynman_recall", "content": "..."}`（提交惩罚解锁内容）
  - `{"type": "barrage", "text": "加油别低头！"}` / `{"type": "sound_effect", "sound": "knock_board"}`（督学官互动）
  - `{"type": "finish_session"}`（正常结束本局并结算军衔/护盾）
- **服务端下行广播**：
  - `{"type": "analysis_update", "score": 88, "strikes_left": 2, "distracted_streak": 0}`（姿态分析结果）
  - `{"type": "ai_dialogue_start", "trigger": "proactive | distraction | chat"}` / `{"type": "ai_dialogue_chunk", "text": "..."}` / `{"type": "ai_audio_chunk", "audio_b64": "..."}` / `{"type": "ai_dialogue_end"}`（指挥官流式字幕与语音）
  - `{"type": "punishment_required", "punishment_type": "feynman_recall", "prompt": "..."}`（三次机会耗尽触发锁定）
  - `{"type": "punishment_cleared", "shield_used": false}`（解锁恢复学习）
  - `{"type": "sentinel_alert", "msg": "房主已连续走神 8 秒"}` / `{"type": "room_interaction", "from": "用户B", "text": "...", "sound": "..."}`（多人互督广播）

---

## 四、 前后端工程目录规划

```text
AIstudy/
├── backend/                     # 原有 app/ 演进目录
│   ├── app/
│   │   ├── config.py            # 新增 LLM API 与 TTS 配置（AI_STUDY_LLM_BASE_URL / API_KEY / MODEL）
│   │   ├── state.py             # 新增 active_rooms: TTLCache、room_connections 广播池与对话记忆队列
│   │   ├── db.py                # 统一 users、study_sessions、scene_cache 表结构与迁移
│   │   ├── auth.py              # 统一用户 JWT 鉴权与房间身份校验
│   │   ├── analyzer.py          # 保持线程隔离 MediaPipe 引擎不变
│   │   ├── scene_generator.py   # [新增] AI 剧本编译器 + 4 套离线精选剧本
│   │   ├── ai_commander.py      # [新增] 实时流式对话指挥官 + 8 轮记忆窗口 + edge-tts 分句合成
│   │   ├── punishment.py        # [新增] 三次机会状态机、自律护盾抵消与费曼总结校验器
│   │   └── routers/
│   │       ├── auth_routes.py
│   │       ├── scene_routes.py  # [新增] 剧本生成与预设列表
│   │       ├── room_routes.py   # [新增] 开房、公告广场、房间 WebSocket 网关
│   │       └── stats_routes.py  # [新增] 军衔成长、单局战报曲线与 CSV 导出
│   ├── tests/                   # pytest 核心守护用例
│   └── Dockerfile
└── frontend/                    # [新增] Vue 3 + TypeScript + Vite 独立前端工程
    ├── src/
    │   ├── api/                 # Axios 拦截器与房间 WebSocket 断线重连管理器
    │   ├── stores/              # Pinia: userStore, roomStore, sceneStore
    │   ├── components/
    │   │   ├── SceneRenderer.vue    # 动态场景外壳（注入 CSS 变量 + tsParticles + Howler）
    │   │   ├── HudOverlay.vue       # 沉浸式 HUD（3 次机会血条、雷达扫描框、流式字幕与弹幕层）
    │   │   ├── CommanderChat.vue    # 指挥官实时对话面板（按住说话 STT + 快捷输入 + 语音队列播放）
    │   │   ├── PunishmentModal.vue  # 建设性惩罚弹窗（费曼输出 / 5分钟战损冲刺 / 军令状拆解）
    │   │   └── FocusChart.vue       # ECharts 专注度波动曲线组件
    │   └── views/
    │       ├── LoginView.vue        # 登录 / 注册
    │       ├── PlazaView.vue        # 督学公告广场 + AI/互督开房配置台
    │       ├── StudyRoomView.vue    # 学习者沉浸式督学主界面
    │       ├── WatchRoomView.vue    # 督学官监督界面（实时画面 + AI 哨兵提示 + 互动工具箱）
    │       └── ProfileView.vue      # 军衔荣誉墙、自律护盾与历史战报
    ├── package.json
    └── vite.config.ts
```

---

## 五、 分阶段实施路线图（Milestones）

1. **M1：后端底座重构（统一用户 + 房间与公告广场网关）**
   - 清理旧的教师/学生双角色路由与 `templates/index.html`，将 `users` 表统一为普通学习者并增加军衔、护盾字段；实现 `active_rooms` 内存管理、`/api/rooms/plaza` 公告广场接口以及支持推帧、广播、弹幕与音效的 `/ws/room/{room_id}` 网关，补齐对应 `pytest` 用例。
2. **M2：AI 剧本编译器 + 实时流式对话指挥官 + 惩罚状态机**
   - 实现 `scene_generator.py`（内置离线剧本 + LLM 结构化编译与数据库缓存）、`ai_commander.py`（开局领命、走神纠偏、双向流式对话与 TTS 分句播报）以及 `punishment.py`（三次机会扣血、护盾抵消与三种建设性惩罚解锁闭环）。
3. **M3：Vue 3 前端工程初始化与动态场景渲染器**
   - 初始化 `frontend/`（Vue 3 + TS + Tailwind + Naive UI + tsParticles + Howler.js），实现登录鉴权、动态 CSS 变量换肤引擎、粒子与环境音调度器，以及字幕打字机 + 语音队列播放器。
4. **M4：双模式核心页面联调（广场 + AI 沉浸房 + 多人互督房）**
   - 完成**督学公告广场页**、**学习者主界面**（摄像头采集、HUD 生命值动画、指挥官对话面板、惩罚解锁弹窗）与**督学官监督界面**（实时画面渲染、AI 哨兵预警接收、快捷弹幕与音效发送）。
5. **M5：战报系统、容器编排与收官验收**
   - 接入 ECharts 战报曲线与 CSV 导出；编写前端 `Dockerfile`（Nginx 静态托管 + `/api` 与 `/ws` 反向代理）与根目录 `docker-compose.yml`，完成一键部署与端到端回归测试。

---

## 六、 工程安全与合规红线（继承自 P0–P3 基准）

- **输入与弹幕防 XSS**：自定义学习目标、督学官弹幕、AI 生成剧本台词与对话流在入库与转发前做长度截断与控制字符清洗，前端一律使用 Vue 默认文本插值 `{{ }}` 渲染，严禁使用 `v-html`。
- **摄像头知情同意与隐私零落盘**：开启摄像头前必须经过用户明确勾选授权；多人互督房间设有效期与人数上限，学习者结束学习或断开连接后，内存中的视频帧缓存立即销毁，绝不落盘存储任何原始视频或人脸图像。
- **密钥与优雅降级**：`AI_STUDY_JWT_SECRET`、`AI_STUDY_LLM_API_KEY` 等敏感配置全部通过环境变量注入；未配置 LLM API Key 时自动平滑降级为内置离线剧本与模板指令，保障零外部依赖即可完整运行。