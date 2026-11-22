# AIstudy 沉浸式游戏化双模督学平台

前后端分离的**沉浸式游戏化督学系统**：把传统单向课堂监控重构为「AI 剧本督学 + 公开互督广场」双模式，
配合实时语音指挥官、三次机会状态机、建设性惩罚与 ECharts 战报荣誉墙，帮助学生在低阻力、强反馈中建立学习自驱力。

- **后端**：FastAPI + MediaPipe + SQLite(WAL) + OpenAI 兼容 SDK，已继承安全加固(P0)/契约纠偏(P1)/并发优化(P2)/模块化拆分(P3)。
- **前端**：Vue 3 + TypeScript + Vite + Pinia + Naive UI + tsParticles + ECharts，数据驱动动态渲染，严禁 `v-html`。
- **运行形态**：前后端容器化一键编排（`docker compose up --build -d`），Nginx 静态托管前端并反代 `/api` 与 `/ws`；后端默认容器内 `8001` 端口。
- **系统定位**：面向学习场景的辅助工具，专注度数据仅供参考，不作为评价学习者的唯一依据。

---

## 一、产品定位与核心机制

取消旧的「教师/学生」硬性角色隔离，所有注册用户均为平等学习者，可随时切换身份：

- **学习者（房主）**：开启摄像头，选择 **AI 剧本督学** 或 **公开互督房间** 开启专注局。
- **督学官（监督者）**：浏览督学公告广场，进入他人公开房间观看实时状态，接收 AI 哨兵提示并发送弹幕与音效。

### 模式一：AI 沉浸式剧本督学
- 输入任意电影/游戏/小说名，后端 AI 剧本编译器经 Structured Output 编译为标准 `SceneConfig` 并缓存；前端动态注入主题色、粒子特效、环境音与角色音色。
- 实时对话指挥官全程在线：开局领命拆解目标、走神即时战术纠偏、卡壳按住说话求助；滑动窗口保留最近 8 轮交互，流式输出按句末标点切分配合 TTS 队列播放。
- 三次机会状态机 + 防误判缓冲（连续走神达阈值才判定有效失误）。建设性惩罚矩阵：费曼输出解锁 / 5 分钟战损冲刺 / 军令状目标拆解。
- 正向激励：无失误完成专注局累计专注时长与军衔；连续 3 局零失误奖励「自律护盾」。

### 模式二：公开互督与公告广场（方案 A）
- 房间状态写入 `TTLCache` 并实时展示在广场大厅；学习者按 1～2 fps 上传压缩 JPEG 帧，后端经房间 WebSocket 广播给督学官。
- 后端 MediaPipe 后台静默分析，连续走神触发 **AI 哨兵** 高优提示；督学官工具箱支持弹幕、音效与纪律警告。

---

## 二、技术栈与系统架构

```text
┌─────────────────────── Vue 3 + TypeScript 前端工程 ───────────────────────┐
│   督学公告广场    │    动态场景渲染引擎     │    专注战报与成长中心        │
│ (公开房间/筛选)  │ CSS变量/粒子/流式TTS   │  ECharts/军衔/护盾/CSV导出   │
└────────┬─────────┴──────────┬────────────┴─────────────┬──────────────────┘
         │ REST API (JWT)     │ WebSocket (/ws/room/...)  │
┌────────▼────────────────────▼──────────────────────────▼──────────────────┐
│                       FastAPI + Python 后端服务                           │
│  AI剧本与对话引擎  │   房间与广场网关      │  MediaPipe 推理池 / 战报接口   │
│ (LLM流式+记忆池)  │ (TTLCache+房间广播)  │ (线程隔离+背压丢弃+统计导出)  │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
                   SQLite (WAL 模式 + 复合索引)
```

| 层级 | 技术选型 | 职责 |
| :--- | :--- | :--- |
| 前端核心 | Vue 3 (`<script setup>`) + TypeScript + Vite | 强类型约束多态 WebSocket 消息与 `SceneConfig` 契约 |
| 状态与路由 | Pinia + Vue Router 4 | 管理 JWT 登录态、房间状态、剩余生命值、护盾与断线重连 |
| UI 与换肤 | Naive UI + Tailwind CSS + CSS Variables + tsParticles | 广场卡片/表单/弹窗、毫秒级全局换肤、粒子特效 |
| 战报可视化 | ECharts | 单局专注度时间轴曲线、60 分参考线、成长档案 |
| 后端服务 | FastAPI + MediaPipe + SQLite + OpenAI SDK | 线程安全推理池、TTL 缓存、剧本编译、流式对话、安全鉴权 |

---

## 三、核心数据契约

- `users`：`id`、`username`(唯一)、`nickname`、`password_hash`、`total_focus_minutes`、`rank_title`、`shield_count`、`streak_clean_sessions`、`created_at`。
- `study_sessions`：`id`、`user_id`、`mode`(`ai_solo`/`peer_1v1`/`peer_group`)、`scene_title`、`goal_text`、`duration_minutes`、`avg_score`、`strikes_used`、`punishment_type`、`status`(`completed`/`failed`)、`created_at`。
- `scene_cache`：`id`、`work_name`(规范化唯一索引)、`scene_json`(TEXT)、`created_at`。
- `behavior_records`：保留原字段与 `idx_behavior_student_time` 复合索引，新增 `session_id` 外键关联单局战报。
- 时间契约：UTC ISO-8601（`Z` 结尾），经 `to_iso_utc` 统一格式化。

### REST 接口清单

- **认证与成长**：`POST /api/auth/register`、`POST /api/auth/login`、`GET /api/users/me/profile`（用户信息、军衔、护盾与近期战报）、`DELETE /api/users/me/data`（清空个人学习数据）。
- **AI 剧本编译**：`GET /api/scenes/presets`、`POST /api/scenes/compile`（入参 `{"work_name": "..."}`，优先查缓存，未配 API Key 回退内置离线剧本）。
- **公告广场与房间**：`POST /api/rooms`、`GET /api/rooms/plaza`、`GET /api/rooms/{room_id}/snapshot.jpg`（受鉴权保护的实时画面快照）。
- **战报与导出**：`GET /api/stats/sessions/{session_id}`（单局专注度折线明细，含 404/403 校验）、`GET /api/stats/export.csv`（个人战报 CSV 导出，UTF-8 BOM + 公式注入转义）。
- **健康检查**：`GET /healthz`（容器探针，无需鉴权，只返回运行状态与计数）。
- **WebSocket**：`/ws/room/{room_id}?token=...`，客户端上行 `frame`/`user_message`/`punishment_submit`/`barrage`/`sound_effect`/`finish_session`，服务端下行 `analysis_update`/`ai_dialogue_*`/`punishment_*`/`sentinel_alert`/`room_interaction`。

---

## 四、环境变量配置表

所有配置项以 `AI_STUDY_` 为前缀，命名与根目录 `.env.example` **完全一致**。本地开发可复制 `.env.example` 为 `.env`；容器部署通过 `-e` / `--env-file` 注入，**严禁把 `.env` 打进镜像**。

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `AI_STUDY_JWT_SECRET` | 生产必填 | 自动生成随机值并告警 | JWT 签名密钥，长度须 ≥ 32 位；生产必须显式固定，否则重启/多副本令牌全部失效 |
| `AI_STUDY_CORS_ORIGINS` | 生产必填 | `http://localhost,http://127.0.0.1,http://localhost:5173,http://127.0.0.1:5173` | CORS 白名单（逗号分隔），同时用于 WebSocket `Origin` 校验；含 `*` 时自动关闭凭证携带 |
| `AI_STUDY_DB` | 否 | 项目根目录 `ai_study_buddy.db` | SQLite 数据库路径；容器内建议指向挂载卷 `/data/ai_study_buddy.db` |
| `AI_STUDY_LOG_LEVEL` | 否 | `INFO` | 日志级别，非法取值回落 `INFO` |
| `AI_STUDY_ENABLE_MODEL` | 否 | `0` | `1`=启用已训练模型，`0`=纯规则引擎；默认无模型亦可运行 |
| `AI_STUDY_MODEL_PATH` | 否 | `trained_model.pkl` | 模型文件路径（`AI_STUDY_ENABLE_MODEL=1` 时生效） |
| `AI_STUDY_MODEL_SHA256` | 否 | 空（跳过校验） | 模型完整性校验和，防止加载被替换的模型 |
| `AI_STUDY_LLM_BASE_URL` | 否 | `https://api.openai.com/v1` | 兼容 OpenAI 协议的模型接口根地址 |
| `AI_STUDY_LLM_API_KEY` | 否 | 空（离线降级） | 大模型 API Key；留空时剧本编译回退内置剧本、对话回退模板短句，不阻断学习 |
| `AI_STUDY_LLM_MODEL` | 否 | `gpt-4o-mini` | 剧本编译与流式对话使用的模型名 |
| `AI_STUDY_STRIKE_SECONDS` | 否 | `10` | 连续走神累计达该秒数判定为一次有效失误（扣 1 点生命值） |
| `AI_STUDY_TTS_ENABLED` | 否 | `0` | `1`=启用 edge-tts 语音合成（需外网），`0`=关闭；失败静默降级为纯文本 |
| `AI_STUDY_DIAGNOSTICS_ENABLED` | 否 | `0` | 姿态诊断日志：`1`=逐帧记录结构化字段用于真实摄像头验证，`0`=关闭（默认）；关闭时对评分/惩罚零副作用 |
| `AI_STUDY_DIAGNOSTICS_FORMAT` | 否 | `jsonl` | 诊断日志导出格式：`jsonl` 或 `csv` |
| `AI_STUDY_DIAGNOSTICS_DIR` | 否 | `logs/diagnostics` | 诊断日志目录；只写已计算字段，不写原始图像与敏感信息 |
| `AI_STUDY_AUTO_ALERT_SECONDS` | 否 | `15` | 持续分心自动预警阈值（秒） |
| `AI_STUDY_AUTO_ALERT_COOLDOWN` | 否 | `60` | 同一学生两次自动提醒的最小间隔（秒），防刷屏 |
| `AI_STUDY_DEBUG_API` | 否 | 空（不注册） | `1`=注册 `/api/debug/*` 诊断接口；**生产必须保持为空或 `0`** |

---

## 五、本地开发启动

```bash
# ---------- 后端 ----------
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
set AI_STUDY_JWT_SECRET=<不少于32位的随机串>
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
# 等价启动：python app.py  /  python -m app.main
curl http://127.0.0.1:8001/healthz

# ---------- 前端（另开终端）----------
cd frontend
npm install
npm run dev                       # Vite 开发服务 http://localhost:5173
```

前端 `vite.config.ts` 将 `/api`、`/ws` 代理到本地后端 `8001`。生产构建：`npm run build`（先 `vue-tsc --noEmit` 类型检查，再 `vite build` 产出 `dist/`）。

---

## 六、Docker Compose 一键启动（M5）

根目录 `docker-compose.yml` 编排 `backend`（纯 API 后端）与 `frontend`（多阶段构建：Node 构建 dist + Nginx 静态托管）；Nginx 通过 `try_files $uri $uri/ /index.html` 支持 Vue Router history 路由，并将 `/api/`、`/ws/` 反向代理至 `http://backend:8001`（含 WebSocket `Upgrade`/`Connection` 头透传）。对外暴露 **80** 端口，SQLite 数据落在命名卷 `ai_study_data`。

```bash
# 1. 准备环境变量（可选，未设置时 compose 使用占位默认值）
set AI_STUDY_JWT_SECRET=<不少于32位的随机串>

# 2. 一键构建并后台启动
docker compose up --build -d

# 3. 校验编排语法 / 查看状态 / 日志
docker compose config --quiet && echo COMPOSE_CONFIG_OK
docker compose ps
docker compose logs -f backend

# 4. 访问
#    前端页面  http://localhost
#    健康探针  http://localhost/api/... （经 Nginx 反代至后端 8001）

# 5. 停止并清理（保留数据卷）
docker compose down
```

> 生产部署务必通过环境变量注入强 `AI_STUDY_JWT_SECRET` 与真实 `AI_STUDY_CORS_ORIGINS`；摄像头 `getUserMedia` 仅在 HTTPS（安全上下文）可用，公网上线需在 Nginx 前置 TLS 终结。

---

## 七、自动化测试

```bash
python -m pytest tests -v
```

- `tests/conftest.py`：导入 `app` 前为缺失的 `cv2`/`mediapipe` 注入轻量桩模块，使测试在未装视觉库的环境也能运行；`client` 夹具用 `tmp_path` 隔离 `AI_STUDY_DB`、注入 JWT 密钥与 CORS 白名单，并在用例前后清理限速与内存缓存。
- `tests/test_core.py`：11 组核心守护用例——

| # | 用例 | 覆盖点 |
| :-- | :--- | :--- |
| 1 | `test_feature_contract` | 13 维特征契约 |
| 2 | `test_rule_score_range` | 规则评分区间 |
| 3 | `test_db_migration_idempotent` | 迁移幂等（`expire_at` 列与复合索引） |
| 4 | `test_health_auth_growth_cleanup` | 健康检查/注册登录/成长/清空数据 |
| 5 | `test_room_create_and_plaza` | 开房与公告广场 |
| 6 | `test_room_ws_auth_barrage_and_snapshot` | 房间 WS 鉴权/弹幕/快照 |
| 7 | `test_sentinel_alert_and_settlement` | AI 哨兵与结算 |
| 8 | `test_scene_presets_and_compile_cache` | 内置剧本与编译缓存 |
| 9 | `test_three_strikes_shield_and_punishment` | 三次机会/护盾/惩罚解锁 |
| 10 | `test_ws_ai_commander_dialogue` | 指挥官流式对话 |
| 11 | `test_stats_session_detail_and_csv_export` | 单局战报详情/越权 403/UTF-8 BOM CSV 导出 |

当前结果：**11 passed**（仅 `httpx` deprecated 警告，不影响通过）。

前端校验：

```bash
cd frontend && npm run build      # vue-tsc 零类型错误 + vite build 成功
```

安全红线自检：全量扫描 `frontend/src`，确认无真实 `v-html`（仅注释中声明禁止使用）。

---

## 八、目录结构

```text
AIstudy/
├── app/                       # 后端分层服务包
│   ├── config.py              # 配置层：阈值表/CORS/JWT/环境变量
│   ├── state.py               # 状态层：TTLCache、房间广播池、对话记忆、推理线程池
│   ├── db.py                  # 持久层：users/study_sessions/scene_cache/behavior_records 迁移与清理
│   ├── auth.py                # 认证层：bcrypt+JWT、登录限速、入参 XSS 白名单
│   ├── analyzer.py            # 分析层：MediaPipe 线程隔离、姿态/视线判定、规则打分
│   ├── scene_generator.py     # AI 剧本编译器 + 内置离线剧本 + 缓存
│   ├── ai_commander.py        # 实时流式对话指挥官 + 8 轮记忆 + TTS 分句
│   ├── punishment.py          # 三次机会状态机 + 护盾抵消 + 惩罚校验
│   ├── routers/               # auth / scene / room / stats 四组路由
│   └── main.py                # 装配层：FastAPI、lifespan、CORS、路由挂载
├── frontend/                  # Vue 3 + TS + Vite 前端工程
│   ├── src/
│   │   ├── api/               # http.ts(Axios 拦截器) / roomSocket.ts(WS 断线重连)
│   │   ├── stores/            # Pinia: user / room / scene
│   │   ├── types/             # 共享类型契约（含 SessionDetail）
│   │   ├── components/        # SceneRenderer / HudOverlay / CommanderChat / PunishmentModal / FocusChart
│   │   └── views/             # Login / Plaza / StudyRoom / WatchRoom / Profile
│   ├── nginx.conf             # 静态托管 + SPA try_files + /api /ws 反代
│   ├── Dockerfile             # 多阶段：Node 构建 dist + Nginx 托管
│   └── package.json
├── tests/                     # pytest 夹具与 11 组守护用例
├── docs/                      # 各阶段基准与报告文档（DEVELOPMENT_PLAN.md 等）
├── app.py                     # 兼容外壳（等价 uvicorn app.main:app）
├── Dockerfile                 # 纯 API 后端镜像（appuser 非 root 运行 + /data 卷）
├── docker-compose.yml         # backend + frontend 一键编排
├── requirements.txt           # 运行时与测试依赖
├── .dockerignore / .gitignore / .env.example
└── README.md
```

---

## 九、安全与隐私合规红线

- **防 XSS**：自定义目标、弹幕、AI 台词与对话流在入库/转发前做长度截断与控制字符清洗，前端一律用 Vue 文本插值 `{{ }}`，**严禁 `v-html`**。
- **摄像头知情同意与零落盘**：开启摄像头前须显式勾选授权；互督房间设有效期与人数上限；断开连接后立即销毁内存帧缓存，绝不落盘原始视频或人脸图像。
- **越权与注入防护**：接口按令牌身份双向隔离，`study_sessions` 详情校验归属（不存在 404、非本人 403）；姓名等回显字段字符白名单；CSV 文本列对 `=`/`+`/`-`/`@` 开头前置单引号防公式注入；登录失败限速防爆破。
- **一键删除本人数据**：`DELETE /api/users/me/data` 仅取 JWT 身份，删除其全部行为记录并清理实时缓存与在线状态。
- **密钥与优雅降级**：`AI_STUDY_JWT_SECRET`、`AI_STUDY_LLM_API_KEY` 全部经环境变量注入；未配 LLM Key 时平滑降级为内置剧本与模板指令，零外部依赖即可完整运行。
