# AI 学习专注度分析系统（ai-study-buddy）

基于摄像头画面的**课堂/自习专注度分析系统**：学生端浏览器采集画面帧，服务端用 MediaPipe 提取
33 点姿态关键点，结合几何规则引擎（可选叠加已训练的回归模型）输出专注度分数、姿态状态、
视线状态与疲劳等级；教师端可查看学生列表、历史行为记录与实时画面，并对低专注学生下发提醒；
连续分心超阈值时由系统自动下发温和提醒，并在教师端高亮该学生卡片。

- 技术栈：FastAPI + Uvicorn + SQLite（WAL）+ MediaPipe/OpenCV + NumPy/scikit-learn + JWT(bcrypt)
- 运行形态：单进程 ASGI 服务（默认端口 `8001`），支持 Docker 部署，前置 Nginx 做 HTTPS/WSS 终结
- 系统定位：**未成年人场景下的辅助工具**，仅提供行为趋势参考，不作为评价学生的唯一依据

---

## 一、模块分层架构（P3-1 重构后）

重构把原本 2100+ 行的单文件服务拆成「单向依赖、无循环」的分层包 `app/`，依赖方向只能自上而下：

```
app/
├── config.py        配置层：默认参数、阈值表、CORS/JWT 加固、日志初始化、环境变量读取
├── state.py         状态层：TTLCache 三级内存缓存、在线学生表、待推送提醒、推理线程池、缓存 janitor
├── db.py            持久层：SQLite 连接与幂等建表迁移、索引、30 天留存清理、学生/教师/行为记录读写
├── auth.py          认证层：bcrypt 哈希（兼容历史 SHA256 并自动升级）、JWT 签发校验、登录限速、入参 XSS 白名单
├── model_loader.py  模型层：13 维特征契约（FEATURE_*）、模型 SHA256 校验加载、契约自检、推理封装
├── analyzer.py      分析层：MediaPipe 线程隔离懒加载、姿态/疲劳/视线判定、规则打分、时序平滑、13 维特征抽取
├── routers/         路由层（仅做接线与鉴权，不含算法）
│   ├── auth_routes.py      /api/student/register|login、/api/teacher/register|login、/api/me
│   ├── student_routes.py   /api/student/behavior、/ws 实时推理、/api/student/notifications、
│   │                       DELETE /api/student/my-data、学生端语音识别封装（火山引擎）
│   └── teacher_routes.py   /api/teacher/students[/{id}/records|/live|/snapshot.jpg]、
│                           POST /api/teacher/students/{id}/notify（实时督学提醒）、条件注册的 /api/debug/*
└── main.py          装配层：FastAPI 实例、lifespan（建表 + 模型加载 + 缓存 janitor）、CORS、/healthz
```

依赖方向（禁止反向 import）：

```
main ──► routers ──► analyzer ──► model_loader ──► state
  │          │            │             │            │
  └──────────┴────────────┴─────────────┴──► auth ──► db ──► config
```

要点：

- `config.py` 被所有层依赖，自身不依赖任何业务模块；`db.py` 对 `auth.hash_password` 采用**函数内延迟导入**，因此不构成循环依赖；
- 原有单文件入口 `app.py`（25 行）保留为兼容外壳，`uvicorn app:app`、`python app.py`、`python -m app.main` 三种启动方式等价可用；
- `GET /healthz` 为容器探针专用，无需鉴权，只返回运行状态与计数（不暴露路径、密钥、堆栈）。

---

## 二、环境变量配置表

所有配置项均以 `AI_STUDY_` 为前缀，命名与根目录 [`.env.example`](.env.example) **完全一致**（共 11 项）。
本地开发可复制 `.env.example` 为 `.env`；容器部署通过 `-e` / `--env-file` 注入，**严禁把 `.env` 打进镜像**。

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `AI_STUDY_JWT_SECRET` | 生产必填 | 自动生成随机值并告警 | JWT 签名密钥，长度须 **≥ 32 位**。未设置或过弱时启动会随机生成，重启/多副本间不一致将导致令牌全部失效，生产必须显式固定 |
| `AI_STUDY_CORS_ORIGINS` | 生产必填 | `http://localhost:8001,http://127.0.0.1:8001,http://localhost:5173,http://127.0.0.1:5173` | CORS 白名单，英文逗号分隔（M1-A 起默认放行 Vite 前端 5173 端口）。含 `*` 通配符时自动关闭凭证携带（`allow_credentials=False`）；同时用于 WebSocket `Origin` 白名单校验 |
| `AI_STUDY_ADMIN_PASSWORD` | 否 | 空（不创建） | 旧教师端管理员初始密码；M1-A 起已剥离双角色，不再初始化教师管理员账号 |
| `AI_STUDY_ENABLE_MODEL` | 否 | `0` | `1`=启用已训练模型，`0`=纯规则引擎。默认纯规则模式，无模型文件亦可运行 |
| `AI_STUDY_MODEL_PATH` | 否 | `trained_model.pkl` | 模型文件路径（`AI_STUDY_ENABLE_MODEL=1` 时生效） |
| `AI_STUDY_MODEL_SHA256` | 否 | 空（跳过校验） | 模型文件完整性校验和，防止加载被替换的模型文件 |
| `AI_STUDY_DEBUG_API` | 否 | 空（不注册） | `1`=注册 `/api/debug/*` 诊断接口（仅教师可用）；**生产必须保持为空或 `0`** |
| `AI_STUDY_DB` | 否 | 项目根目录 `ai_study_buddy.db` | SQLite 数据库文件路径。容器部署建议指向挂载卷，如 `/data/ai_study_buddy.db` |
| `AI_STUDY_LOG_LEVEL` | 否 | `INFO` | 日志级别：`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`；非法取值自动回落 `INFO` |
| `AI_STUDY_AUTO_ALERT_SECONDS` | 否 | `15` | 持续分心自动预警阈值：连续分心（未检测到人体或专注度 < 60）累计达到该秒数即触发一次温和提醒（`feat/extensions` 扩展二） |
| `AI_STUDY_AUTO_ALERT_COOLDOWN` | 否 | `60` | 自动提醒冷却时间：同一学生两次自动提醒的最小间隔秒数，冷却期内持续分心也不会重复下发，防止刷屏 |

> 说明：`AI_STUDY_DB` 与 `AI_STUDY_LOG_LEVEL` 在 P3-4 收尾时补齐为真实生效的配置项
> （此前 README 约定但代码未读取，属「文档承诺未落地」缺口），便于容器挂卷与统一调级，行为默认值与原实现完全一致。

---

## 三、本地启动

```bash
# 1. 安装依赖（建议 3.11+ / 3.12，示例为 Windows；Linux/macOS 使用对应激活命令）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置环境变量（最小可用集）
set AI_STUDY_JWT_SECRET=<不少于32位的随机串>
set AI_STUDY_CORS_ORIGINS=http://localhost:8001,http://127.0.0.1:8001,http://localhost:5173,http://127.0.0.1:5173

# 3. 启动（三种方式等价，任选其一）
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
python app.py
python -m app.main

# 4. 自检
curl http://127.0.0.1:8001/healthz
```

M1-A 起后端仅提供 API（健康检查：`http://127.0.0.1:8001/healthz`）；前端独立运行于 Vite 开发服务 `http://localhost:5173`。

---

## 四、Docker 构建与运行

```bash
# 构建
docker build -t ai-study-buddy:latest .

# 运行（数据落在命名卷上，容器重建不丢库；密钥走运行时注入）
docker run -d --name ai-study \
  -p 8001:8001 \
  -e AI_STUDY_JWT_SECRET="change-me-at-least-32-characters-long" \
  -e AI_STUDY_CORS_ORIGINS="https://study.example.com" \
  -e AI_STUDY_DB=/data/ai_study_buddy.db \
  -e AI_STUDY_LOG_LEVEL=INFO \
  -v ai_study_data:/data \
  ai-study-buddy:latest

# 查看日志与健康状态
docker logs -f ai-study
curl -s http://127.0.0.1:8001/healthz
```

镜像要点：

- 基础镜像 `python:3.11-slim`，额外安装 `libgl1`、`libglib2.0-0`（OpenCV/MediaPipe 运行时必需，缺失会报 `libGL.so.1`）；
- 容器内以非 root 用户 `appuser` 运行，日志目录 `/app/logs` 与数据目录 `/data` 已预建并授权；
- 镜像内**不含** `tests/`、`*.db*`、`logs/`、`.env`（见 `.dockerignore`）；
- 启动命令固定为 `uvicorn app.main:app --host 0.0.0.0 --port 8001`，对外由 Nginx 终结 TLS。

---

## 五、生产环境 Nginx 反向代理（HTTPS + WebSocket）

`/ws` 为实时推理通道，必须开启 `Upgrade`/`Connection` 透传，并把读超时放大，避免长连接被代理切断：

```nginx
upstream ai_study_upstream {
    server 127.0.0.1:8001;
    keepalive 32;
}

# HTTP 强制跳转 HTTPS
server {
    listen 80;
    server_name study.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name study.example.com;

    ssl_certificate     /etc/nginx/certs/study.example.com.pem;
    ssl_certificate_key /etc/nginx/certs/study.example.com.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_session_cache   shared:SSL:10m;

    # 安全响应头（视频采集仅本站使用，摄像头权限限定 self）
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Permissions-Policy "camera=(self), microphone=()" always;

    # 请求体上限：实时帧为 base64 图片，需放宽到 10M
    client_max_body_size 10m;

    # ---- WebSocket：wss://study.example.com/ws ----
    location /ws {
        proxy_pass http://ai_study_upstream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout  3600s;
        proxy_send_timeout  3600s;
    }

    # ---- HTTP 接口与页面 ----
    location / {
        proxy_pass http://ai_study_upstream;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }
}
```

配套要求：

- `AI_STUDY_CORS_ORIGINS` 必须写入真实前端来源（如 `https://study.example.com`），否则浏览器跨域与 WebSocket `Origin` 校验会被拒（WebSocket 关闭码 `4403`）；
- 前端页面须在 HTTPS 下访问，否则浏览器禁止调用摄像头（`getUserMedia` 仅在安全上下文可用）；
- 若有多副本，`AI_STUDY_JWT_SECRET` 必须一致，数据库需使用共享存储（SQLite 单写特性下建议单副本或改用外部数据库）。

---

## 六、未成年人数据采集与隐私合规

本系统采集的是**未成年人**的人脸/姿态画面帧与专注度记录，按「最小必要 + 明示同意 + 可撤回删除」设计：

1. **知情同意**：学生账号须由监护人与学校知情同意后开通；学生端页面明确展示采集范围（画面帧仅用于本地推理展示，服务端只落专注度等结构化指标）。
2. **默认关闭音频**：语音识别链路默认不启用（`ASR_CONFIG.enable_volcengine`/`enable_browser_asr` 需显式开启），系统默认只处理画面，不采集、不存储任何音频。
3. **30 天自动清理**：每条行为记录写入时即生成 `expire_at`（写入时刻 + 30 天），启动期与后台 janitor 定期删除到期数据；`behavior_records(student_id, created_at)` 复合索引支撑按学生批量清理。
4. **一键删除本人数据**：`DELETE /api/student/my-data` 由学生本人（令牌身份）触发，删除其全部行为记录并清理实时画面缓存、在线状态、待推送提醒与持续分心预警状态；**身份只取自 JWT**，不接受外部传入 `student_id`，教师角色调用返回 `403`。
5. **越权与注入防护**：教师/学生接口按角色双向隔离；姓名等回显字段使用字符白名单（拒绝 `<script>` 等载荷，返回 `422`）；前端渲染统一 `esc()` 转义；登录失败限速防爆破。
6. **不对外传输**：除显式启用的第三方语音服务外，画面帧不外发；`/healthz` 等对外响应脱敏，不含路径、密钥与堆栈。

> 合规提示：请依据《未成年人保护法》《个人信息保护法》及教育主管部门要求，完成告知同意、留存期限说明与内部审批后再上线使用。

---

## 七、自动化测试

```bash
python -m pytest tests -v
```

- `tests/conftest.py`：在导入 `app` 之前为缺失的 `cv2` / `mediapipe` 注入轻量桩模块，使测试在**未安装视觉库的环境**也能直接运行；`client` 夹具用 `tmp_path` 隔离 `AI_STUDY_DB`、注入 ≥32 位 JWT 密钥与 CORS 白名单，并在用例前后清理登录限速与内存缓存（通过 `with TestClient(app)` 触发完整 lifespan）。
- `tests/test_core.py`：8 组核心守护用例——13 维特征契约、规则分区间、WebSocket 握手鉴权（4401/4403 与伪造 `student_id` 不污染他人）、越权矩阵（403/404 + XSS 422）、迁移幂等（`expire_at` 列与 `idx_behavior_student_time` 索引）、时间格式（`Z` 结尾的 UTC ISO-8601）与 `/healthz`、教师督学提醒闭环（发送 → 学生轮询取走并销毁 → 参数校验 422）、持续分心自动预警（阈值触发、冷却防刷屏、恢复解除、删除数据清理）。

当前结果：**8 passed**。

---

## 八、目录结构

```
ai-study-buddy/
├── app/                 # 重构后的分层服务包（见第一节）
├── templates/           # 教师端 / 学生端页面模板
├── tests/               # pytest 夹具与核心守护用例
├── docs/                # 各阶段重构报告（P0~P3）
├── app.py               # 兼容外壳（等价于 uvicorn app.main:app）
├── requirements.txt     # 运行时与测试依赖
├── Dockerfile           # 生产镜像
├── .dockerignore        # 构建上下文排除清单
├── .env.example         # 环境变量示例（与本文配置表一致）
└── README.md
```
