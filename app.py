"""
AI学伴 · 1对1专注度智能监护系统
- 学生注册/登录 → 进入自己的专注度监控页面（摄像头）
- 教师登录 → 查看所有学生列表 → 点击学生查看其上课行为
- 集成训练好的专注度模型进行预测
"""

""" AI学伴 · 1对1专注度智能监护系统 """

import asyncio
import base64
import json
import logging
import logging.handlers
import random
import time
import math
import os
import sys
import hmac
import joblib
import threading
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import hashlib
import secrets
import sqlite3
import bcrypt
import jwt
import requests
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager, contextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import cv2                         
import mediapipe as mp          
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== P2-5：结构化日志初始化（控制台 + 滚动文件） ==========
# 约定：模块 logger 固定命名为 "aistudy"；日志目录固定为项目根目录下的 logs/，
# 单文件上限 10MB、保留 5 份历史；全文件不再使用 print，统一走 logger 分级输出。
# 对外 HTTP/WS 响应中严禁携带任何内部异常细节（str(e) / 堆栈），只返回通用安全提示。
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger("aistudy")
# 仅在尚未挂载处理器时初始化，避免 uvicorn --reload 或多进程重复 import 导致日志重复写入
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _LOG_FORMAT = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(_LOG_FORMAT)
    logger.addHandler(_console_handler)
    try:
        # 启动时自动建立日志目录
        os.makedirs(LOG_DIR, exist_ok=True)
        _file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,   # 单文件 10MB
            backupCount=5,               # 保留 5 份滚动历史
            encoding="utf-8",
        )
        _file_handler.setFormatter(_LOG_FORMAT)
        logger.addHandler(_file_handler)
    except OSError as _log_init_err:
        # 日志目录不可写时降级为仅控制台输出，绝不因日志问题阻塞服务启动
        logger.warning("日志文件处理器初始化失败，已降级为仅控制台输出：%s", _log_init_err)
    logger.propagate = False


# ========== 先加载配置 ==========
try:
    from config import (
        VOLC_ACCESS_KEY, VOLC_SECRET_KEY, VOLC_TOKEN_URL, VOLC_ASR_ENDPOINT,
        HOST, PORT, HEAD_DOWN_THRESHOLD, BODY_SWAY_LIMIT, EYE_CLOSED_FRAMES,
        DISTRACTION_INTERVAL, WARNING_LEVELS, ASR_CONFIG, TTS_CONFIG,
        POSE_CONFIG, POSE_THRESHOLDS, JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_DAYS
    )
except ImportError:
    VOLC_ACCESS_KEY = "YOUR_ACCESS_KEY"
    VOLC_SECRET_KEY = "YOUR_SECRET_KEY"
    VOLC_TOKEN_URL = "https://open.volcengineapi.com/api/v3/auth/token"
    VOLC_ASR_ENDPOINT = "https://openspeech.bytedance.com/api/v1/auc/recognize"
    HOST = "0.0.0.0"
    PORT = 8001
    # ===== P1-8：以下默认值已逐项对齐「当前业务代码实际生效的运行值」=====
    # 目的：接入配置项后判定阈值零漂移（原 fallback 中的 25/10/5/{"mild":60,"severe":20}
    # 以及 POSE_CONFIG 的静态图模式、model_complexity=1 等，都是与真实运行行为不一致的陈旧值）
    HEAD_DOWN_THRESHOLD = 0.15          # 明显低头阈值（头部下沉比，越小越严格）
    BODY_SWAY_LIMIT = 15                # 身体摆动上限（×100 百分比刻度）
    EYE_CLOSED_FRAMES = 60              # 确认闭眼所需连续帧数（约 2 秒 @30fps）
    DISTRACTION_INTERVAL = 3            # 连续低专注帧数达到该值计一次分心
    WARNING_LEVELS = {"excellent": 80, "mild": 60, "moderate": 40}   # 建议语分档；低于 moderate 即 severe
    ASR_CONFIG = {"language": "zh-CN", "sample_rate": 16000, "silence_timeout": 120, "enable_volcengine": True, "enable_browser_asr": True}
    TTS_CONFIG = {"enabled": True, "language": "zh-CN", "rate": 0.9, "pitch": 1.1, "volume": 0.8, "cooldown": 30}
    POSE_CONFIG = {"static_image_mode": False, "model_complexity": 2, "min_detection_confidence": 0.4, "min_tracking_confidence": 0.3}
    POSE_THRESHOLDS = {
        "head_tilt_minor": 10, "head_tilt_major": 20,
        "head_drop_minor": 0.35, "head_drop_major": 0.15,
        "body_sway_minor": 8, "body_sway_major": 15,
        "shoulder_level_minor": 0.08, "shoulder_level_major": 0.15,
        "iris_offset_minor": 0.12, "iris_offset_major": 0.25,
        # EAR（眼纵横比）分档：与 _evaluate_fatigue 实际生效值一致
        "ear_fatigue_major": 0.22, "ear_fatigue_minor": 0.30,
        # EAR 缺省值（人脸关键点缺失时的中性假想值）与闭眼帧衰减步长
        "ear_default": 0.3, "eyes_closed_decay": 2,
        # 关键点坐标为 0-1 归一化值，转成百分制时需要乘以该系数（纯单位换算）
        "coord_scale": 100.0, "shoulder_width_eps": 0.01,
        "consecutive_low_focus": 3, "distraction_score_threshold": 60,
    }
    # P0-5：此处不再内置任何可用密钥，留空后交由下方「安全配置加固」段处理
    # （优先读取环境变量 AI_STUDY_JWT_SECRET，未配置则生成进程内随机密钥）
    JWT_SECRET_KEY = ""
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_DAYS = 7

# ========== P1-8：配置归一化（唯一真源原则） ==========
# 说明：旧的四个标量配置项与 POSE_THRESHOLDS 存在语义重叠，这里统一以标量配置为准回灌到阈值字典，
# 保证「无论配置来自 config.py 还是 fallback，判定阈值只有一份来源」，避免同名阈值两处各说各话。
POSE_THRESHOLDS["head_drop_major"] = HEAD_DOWN_THRESHOLD      # 明显低头
POSE_THRESHOLDS["body_sway_major"] = BODY_SWAY_LIMIT          # 明显晃动
POSE_THRESHOLDS["consecutive_low_focus"] = DISTRACTION_INTERVAL

# ========== P1-8：分心扣分表集中（原散落在 _calculate_focus_score 中的魔法数字） ==========
# 全部数值均等于重构前代码中的硬编码字面量，接入后评分行为零漂移。
SCORE_PENALTIES = {
    "head_pose_major": 20,            # 头部明显偏移
    "head_pose_minor": 10,            # 头部轻微偏移
    "head_down_major": 8,             # 明显低头
    "head_down_minor": 4,             # 轻微低头
    "body_major": 12,                 # 明显晃动
    "body_minor": 5,                  # 轻微晃动
    "fatigue_major": 15,              # 明显疲劳
    "fatigue_minor": 8,               # 轻度疲劳
    "eye_closed": 30,                 # 闭眼/走神
    "eye_blur": 18,                   # 视线模糊
    "eye_deviation_frequent": 12,     # 视线频繁偏离
    "eye_deviation_occasional": 3,    # 视线偶尔偏离（正常眨眼级）
    "stability_bonus": 4,             # 姿态高度稳定加分
    "stability_penalty": 3,           # 姿态极不稳定扣分
    "stability_high": 0.85,           # 触发加分的稳定性门槛
    "stability_low": 0.2,             # 触发扣分的稳定性门槛
    "scene_onetoone_bonus": 3,        # 一对一场景加严放宽
    "scene_online_penalty": 3,        # 线上场景加严
    "score_min": 10,                  # 分数下限
    "score_max": 100,                 # 分数上限
}

# ========== P1-8：动作小分类几何阈值集中（原散落在 _classify_action 中的魔法数字） ==========
# 数值与重构前完全一致，行为零漂移。
ACTION_THRESHOLDS = {
    "sleep_head_drop_ratio": 0.15,        # 鼻子低于肩线的比例 → 判睡觉
    "sleep_head_drop_ratio_loose": 0.08,  # 更宽松的低头比例（需同时明显疲劳）
    "wrist_torso_above_shoulder": 0.15,   # 手腕允许高于肩线的距离
    "wrist_torso_below_hip": 0.05,        # 手腕允许低于髋线的距离
    "wrist_center_ratio": 0.8,            # 手腕距中线小于肩宽的该比例视为靠近身体中心
    "elbow_bend_dy": 0.15,                # 手腕与肘部 y 差值小于该值视为肘部弯曲
    "action_history_size": 8,             # 动作历史窗口
    "action_vote_window": 5,              # 众数投票窗口
    "action_vote_min": 3,                 # 触发投票的最小历史帧数
}

# ========== P1-8：分数时序平滑参数集中（原散落在 _smooth_focus_score 中的魔法数字） ==========
SCORE_SMOOTHING = {
    "history_size": 15,      # 平滑历史窗口
    "min_history": 3,        # 触发平滑的最小历史帧数
    "decay": 0.9,            # 指数平滑衰减系数
    "inertia_window": 5,     # 惯性判断窗口
    "inertia_gap": 8,        # 与近期均值差距小于该值才施加惯性
    "inertia_blend": 0.7,    # 惯性混合权重（平滑值占比，其余为近期均值）
}

# ========== P1-8：姿态稳定性（时序方差）参数集中 ==========
POSE_STABILITY = {
    "history_size": 10,          # 历史窗口
    "min_history": 3,            # 触发计算的最小历史帧数
    "tilt_var_scale": 100.0,     # 头部水平偏移方差归一系数
    "shoulder_var_scale": 10.0,  # 肩高差方差归一系数
}

# ========== 安全配置加固（P0-5 / P0-6） ==========
# P0-5 JWT 密钥：不再使用硬编码兜底密钥，改为「环境变量优先 + 缺失时进程内随机生成」。
#   环境变量名：AI_STUDY_JWT_SECRET
#   设置示例（Windows PowerShell）：$env:AI_STUDY_JWT_SECRET="<足够长的随机字符串>"
#   若未设置、或仍是示例占位串，则生成一次性随机密钥：服务可正常启动，
#   但重启后旧令牌全部失效、需重新登录。生产环境务必显式配置该环境变量。
_INSECURE_JWT_SECRETS = {
    "",
    "your-secret-key-change-in-production",
    "change-me",
    "changeme",
    "secret",
}

_ENV_JWT_SECRET = (os.getenv("AI_STUDY_JWT_SECRET") or "").strip()
if _ENV_JWT_SECRET:
    JWT_SECRET_KEY = _ENV_JWT_SECRET
if str(JWT_SECRET_KEY or "").strip() in _INSECURE_JWT_SECRETS:
    JWT_SECRET_KEY = secrets.token_urlsafe(48)
    logger.warning("[安全警告] 未配置环境变量 AI_STUDY_JWT_SECRET，已生成进程内临时 JWT 密钥：服务重启后所有登录令牌将失效，生产环境请务必配置该环境变量。")

# P0-6 CORS：禁止 allow_origins=["*"] 与 allow_credentials=True 同时出现（浏览器规范不允许该组合）。
#   环境变量名：AI_STUDY_CORS_ORIGINS，多个来源用英文逗号分隔
#   设置示例：http://localhost:8001,https://foucs-analyze.cyou
#   未配置时仅放行本机同端口来源，避免任意站点携带凭证跨域访问接口。
_ENV_CORS_ORIGINS = (os.getenv("AI_STUDY_CORS_ORIGINS") or "").strip()
if _ENV_CORS_ORIGINS:
    CORS_ALLOW_ORIGINS = [o.strip() for o in _ENV_CORS_ORIGINS.split(",") if o.strip()]
else:
    CORS_ALLOW_ORIGINS = [f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"]

# 通配来源与凭证模式互斥：来源中出现 "*" 时强制关闭凭证，防止任意站点越权读取 Cookie / Authorization
CORS_ALLOW_CREDENTIALS = "*" not in CORS_ALLOW_ORIGINS
if not CORS_ALLOW_CREDENTIALS:
    logger.warning("[安全警告] CORS 来源白名单中包含通配符 \"*\"，已自动关闭 allow_credentials。")


# ========== P2-2：有界推理线程池 ==========
# 说明：MediaPipe 推理是 CPU 密集的同步调用，直接放在 /ws 协程里会阻塞整个事件循环，
# 导致其他学生的 REST 请求（列表/记录/提醒）排队超时。这里用固定 4 个工作线程的池承接推理，
# 线程名统一加 "infer" 前缀便于日志排查；进程退出时在 lifespan 中 shutdown。
_INFER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="infer")

# ========== P2-2：单学生背压集合 ==========
# 记录「当前已有帧正在推理中」的学生 id：同一学生上一帧未算完时，新帧直接丢弃，
# 避免弱网/低配机器上帧缓冲无限积压导致的雪崩。
# 该集合只在事件循环线程内增删（工作线程不触碰），因此无需额外加锁。
_inflight_students: set = set()


# ========== P2-3：带 TTL 与容量上限的线程安全内存缓存 ==========
class TTLCache:
    """轻量 TTL + LRU 容量上限缓存（防止长期运行的全局字典无限膨胀导致 OOM）

    设计要点：
    - 用 OrderedDict 维护插入顺序，配合 move_to_end 实现 LRU 淘汰；
    - 全部读写都在 RLock 保护下进行，可被事件循环线程与推理工作线程同时访问；
    - get / items / __contains__ / __len__ 均会顺带判断过期（懒惰淘汰），
      另有 cleanup() 供后台任务批量回收，返回本次实际清除的条数；
    - set 时若 key 已存在先移除再插入，保证 LRU 顺序与「最新一次写入」一致。
    """

    def __init__(self, maxsize=1000, ttl=60.0):
        self.maxsize = int(maxsize)
        self.ttl = float(ttl)
        self._store = OrderedDict()      # key -> (expire_at_ts, value)
        self._lock = threading.RLock()

    # ---------- 内部工具 ----------
    def _is_expired(self, expire_at) -> bool:
        """按单调时钟判断是否过期（避免系统时间回拨导致条目永不过期）"""
        return expire_at <= time.monotonic()

    def _purge_locked(self) -> int:
        """在已持锁前提下清除所有过期条目，返回清除数量"""
        now = time.monotonic()
        dead = [k for k, (exp, _) in self._store.items() if exp <= now]
        for k in dead:
            self._store.pop(k, None)
        return len(dead)

    def _evict_locked(self):
        """超容量时按 LRU 淘汰最旧条目"""
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    # ---------- 对外接口 ----------
    def set(self, key, value):
        """写入/覆盖一条记录，TTL 从本次写入时刻重新计时"""
        with self._lock:
            self._store.pop(key, None)
            self._store[key] = (time.monotonic() + self.ttl, value)
            self._evict_locked()

    def get(self, key, default=None):
        """读取记录；不存在或已过期一律返回 default"""
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return default
            expire_at, value = item
            if self._is_expired(expire_at):
                # 懒惰淘汰：读到过期项立即清除，保证教师端不会看到离线学生的旧画面
                self._store.pop(key, None)
                return default
            self._store.move_to_end(key)
            return value

    def pop(self, key, default=None):
        """取出并删除（用于提醒的一次性消费）"""
        with self._lock:
            item = self._store.pop(key, None)
            if item is None:
                return default
            expire_at, value = item
            if self._is_expired(expire_at):
                return default
            return value

    def cleanup(self) -> int:
        """清除全部过期条目，返回清除数量（供后台 janitor 定时调用）"""
        with self._lock:
            return self._purge_locked()

    def items(self):
        """返回未过期条目的快照列表（调试接口用，避免持锁遍历时被外部修改）"""
        with self._lock:
            now = time.monotonic()
            return [(k, v) for k, (exp, v) in self._store.items() if exp > now]

    def __contains__(self, key) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            self._purge_locked()
            return len(self._store)


# ========== P2-3：后台缓存清理协程 ==========
async def _cache_janitor():
    """每 60 秒清理一次三个内存缓存中的过期条目（防 OOM）"""
    while True:
        try:
            await asyncio.sleep(60)
            purged_video = student_video_cache.cleanup()
            purged_online = online_students.cleanup()
            purged_notice = student_notifications.cleanup()
            if purged_video or purged_online or purged_notice:
                logger.info(
                    "[缓存清理] 过期回收：视频帧 %s 条 / 在线状态 %s 条 / 提醒 %s 条",
                    purged_video, purged_online, purged_notice,
                )
        except asyncio.CancelledError:
            # 生命周期结束时被 cancel，正常退出
            raise
        except Exception:
            # 清理任务本身异常不能拖垮主服务
            logger.exception("[缓存清理] 后台清理任务执行异常")


# ========== MediaPipe 姿态识别 ==========
# P1-6：数据库建表/迁移与模型加载统一放在 lifespan 中，
# 保证无论是 `python app.py` 还是 `uvicorn app:app` 启动都会自动执行一次。
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：启动阶段完成建表、历史表迁移与专注度模型校验加载，
    并启动缓存清理后台任务；关闭阶段优雅取消该任务并释放推理线程池。"""
    init_db()
    load_trained_model()
    # P2-3：启动后台缓存清理任务
    janitor_task = asyncio.create_task(_cache_janitor())
    logger.info("[启动] 数据库初始化与模型加载完成，缓存清理任务已启动")
    try:
        yield
    finally:
        # P2-3：优雅取消后台清理任务
        janitor_task.cancel()
        try:
            await janitor_task
        except asyncio.CancelledError:
            pass
        # P2-2：不等待在途推理完成，避免退出阻塞（守护语义）
        _INFER_POOL.shutdown(wait=False)
        logger.info("[退出] 应用已关闭，后台任务与推理线程池已释放")

app = FastAPI(lifespan=lifespan)
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,            # P0-6：来源白名单，由环境变量 AI_STUDY_CORS_ORIGINS 配置
    allow_credentials=CORS_ALLOW_CREDENTIALS,    # P0-6：来源含通配符时强制关闭凭证
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== P2-5：全局兜底异常处理器（对外脱敏） ==========
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    """兜底捕获未处理异常：日志留完整堆栈，响应只回通用安全提示

    严禁把 str(exc)、异常类型名、堆栈等内部细节写进响应体，避免泄露表名、
    文件路径、第三方接口地址等敏感实现细节。
    """
    logger.exception("未处理异常：%s %s", getattr(request, "method", "-"), getattr(request.url, "path", "-"))
    return JSONResponse(status_code=500, content={"detail": "服务暂时不可用"})


# ========== MediaPipe 初始化 ==========
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# ========== P2-1：MediaPipe 实例按线程隔离（消除全局单例串台） ==========
# 背景：MediaPipe 的 Pose / FaceMesh 对象内部持有 C++ 图与状态，不是线程安全的。
# 原实现在模块加载时创建一份全局单例，被所有学生、所有工作线程共享，存在
# 「学生 A 的推理结果被学生 B 的帧覆盖 / 崩溃」风险。
# 现改为 threading.local() 按工作线程懒加载：每个推理线程各持一份独立实例。
_MP_TLS = threading.local()


def _build_pose_config():
    """构造 Pose 初始化参数（P2-1：强制逐帧独立模式）"""
    cfg = dict(POSE_CONFIG or {})
    # 关键防串台修正：推理在公共线程池中调度，同一线程会先后服务不同学生。
    # 若沿用视频流跟踪模式（static_image_mode=False），线程复用时上一帧的 ROI 跟踪框会
    # 污染下一个学生的检测结果；因此这里强制 static_image_mode=True，逐帧独立、无状态检测。
    cfg["static_image_mode"] = True
    return cfg


def _get_pose():
    """获取当前线程独占的 Pose 实例（懒加载）"""
    inst = getattr(_MP_TLS, "pose", None)
    if inst is None:
        inst = mp_pose.Pose(**_build_pose_config())
        _MP_TLS.pose = inst
    return inst


def _get_face_mesh():
    """获取当前线程独占的 FaceMesh 实例（懒加载，同样强制静态图模式）"""
    inst = getattr(_MP_TLS, "face_mesh", None)
    if inst is None:
        inst = mp_face_mesh.FaceMesh(
            static_image_mode=True,      # P2-1：逐帧独立检测，杜绝跨学生跟踪状态污染
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        )
        _MP_TLS.face_mesh = inst
    return inst


# ========== 模型加载（全局缓存，只加载一次）==========
# P0-7：模型反序列化加固 —— 彻底移除不安全的反序列化调用，改用 joblib；且必须通过 SHA256 白名单校验
# P1-1：模型目录真实名称为 model/（历史代码误写为 models/），并支持环境变量覆盖
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = (os.getenv("AI_STUDY_MODEL_PATH") or "").strip() or os.path.join(MODEL_DIR, "focus_model.pkl")
MODEL_SHA256_ENV = "AI_STUDY_MODEL_SHA256"   # 模型文件 SHA256 白名单（环境变量）
MODEL_ENABLE_ENV = "AI_STUDY_ENABLE_MODEL"   # 模型推理开关（默认 "0"：纯规则引擎）

# ========== P1-2：13 维特征契约 ==========
# 说明：仓库历史上同时存在两套特征口径 —— MediaPipePoseAnalyzer 内的 51 维（17 点 × 3 坐标）
# 与模块级 extract_pose_features 的 13 维。本地训练模型 n_features_in_ = 13，
# 故此处以 13 维为唯一契约，并把该几何定义提为唯一实现（extract_model_features），51 维口径彻底移除。
FEATURE_VERSION = "pose13-v1"
FEATURE_DIM = 13
# 特征顺序（每个位置的语义，改动顺序必须同步升 FEATURE_VERSION）
FEATURE_LAYOUT = (
    "head_tilt",            # 0  双耳连线倾角（弧度）
    "head_dx", "head_dy", "head_dz",   # 1-3 鼻子相对肩中心的偏移量
    "eye_distance",         # 4  双眼间距
    "shoulder_level",       # 5  双肩高度差
    "body_angle",           # 6  双肩连线倾角（弧度）
    "head_shoulder_dist",   # 7  头-肩中心距离
    "eye_shoulder_dist",    # 8  眼中心-肩中心距离
    "hip_cx", "hip_cy", "hip_cz",      # 9-11 髋中心坐标（躯干基准点）
    "body_lean",            # 12 肩中心与髋中心的深度差（前倾/后仰）
)
# 参与计算的 MediaPipe 关键点索引（33 点姿态模型下标）
FEATURE_LANDMARK_INDICES = (0, 2, 5, 7, 8, 11, 12, 23, 24)

# ========== P1-4：模型置信度门限 ==========
# 模型置信度低于该值时不对外输出模型判断，只保留规则引擎结论，避免低质量推理污染展示
MODEL_MIN_CONFIDENCE = 0.65

_cached_model = None
_cached_model_loaded = False
# P1-3：仅在模型文件是 {"model":..., "scaler":..., "feature_version":...} 字典包时才会有值
_cached_scaler = None


def _verify_model_sha256(path: str, expected_sha256: str) -> None:
    """校验模型文件 SHA256（分块流式计算，避免大文件占满内存）

    校验不通过一律抛出 RuntimeError，阻止后续反序列化。
    """
    expected = (expected_sha256 or "").strip().lower()
    if not expected:
        raise RuntimeError("模型文件 SHA256 校验失败，拒绝加载")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError("模型文件 SHA256 校验失败，拒绝加载")


def extract_model_features(landmarks):
    """从 MediaPipe 33 点姿态关键点提取 13 维模型特征（唯一实现）

    - 输入：results.pose_landmarks.landmark（长度 33）
    - 输出：np.ndarray，形状恒为 (FEATURE_DIM,)，与 FEATURE_LAYOUT 顺序严格对应
    - 返回 None 表示提取失败（关键点缺失/数值异常）
    """
    try:
        if landmarks is None or len(landmarks) < 33:
            raise ValueError(f"关键点数量不足：{0 if landmarks is None else len(landmarks)} < 33")

        def _lm(idx):
            lm = landmarks[idx]
            return np.array([float(lm.x), float(lm.y), float(lm.z)], dtype=np.float64)

        nose = _lm(0)
        left_eye, right_eye = _lm(2), _lm(5)
        left_ear, right_ear = _lm(7), _lm(8)
        left_shoulder, right_shoulder = _lm(11), _lm(12)
        left_hip, right_hip = _lm(23), _lm(24)

        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        eye_center = (left_eye + right_eye) / 2.0
        hip_center = (left_hip + right_hip) / 2.0
        head_relative = nose - shoulder_center

        features = [
            float(np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])),   # head_tilt
            float(head_relative[0]), float(head_relative[1]), float(head_relative[2]),   # head_dx/dy/dz
            float(np.linalg.norm(left_eye - right_eye)),                                 # eye_distance
            float(abs(left_shoulder[1] - right_shoulder[1])),                            # shoulder_level
            float(np.arctan2(right_shoulder[1] - left_shoulder[1],
                             right_shoulder[0] - left_shoulder[0])),                     # body_angle
            float(np.linalg.norm(nose - shoulder_center)),                               # head_shoulder_dist
            float(np.linalg.norm(eye_center - shoulder_center)),                         # eye_shoulder_dist
            float(hip_center[0]), float(hip_center[1]), float(hip_center[2]),            # hip_cx/cy/cz
            float(shoulder_center[2] - hip_center[2]),                                   # body_lean
        ]
        arr = np.asarray(features, dtype=np.float64)
        if arr.shape != (FEATURE_DIM,):
            raise ValueError(f"特征维度异常：{arr.shape}，契约要求 ({FEATURE_DIM},)")
        if not np.all(np.isfinite(arr)):
            raise ValueError("特征包含非有限值（NaN/Inf）")
        return arr
    except Exception as e:
        logger.warning("模型特征提取失败: %s", e)
        return None


def assert_model_contract(model):
    """P1-2：启动期特征契约自检

    校验模型声明的输入维度与代码 FEATURE_DIM 完全一致；不一致直接抛 RuntimeError，
    绝不允许「加载成功、运行时才对不上」的静默失败。
    """
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        raise RuntimeError(
            f"模型契约校验失败：加载对象（{type(model).__name__}）缺少 n_features_in_，"
            f"无法确认其输入维度，代码当前契约 {FEATURE_VERSION}={FEATURE_DIM} 维"
        )
    if int(n_features) != FEATURE_DIM:
        raise RuntimeError(
            f"模型契约校验失败：模型期望 {int(n_features)} 维特征，代码当前输出 {FEATURE_DIM} 维"
            f"（{FEATURE_VERSION}），二者不一致，拒绝启用模型推理"
        )
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) != FEATURE_DIM:
        raise RuntimeError(
            f"模型契约校验失败：feature_names_in_ 长度 {len(names)} 与契约 {FEATURE_DIM} 维不一致"
        )
    return True


def load_trained_model():
    """加载训练好的模型（全局缓存，进程内只尝试一次）

    P1-1：默认「纯规则引擎模式」，只有显式设置 AI_STUDY_ENABLE_MODEL=1 才尝试加载，
          避免本地存在模型文件却未配哈希时阻塞启动。
    P0-7：启用模型后必须通过 AI_STUDY_MODEL_SHA256 校验，才允许 joblib.load。
    P1-2：加载后立即执行 assert_model_contract 维度自检。
    P1-3：兼容 {"model": clf, "scaler": sc, "feature_version": ...} 字典包。
    """
    global _cached_model, _cached_scaler, _cached_model_loaded
    if _cached_model_loaded:
        return _cached_model

    # 开关关闭：直接进入纯规则引擎模式（不读文件、不校验哈希、不抛错）
    if (os.getenv(MODEL_ENABLE_ENV) or "0").strip() != "1":
        _cached_model_loaded = True   # 纯规则模式为稳定状态，可安全缓存
        logger.info("当前处于纯规则引擎模式（未启用模型推理）：如需启用请设置 %s=1", MODEL_ENABLE_ENV)
        return None

    # 开关打开：模型文件、哈希、契约三者缺一不可，任何一步失败都直接抛错
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"已开启模型推理（{MODEL_ENABLE_ENV}=1）但未找到模型文件: {MODEL_PATH}")

    expected_sha256 = (os.getenv(MODEL_SHA256_ENV) or "").strip()
    if not expected_sha256:
        raise RuntimeError(f"未配置环境变量 {MODEL_SHA256_ENV}，拒绝加载模型文件")

    _verify_model_sha256(MODEL_PATH, expected_sha256)   # 校验失败直接抛错，不降级

    try:
        loaded = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"模型文件加载失败，拒绝启用模型: {e}")

    # P1-3：字典包解包（训练管线带标准化器时使用）
    if isinstance(loaded, dict):
        bundle_version = loaded.get("feature_version")
        if bundle_version != FEATURE_VERSION:
            raise RuntimeError(
                f"模型包特征版本不匹配：文件为 {bundle_version!r}，代码期望 {FEATURE_VERSION!r}"
                f"（{FEATURE_DIM} 维），拒绝启用"
            )
        model = loaded.get("model")
        if model is None:
            raise RuntimeError("模型包格式非法：缺少 model 字段")
        scaler = loaded.get("scaler")
    else:
        model = loaded
        scaler = None

    assert_model_contract(model)   # P1-2：维度不符直接 RuntimeError

    _cached_model = model
    _cached_scaler = scaler
    _cached_model_loaded = True    # 仅在完全加载成功后置位，避免失败被缓存成「静默降级」
    logger.info(
        "专注度模型已通过 SHA256 校验并加载: %s (特征维度=%s, 版本=%s, 标准化=%s)",
        MODEL_PATH, FEATURE_DIM, FEATURE_VERSION, "是" if scaler is not None else "否",
    )
    return _cached_model


def predict_focus(features):
    """模型推理（P1-4：保留 predict_proba 原始概率语义，不做任何区间重映射）

    返回 (结果字典, prob_focused)；模型不可用或推理失败返回 (None, 0.0)。
    """
    global _cached_model, _cached_scaler
    if _cached_model is None:
        return None, 0.0
    try:
        arr = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if arr.shape[1] != FEATURE_DIM:
            raise ValueError(f"特征维度不匹配：输入 {arr.shape[1]} 维，契约要求 {FEATURE_DIM} 维（{FEATURE_VERSION}）")
        if _cached_scaler is not None:
            arr = _cached_scaler.transform(arr)
        if not hasattr(_cached_model, "predict_proba"):
            logger.warning("模型推理失败: 当前模型不支持 predict_proba")
            return None, 0.0
        proba = _cached_model.predict_proba(arr)[0]
        # 类别顺序约定为 [分心, 专注]（与训练脚本 classes_ 一致）；若模型自带 classes_ 则以其为准，
        # 避免类别顺序变化导致概率语义被静默颠倒。
        classes = getattr(_cached_model, "classes_", None)
        focused_idx = 1
        if classes is not None:
            cls_list = list(classes)
            if 1 in cls_list:
                focused_idx = cls_list.index(1)
            elif len(cls_list) != 2:
                raise ValueError(f"模型类别数异常：{cls_list}，期望二分类 [0, 1]")
        prob_focused = round(float(proba[focused_idx]), 4)
        prob_distracted = round(1.0 - prob_focused, 4)
        return {
            "is_focused": prob_focused > 0.5,
            "confidence": round(max(prob_focused, prob_distracted), 4),
            "label": "专注" if prob_focused > 0.5 else "分心",
            "prob_focused": prob_focused,
            "prob_distracted": prob_distracted,
        }, prob_focused
    except Exception as e:
        logger.warning("模型推理失败: %s", e)
        return None, 0.0


def extract_pose_features(image_path):
    """从图片文件中提取 13 维姿态特征（P1-2：统一委托给 extract_model_features）"""
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        return extract_pose_features_from_frame(frame)
    except Exception as e:
        logger.warning("特征提取失败: %s", e)
        return None


def extract_pose_features_from_frame(frame):
    """从内存中的视频帧提取 13 维姿态特征（避免磁盘 IO）"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _get_pose().process(rgb_frame)
        if not results.pose_landmarks:
            return None
        return extract_model_features(results.pose_landmarks.landmark)
    except Exception as e:
        logger.warning("内存特征提取失败: %s", e)
        return None

# ========== 数据库初始化 ==========
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_study_buddy.db")

def init_db():
    """初始化数据库（建表 / 历史表迁移 / 索引创建 / 留存清理 / 管理员初始化）"""
    with get_db() as conn:
        _init_db_schema(conn)
    logger.info("数据库初始化完成")

def _init_db_schema(conn):
    """init_db 的实际执行体（传入已开启 WAL 与事务的连接，提交与关闭由 get_db 统一负责）"""
    cursor = conn.cursor()

    # ===== 教师表 =====
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ===== 学生表（学生自己注册）=====
    # 先检查旧表结构是否需要迁移
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    table_exists = cursor.fetchone() is not None

    need_migration = False
    if table_exists:
        # 检查是否有 username 列
        cursor.execute("PRAGMA table_info(students)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'username' not in columns or 'password_hash' not in columns:
            need_migration = True
            logger.warning("检测到旧版 students 表结构，正在执行迁移")

    if need_migration:
        # 备份旧数据
        cursor.execute("ALTER TABLE students RENAME TO students_old")
        # 创建新表
        cursor.execute('''
            CREATE TABLE students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                student_no TEXT DEFAULT '',
                class_name TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 迁移旧数据（为旧学生生成默认用户名和密码）
        # P0-9：不再使用可猜测的旧默认弱口令，改为随机不可预测口令的哈希
        cursor.execute('''
            INSERT INTO students (id, username, password_hash, name, student_no, class_name, created_at)
            SELECT id, 
                   'student_' || id, 
                   ?, 
                   COALESCE(name, '学生' || id), 
                   COALESCE(student_no, ''), 
                   COALESCE(class_name, ''), 
                   COALESCE(created_at, CURRENT_TIMESTAMP)
            FROM students_old
        ''', (hash_password(secrets.token_urlsafe(16)),))
        # 删除旧表
        cursor.execute("DROP TABLE students_old")
        logger.info("students 表迁移完成")
    else:
        # 直接创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                student_no TEXT DEFAULT '',
                class_name TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    # ===== 行为记录表 =====
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS behavior_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            focus_score INTEGER,
            distraction_count INTEGER,
            head_pose TEXT,
            body_stability TEXT,
            fatigue_level TEXT,
            eye_focus TEXT,
            suggestion TEXT,
            model_prediction TEXT,  -- 新增：模型预测结果
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expire_at TIMESTAMP,    -- P0-10：数据留存到期时间（默认写入后 30 天）
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')

    # ===== P0-10：行为记录 30 天留存策略 =====
    # 1) 为已存在的旧表幂等补充 expire_at 字段（重复执行不会报错）
    cursor.execute("PRAGMA table_info(behavior_records)")
    _br_columns = [col[1] for col in cursor.fetchall()]
    if "expire_at" not in _br_columns:
        cursor.execute("ALTER TABLE behavior_records ADD COLUMN expire_at TIMESTAMP")
        logger.info("behavior_records 表已补充 expire_at 字段")
    # 2) 回填历史记录的到期时间（写入时间 + 30 天）
    cursor.execute(
        "UPDATE behavior_records SET expire_at = datetime(created_at, '+30 days') "
        "WHERE expire_at IS NULL AND created_at IS NOT NULL"
    )
    # 3) 清理已过期记录（未成年人隐私最小化：超期数据不留存）
    cursor.execute(
        "DELETE FROM behavior_records "
        "WHERE (expire_at IS NOT NULL AND expire_at < CURRENT_TIMESTAMP) "
        "   OR (expire_at IS NULL AND created_at < datetime('now', '-30 days'))"
    )
    if cursor.rowcount and cursor.rowcount > 0:
        logger.info("已清理 %s 条超过 30 天留存期的行为记录", cursor.rowcount)

    # ===== 管理员账号初始化（P0-4：不再内置 teacher 默认弱口令）=====
    # 仅当环境变量 AI_STUDY_ADMIN_PASSWORD 非空、且库中不存在 teacher 账号时才创建
    _admin_password = (os.getenv("AI_STUDY_ADMIN_PASSWORD") or "").strip()
    cursor.execute("SELECT COUNT(*) FROM teachers WHERE username = 'teacher'")
    if cursor.fetchone()[0] == 0:
        if _admin_password:
            cursor.execute(
                "INSERT INTO teachers (username, password_hash) VALUES (?, ?)",
                ("teacher", hash_password(_admin_password))
            )
            logger.info("已依据环境变量 AI_STUDY_ADMIN_PASSWORD 初始化管理员账号：teacher")
        else:
            logger.info("未配置环境变量 AI_STUDY_ADMIN_PASSWORD，跳过管理员账号初始化")

    # ===== P2-4：复合索引（幂等创建）=====
    # 用途：学生最新一条行为记录的查询（WHERE student_id=? ORDER BY created_at DESC LIMIT 1）
    # 由全表扫描 + 临时排序，降为一次索引定位；教师端批量取最新记录的分组聚合也能借助
    # student_id 有序前缀完成 GROUP BY，避免额外排序。
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_behavior_student_time "
        "ON behavior_records(student_id, created_at DESC)"
    )
    logger.info("复合索引 idx_behavior_student_time 已就绪")

# ========== JWT 认证工具 ==========
def create_jwt_token(user_id: int, role: str, username: str) -> str:
    """创建JWT令牌，包含角色信息

    P1-7：统一使用带时区的 datetime.now(timezone.utc)，替代已弃用的 datetime.utcnow()，
    避免生成无时区信息的“朴素时间”导致前后端解析出现 8 小时偏差。
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub": username,
        "user_id": user_id,
        "role": role,
        "exp": expire,
        "iat": now
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    """验证JWT令牌"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token已过期")
    except jwt.InvalidTokenError:  # PyJWT 2.x：JWTError 已移除，改用 InvalidTokenError 基类
        raise HTTPException(status_code=401, detail="无效的Token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前登录用户"""
    token = credentials.credentials
    return verify_jwt_token(token)

# ========== 时间格式工具（P1-7） ==========
def to_iso_utc(v):
    """将 SQLite 存储的 UTC 时间字符串转为带 Z 后缀的 ISO-8601 格式"""
    if not v:
        return None
    s = str(v).replace(" ", "T")
    return s if s.endswith("Z") or "+" in s else s + "Z"

# ========== 密码安全（P0-4） ==========
def hash_password(password: str) -> str:
    """生成密码哈希：bcrypt（随机盐，cost=12）"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

def is_bcrypt_hash(hashed: str) -> bool:
    """判断哈希是否为 bcrypt 格式（$2a$ / $2b$ / $2y$）"""
    return isinstance(hashed, str) and hashed.startswith("$2")

def verify_password(password: str, hashed: str) -> bool:
    """校验密码：bcrypt 哈希用 bcrypt.checkpw；历史无盐 SHA256 哈希用常量时间比对"""
    if not hashed:
        return False
    if is_bcrypt_hash(hashed):
        try:
            return bcrypt.checkpw((password or "").encode("utf-8"), hashed.encode("utf-8"))
        except (ValueError, TypeError):
            return False
    # 兼容历史 64 位十六进制无盐 SHA256 哈希
    legacy_hash = hashlib.sha256((password or "").encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy_hash, hashed)

def upgrade_password_hash(role: str, user_id: int, new_hash: str) -> None:
    """将密码哈希原地升级（仅用于历史 SHA256 → bcrypt 的平滑迁移）"""
    table = "students" if role == "student" else "teachers"
    # 表名由固定分支产生（非用户输入），不存在 SQL 注入风险
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE " + table + " SET password_hash = ? WHERE id = ?", (new_hash, user_id))
    except Exception:
        # 内部堆栈只进日志，绝不返回给前端
        logger.exception("密码哈希升级失败（%s:%s）", role, user_id)

# ========== 登录失败限速（P0-4） ==========
_LOGIN_MAX_FAILS = 5          # 连续失败次数上限
_LOGIN_LOCK_SECONDS = 900     # 锁定时长：15 分钟
_login_fail = {}              # 结构：{"role:username": {"count": int, "locked_until": float}}
_login_fail_lock = threading.Lock()   # 保护 _login_fail 的模块级锁

def _login_key(role: str, username: str) -> str:
    """限速键：角色 + 用户名（统一小写、去空格，避免大小写绕过）"""
    return f"{role}:{(username or '').strip().lower()}"

def _raise_if_login_locked(role: str, username: str) -> None:
    """锁定期内直接返回 429"""
    key = _login_key(role, username)
    now = time.time()
    remain = 0
    with _login_fail_lock:
        record = _login_fail.get(key)
        if not record:
            return
        locked_until = record.get("locked_until", 0)
        if locked_until and locked_until > now:
            remain = int(locked_until - now) + 1
        elif locked_until:
            # 锁定期已结束，重置记录
            _login_fail.pop(key, None)
            return
        else:
            return
    raise HTTPException(status_code=429, detail=f"登录失败次数过多，请在 {remain} 秒后重试")

def _record_login_failure(role: str, username: str) -> None:
    """记录一次登录失败，达到上限则锁定"""
    key = _login_key(role, username)
    now = time.time()
    with _login_fail_lock:
        record = _login_fail.setdefault(key, {"count": 0, "locked_until": 0.0})
        if record.get("locked_until") and record["locked_until"] <= now:
            record["count"] = 0
            record["locked_until"] = 0.0
        record["count"] += 1
        if record["count"] >= _LOGIN_MAX_FAILS:
            record["locked_until"] = now + _LOGIN_LOCK_SECONDS

def _clear_login_failures(role: str, username: str) -> None:
    """登录成功后清零失败计数"""
    with _login_fail_lock:
        _login_fail.pop(_login_key(role, username), None)

# ========== 数据模型 ==========
# P0-9：姓名白名单 —— 仅允许中文、字母、数字、下划线、间隔号、连字符与空白，长度 1~32
# 该正则从输入侧拦截 < > " ' & 等可构成 XSS 载荷的特殊字符
NAME_PATTERN = r"^[\u4e00-\u9fa5A-Za-z0-9_·\-\s]{1,32}$"
# 学号/班级允许为空，字符白名单与姓名一致
NAME_OPT_PATTERN = r"^[\u4e00-\u9fa5A-Za-z0-9_·\-\s]{0,32}$"


class StudentRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=64)
    # 姓名与学号/班级会回显到教师端页面，必须做长度与字符白名单约束（存储型 XSS 防护）
    name: str = Field(..., min_length=1, max_length=32, pattern=NAME_PATTERN)
    student_no: str = Field(default="", max_length=32, pattern=NAME_OPT_PATTERN)
    class_name: str = Field(default="", max_length=32, pattern=NAME_OPT_PATTERN)


class LoginRequest(BaseModel):
    username: str
    password: str

# ========== 数据库操作 ==========
# P2-4：get_db 改为上下文管理器，统一连接开箱参数与自动事务收尾
@contextmanager
def get_db():
    """数据库连接上下文管理器

    统一行为（每次取连接即就位，避免各处漏设 PRAGMA）：
      1) journal_mode=WAL：读写并发互不阻塞，学生持续写行为记录时教师端查询不会长时间被锁；
      2) foreign_keys=ON：SQLite 默认不启用外键约束，这里显式打开以保证引用完整性；
      3) synchronous=NORMAL：WAL 模式下兼顾安全与写入性能的平衡点；
      4) 正常退出自动 commit()，抛异常自动 rollback()，finally 必定 close()。
    使用方式统一为 `with get_db() as conn:`，不再需要散落的手动 close()。
    """
    conn = sqlite3.connect(DB_FILE, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def get_teacher_by_username(username: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM teachers WHERE username = ?", (username,))
        return cursor.fetchone()

def get_student_by_username(username: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE username = ?", (username,))
        return cursor.fetchone()

def create_student(username: str, password: str, name: str, student_no: str, class_name: str):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO students (username, password_hash, name, student_no, class_name) VALUES (?, ?, ?, ?, ?)",
                (username, hash_password(password), name, student_no, class_name)
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        # 用户名唯一约束冲突：交由上层返回友好提示
        return None

def get_all_students():
    """教师获取所有学生"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, name, student_no, class_name, created_at FROM students ORDER BY created_at DESC")
        students = cursor.fetchall()
    result = []
    for s in students:
        item = dict(s)
        item["created_at"] = to_iso_utc(item.get("created_at"))
        result.append(item)
    return result

def get_student_by_id(student_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, name, student_no, class_name, created_at FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
    if not student:
        return None
    item = dict(student)
    item["created_at"] = to_iso_utc(item.get("created_at"))
    return item

def save_behavior_record(student_id: int, data: dict):
    # P0-10：写入时即设定 30 天留存到期时间（SQLite datetime('now') 为 UTC，与 created_at 默认值一致）
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO behavior_records
               (student_id, focus_score, distraction_count, head_pose, body_stability, fatigue_level, eye_focus, suggestion, model_prediction, expire_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '+30 days'))""",
            (student_id, data.get("focus_score"), data.get("distraction_count"),
             data.get("head_pose"), data.get("body_stability"), data.get("fatigue_level"),
             data.get("eye_focus"), data.get("suggestion"), data.get("model_prediction"))
        )

def normalize_record_times(record: dict) -> dict:
    """P1-7：把行为记录中的时间字段统一转成带 Z 的 ISO-8601，前端按本地时区正确渲染"""
    if not record:
        return record
    for key in ("created_at", "expire_at"):
        if key in record:
            record[key] = to_iso_utc(record.get(key))
    return record

def get_student_behavior_records(student_id: int, limit: int = 50):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM behavior_records WHERE student_id = ? ORDER BY created_at DESC LIMIT ?",
            (student_id, limit)
        )
        records = cursor.fetchall()
    return [normalize_record_times(dict(r)) for r in records]

def get_student_latest_record(student_id: int):
    """获取学生最新一条行为记录（依赖复合索引 idx_behavior_student_time，走索引定位无需回扫）"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM behavior_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
            (student_id,)
        )
        record = cursor.fetchone()
    return normalize_record_times(dict(record)) if record else None

def get_latest_records_for_students(student_ids):
    """一次性批量取出多个学生各自的最新一条行为记录（P2-4：消除教师端列表的 N+1 查询）

    原实现是在教师端学生列表的 for 循环里逐个调用 get_student_latest_record()，
    20 个学生就会触发 21 次数据库查询（N+1）。这里改为「一条 GROUP BY 聚合 + JOIN」：
    先用子查询按 student_id 分组取 MAX(id)（id 自增，最大值即该学生最新写入的记录），
    再 JOIN 回主表取回完整字段，无论多少个学生查库次数恒为 1 次。
    聚合子查询只涉及 (student_id, id)，可借助复合索引 idx_behavior_student_time 的
    student_id 有序前缀完成分组，避免全表排序。
    """
    if not student_ids:
        return {}
    id_list = [int(x) for x in student_ids]
    placeholders = ",".join("?" for _ in id_list)
    sql = (
        "SELECT b.* FROM behavior_records b "
        "JOIN ("
        "  SELECT student_id, MAX(id) AS latest_id FROM behavior_records "
        "  WHERE student_id IN (" + placeholders + ") GROUP BY student_id"
        ") m ON m.latest_id = b.id"
    )
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(id_list))
        rows = cursor.fetchall()
    latest_map = {}
    for r in rows:
        item = normalize_record_times(dict(r))
        latest_map[item.get("student_id")] = item
    return latest_map

# ========== API 路由 ==========

# --- 学生注册 ---
@app.post("/api/student/register")
async def student_register(request: StudentRegisterRequest):
    if len(request.username) < 3 or len(request.password) < 6:
        raise HTTPException(status_code=400, detail="用户名至少3位，密码至少6位")
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="请输入真实姓名")

    student_id = create_student(request.username, request.password, request.name, request.student_no, request.class_name)
    if student_id is None:
        raise HTTPException(status_code=400, detail="用户名已存在")

    return {"message": "注册成功", "student_id": student_id}

# --- 教师注册 ---
@app.post("/api/teacher/register")
async def teacher_register(request: LoginRequest):
    """教师注册"""
    if len(request.username) < 3 or len(request.password) < 6:
        raise HTTPException(status_code=400, detail="用户名至少3位，密码至少6位")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO teachers (username, password_hash) VALUES (?, ?)",
                (request.username, hash_password(request.password))
            )
            teacher_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="用户名已存在")
    return {"message": "注册成功", "teacher_id": teacher_id}

# --- 学生登录 ---
@app.post("/api/student/login")
async def student_login(request: LoginRequest):
    # P0-4：失败次数达到上限则直接拒绝（HTTP 429）
    _raise_if_login_locked("student", request.username)

    student = get_student_by_username(request.username)
    if not student or not verify_password(request.password, student["password_hash"]):
        _record_login_failure("student", request.username)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 登录成功：清零失败计数
    _clear_login_failures("student", request.username)

    # P0-4：历史 SHA256 哈希校验通过后，立即原地升级为 bcrypt
    if not is_bcrypt_hash(student["password_hash"]):
        upgrade_password_hash("student", student["id"], hash_password(request.password))

    token = create_jwt_token(student["id"], "student", request.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": request.username,
        "name": student["name"],
        "role": "student",
        "student_id": student["id"]
    }

# --- 教师登录 ---
@app.post("/api/teacher/login")
async def teacher_login(request: LoginRequest):
    # P0-4：失败次数达到上限则直接拒绝（HTTP 429）
    _raise_if_login_locked("teacher", request.username)

    teacher = get_teacher_by_username(request.username)
    if not teacher or not verify_password(request.password, teacher["password_hash"]):
        _record_login_failure("teacher", request.username)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 登录成功：清零失败计数
    _clear_login_failures("teacher", request.username)

    # P0-4：历史 SHA256 哈希校验通过后，立即原地升级为 bcrypt
    if not is_bcrypt_hash(teacher["password_hash"]):
        upgrade_password_hash("teacher", teacher["id"], hash_password(request.password))

    token = create_jwt_token(teacher["id"], "teacher", request.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": request.username,
        "role": "teacher"
    }

# --- 获取当前用户信息 ---
@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    role = user.get("role")
    if role == "student":
        student = get_student_by_username(user["sub"])
        if not student:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"username": student["username"], "name": student["name"], "role": "student", "student_id": student["id"]}
    elif role == "teacher":
        teacher = get_teacher_by_username(user["sub"])
        if not teacher:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"username": teacher["username"], "role": "teacher"}

# --- 教师获取所有学生列表 ---
@app.get("/api/teacher/students")
async def teacher_list_students(user: dict = Depends(get_current_user)):
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    students = get_all_students()
    # P2-4：先一次性批量取回所有学生的最新行为记录（查库次数恒为 1，彻底消除 N+1），
    # 再在内存里做归并，避免原实现「循环内逐个查库」在 20 个学生时触发 21 次查询。
    latest_map = get_latest_records_for_students([s["id"] for s in students])
    for s in students:
        # 附加最新行为记录（无记录则为 None，前端按空态渲染）
        s["latest_record"] = latest_map.get(s["id"])
        # 检查在线状态：视频帧缓存 TTL 为 10 秒，取得到即视为在线
        cache = get_student_video_cache(s["id"])
        if cache:
            s["online"] = True
            s["live_focus_score"] = cache["focus_score"]
        else:
            s["online"] = False
            s["live_focus_score"] = None
    return {"students": students}

# --- 教师查看某个学生的行为记录 ---
@app.get("/api/teacher/students/{student_id}/records")
async def teacher_view_records(student_id: int, user: dict = Depends(get_current_user)):
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    records = get_student_behavior_records(student_id)
    return {"student": student, "records": records}

# --- 教师获取学生实时视频帧 ---
@app.get("/api/teacher/students/{student_id}/live")
async def teacher_get_student_live(student_id: int, user: dict = Depends(get_current_user)):
    """教师获取学生的实时摄像头画面"""
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    
    cache = get_student_video_cache(student_id)
    if not cache:
        return {
            "student_id": student_id,
            "online": False,
            "message": "学生未在线或未开启监控"
        }
    
    # P2-3：视频帧缓存 TTL=10s，取到即说明最近 10 秒内有更新；此处再按时间戳二次校验
    is_online = (time.time() - cache.get("timestamp", 0)) < 10
    
    return {
        "student_id": student_id,
        "online": is_online,
        # P2-6：不再内联 base64 大图（单帧可达数百 KB，2 秒轮询会让教师端带宽爆掉），
        # 只回状态与分数；画面由 GET /api/teacher/students/{id}/snapshot.jpg 提供。
        "has_image": bool(cache.get("image")),
        "focus_score": cache["focus_score"],
        "timestamp": cache["timestamp"]
    }

# --- 教师拉取学生最新画面快照（P2-6）---
@app.get("/api/teacher/students/{student_id}/snapshot.jpg")
async def teacher_get_student_snapshot(student_id: int, user: dict = Depends(get_current_user)):
    """教师端拉取学生最新一帧画面（二进制 JPEG）

    背景：/ws 不再下行整帧 base64 图像，原始画面只留在服务端缓存里，
    教师端需要看图时走这个受鉴权保护的二进制接口，避免 base64 膨胀 33% 与 JSON 解析开销。
    无画面、缓存过期、解码失败统一返回 404，不向外暴露任何内部异常信息。
    """
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    cache = get_student_video_cache(student_id)
    if not cache:
        raise HTTPException(status_code=404, detail="暂无画面")
    image_b64 = cache.get("image") or ""
    if not image_b64:
        raise HTTPException(status_code=404, detail="暂无画面")
    try:
        raw = base64.b64decode(image_b64)
    except Exception:
        # 内部解码异常只进日志，对外统一 404
        logger.exception("学生 %s 画面快照解码失败", student_id)
        raise HTTPException(status_code=404, detail="暂无画面")
    if not raw:
        raise HTTPException(status_code=404, detail="暂无画面")
    # 画面实时变化，禁止浏览器/中间层缓存，否则教师端会看到旧帧
    return Response(content=raw, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

# --- 学生保存自己的行为数据 ---
@app.post("/api/student/behavior")
async def student_save_behavior(data: dict, user: dict = Depends(get_current_user)):
    if user.get("role") != "student":
        raise HTTPException(status_code=403, detail="仅学生可访问")
    student_id = user.get("user_id")
    save_behavior_record(student_id, data)
    return {"message": "行为数据已保存"}

# ========== 火山引擎语音识别封装 ==========
class VolcEngineASR:
    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key
        self.token_url = VOLC_TOKEN_URL
        self.asr_url = VOLC_ASR_ENDPOINT

    def get_token(self):
        try:
            resp = requests.post(self.token_url, data={
                "grant_type": "client_credentials",
                "client_id": self.access_key,
                "client_secret": self.secret_key
            }, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("access_token", "")
        except Exception:
            logger.warning("获取火山引擎 Token 失败", exc_info=True)
        return ""

    def recognize_audio(self, audio_bytes, sample_rate=16000):
        if not self.access_key or self.access_key == "YOUR_ACCESS_KEY":
            return None
        token = self.get_token()
        if not token:
            return None
        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer;{token}"}
            payload = {
                "app": {"appid": "default", "cluster": "volcengine_streaming_common"},
                "user": {"uid": "ai_study_buddy"},
                "audio": {"format": "wav", "rate": sample_rate, "language": "zh-CN", "bits": 16, "channel": 1, "codec": "raw"},
                "request": {"reqid": f"req_{int(time.time())}", "sequence": 1, "nbest": 1, "text": ""},
                "audio_data": base64.b64encode(audio_bytes).decode('utf-8')
            }
            resp = requests.post(self.asr_url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                return result.get("result", {}).get("text", "")
        except Exception:
            logger.warning("语音识别失败", exc_info=True)
        return None

asr_engine = VolcEngineASR(VOLC_ACCESS_KEY, VOLC_SECRET_KEY)

# ========== MediaPipe 姿态分析器（集成训练模型） ==========
class MediaPipePoseAnalyzer:
    def __init__(self):
        self.consecutive_low_focus = 0
        self.total_distractions = 0
        self.last_analysis_time = time.time()
        self.prev_landmarks = None
        self.head_tilt_history = []
        self.shoulder_history = []
        self.model_loaded = False
        self.model_n_features = 0

        self.focus_score_history = []
        self.last_valid_score = 80
        # 动作分类历史（用于平滑）
        self.action_history = []
        # 闭眼/疲劳持续时间计数（避免眨眼误检）
        self.eyes_closed_frames = 0
        self.eyes_closed_threshold = EYE_CLOSED_FRAMES   # P1-8：接入配置项（默认 60 帧 ≈ 2 秒 @30fps）
        self.confirmed_eyes_closed = False

        self.check_model_status()

    def check_model_status(self):
        """检查模型是否已训练"""
        try:
            model = load_trained_model()
            self.model_loaded = model is not None
            if self.model_loaded:
                self.model_n_features = int(getattr(model, 'n_features_in_', FEATURE_DIM))
                logger.info("专注度模型已加载（特征维度=%s）", self.model_n_features)
            else:
                logger.info("专注度预测模型未启用，当前使用纯规则引擎分析")
        except Exception as e:
            logger.warning("模型检查失败", exc_info=True)
            self.model_loaded = False

    def _extract_model_features(self, landmarks):
        """P1-2：51 维口径已废弃，统一委托给 13 维契约实现 extract_model_features"""
        return extract_model_features(landmarks)

    def analyze(self, image_base64, scene="selfstudy", student_id=None):
        try:
            img_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return self._fallback_analysis(scene, image_base64)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # P2-1：从 threading.local 取当前工作线程独占的实例，避免多学生并发串台
            results = _get_pose().process(rgb_frame)
            face_results = _get_face_mesh().process(rgb_frame)
            if not results.pose_landmarks:
                return self._fallback_analysis(scene, image_base64)
            landmarks = results.pose_landmarks.landmark

            # 绘制骨架
            annotated = self._draw_skeleton(frame, landmarks)
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')

            # 提取关键点
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            # 计算传统姿态特征
            head_tilt = self._calc_head_tilt(nose, left_shoulder, right_shoulder)
            head_drop = self._calc_head_drop(nose, left_shoulder, right_shoulder)
            head_pose_status = self._evaluate_head_pose(head_tilt, head_drop)
            shoulder_level = abs(left_shoulder.y - right_shoulder.y)
            body_sway = self._calc_body_sway(left_shoulder, right_shoulder, left_hip, right_hip)
            body_stability_status = self._evaluate_body_stability(shoulder_level, body_sway)
            # ===== 眼神/疲劳：用 Face Mesh 虹膜 + EAR（疲劳优先）=====
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0].landmark
                ear_value = self._calc_ear(face_lm)
                fatigue_status = self._evaluate_fatigue(ear_value)
                # 闭眼持续时间检测：避免眨眼误检
                is_eyes_closed_now = (fatigue_status == "明显疲劳")
                if is_eyes_closed_now:
                    self.eyes_closed_frames += 1
                else:
                    self.eyes_closed_frames = max(0, self.eyes_closed_frames - POSE_THRESHOLDS["eyes_closed_decay"])
                # 只有持续闭眼超过阈值才确认闭眼
                if self.eyes_closed_frames >= self.eyes_closed_threshold:
                    self.confirmed_eyes_closed = True
                elif self.eyes_closed_frames == 0:
                    self.confirmed_eyes_closed = False
                # 第1优先级：确认闭眼→闭眼/走神，否则用疲劳状态
                if self.confirmed_eyes_closed:
                    eye_focus_status = "闭眼/走神"
                elif fatigue_status == "明显疲劳":
                    eye_focus_status = "视线模糊"  # 疲劳但未持续闭眼
                elif fatigue_status == "轻度疲劳":
                    eye_focus_status = "视线模糊"
                else:
                    # 第2优先级：眼睛睁开时，用虹膜偏移判断视线
                    iris_offset = self._calc_iris_offset(face_lm)
                    if iris_offset > POSE_THRESHOLDS["iris_offset_major"]:
                        eye_focus_status = "频繁偏离"
                    elif iris_offset > POSE_THRESHOLDS["iris_offset_minor"]:
                        eye_focus_status = "偶尔偏离"
                    else:
                        eye_focus_status = "注视屏幕"
            else:
                # P3-2：Face Mesh 未检测到人脸（短暂转头 / 眼镜反光 / 完全闭眼等）
                # 历史实现对这种情况赋「轻度疲劳 + 视线模糊」，导致固定扣 26 分并被误判为分心。
                # 现改为中性标记「未检测」，疲劳与眼神两项均不参与扣分，由姿态类指标独立判定。
                fatigue_status = "未检测"
                eye_focus_status = "未检测"

            # 姿态稳定性（辅助指标）
            pose_stability = self._calculate_pose_stability(head_tilt, shoulder_level)

            # ========== P1-4：透明双轨输出（规则分独立可解释，模型结论单独呈现） ==========
            # 规则分永远是最终 focus_score 的唯一来源，模型只作为辅助轨道输出，不再做任何权重混合。
            rule_score = self._calculate_focus_score(head_pose_status, body_stability_status, fatigue_status, eye_focus_status, scene, pose_stability)
            focus_score = rule_score

            model_meta = None          # 高置信度时才填充，供接口返回
            model_prediction = None    # 兼容旧字段：模型原始结果字典
            score_source = "rule"
            if self.model_loaded:
                features = self._extract_model_features(landmarks)
                if features is not None:
                    model_prediction, model_conf = predict_focus(features)
                    if model_prediction is not None and model_conf >= MODEL_MIN_CONFIDENCE:
                        model_meta = model_prediction
                        score_source = "rule+model"
                    elif model_prediction is not None:
                        logger.debug("模型置信度 %.2f 低于门限 %.2f，本次仅采用规则结论", model_conf, MODEL_MIN_CONFIDENCE)
            logger.debug(
            "评分明细 规则分=%s 最终=%s 来源=%s 姿态=%s 眼神=%s 疲劳=%s",
            rule_score, focus_score, score_source, head_pose_status, eye_focus_status, fatigue_status,
        )

            # ========== 时序平滑 ==========
            focus_score = self._smooth_focus_score(focus_score)

            # ========== 保存有效分数（供fallback使用）==========
            self.last_valid_score = focus_score

            is_distracted = focus_score < POSE_THRESHOLDS["distraction_score_threshold"]
            if is_distracted:
                self.consecutive_low_focus += 1
                if self.consecutive_low_focus >= POSE_THRESHOLDS.get("consecutive_low_focus", 3):
                    self.total_distractions += 1
                    self.consecutive_low_focus = 0
                    # 自动发送语音提醒到学生端
                    try:
                        reminder_msg = self._generate_suggestion(focus_score, head_pose_status, fatigue_status, scene)
                        # P2-3：TTLCache 内部带锁且带容量上限，工作线程写入安全
                        student_notifications.set(student_id, {
                            "student_id": student_id,
                            "message": reminder_msg,
                            "timestamp": time.time(),
                            "type": "voice"
                        })
                    except Exception:
                        logger.exception("发送提醒失败（student_id=%s）", student_id)
            else:
                self.consecutive_low_focus = max(0, self.consecutive_low_focus - 1)

            suggestion = self._generate_suggestion(focus_score, head_pose_status, fatigue_status, scene)

            # 动作小分类检测
            raw_action = self._classify_action(landmarks, head_pose_status, eye_focus_status, fatigue_status, body_stability_status)
            action_label = self._smooth_action(raw_action)
            action_cn = {"focused": "专注", "sleep": "睡觉", "playphone": "玩手机", "distracted": "分心"}.get(action_label, action_label)
            if action_label != "focused":
                logger.debug("动作分类: %s (%s)", action_cn, action_label)

            # P1-4：构建模型分轨信息（仅高置信度时才有内容，便于前端透明展示）
            model_info = ""
            if model_meta:
                model_info = (f"模型预测: {model_meta.get('label', '未知')} "
                              f"(专注概率: {model_meta.get('prob_focused', 0.0):.0%}, 置信度: {model_meta.get('confidence', 0.0):.0%})")

            return {
                "focus_score": focus_score,
                "action_label": action_label,
                "distraction_count": self.total_distractions,
                "head_pose": head_pose_status,
                "body_stability": body_stability_status,
                "fatigue_level": fatigue_status,
                "eye_focus": eye_focus_status,
                "suggestion": suggestion,
                "pose_detected": True,
                "annotated_image": annotated_b64,
                "model_prediction": model_info,
                "model_enabled": self.model_loaded,
                # P1-4：透明双轨字段
                "model_available": model_meta is not None,
                "model": model_meta,
                "score_source": score_source,
                "rule_score": rule_score,
                "pose_stability": round(pose_stability, 2)
            }
        except Exception:
            # P2-5：完整堆栈只进日志；对外仍返回可用的兜底结果，不携带任何异常文本
            logger.exception("MediaPipe 分析异常（student_id=%s）", student_id)
            return self._fallback_analysis(scene, image_base64)

    def _draw_skeleton(self, frame, landmarks):
        """绘制人体骨架"""
        annotated = frame.copy()
        h, w, _ = annotated.shape
        SKELETON_CONNECTIONS = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
            (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
            (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
            (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EAR),
            (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
            (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
            (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
            (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_EAR),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_EAR),
        ]
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            sp, ep = landmarks[start_idx], landmarks[end_idx]
            cv2.line(annotated, (int(sp.x*w), int(sp.y*h)), (int(ep.x*w), int(ep.y*h)), (34, 211, 238), 2, cv2.LINE_AA)
        KEY_POINTS = [
            mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
            mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        for kp_idx in KEY_POINTS:
            kp = landmarks[kp_idx]
            cx, cy = int(kp.x*w), int(kp.y*h)
            cv2.circle(annotated, (cx, cy), 8, (99, 102, 241), -1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 4, (22, 211, 238), -1, cv2.LINE_AA)
        return annotated

    def _calc_head_tilt(self, n, ls, rs):
        """计算头部水平偏移：鼻子相对肩膀中心的水平偏差（乘坐标比例系数转为百分制）"""
        shoulder_center_x = (ls.x + rs.x) / 2
        return abs(n.x - shoulder_center_x) * POSE_THRESHOLDS["coord_scale"]

    def _calc_head_drop(self, n, ls, rs):
        sw = abs(ls.x - rs.x)
        return ((ls.y + rs.y) / 2 - n.y) / sw if sw > POSE_THRESHOLDS["shoulder_width_eps"] else 0

    def _evaluate_head_pose(self, tilt, drop):
        """头部姿态分级（阈值全部取自 POSE_THRESHOLDS）"""
        th = POSE_THRESHOLDS
        if tilt > th["head_tilt_major"]: return "明显偏移"
        elif tilt > th["head_tilt_minor"]: return "轻微偏移"
        elif drop < th["head_drop_major"]: return "明显低头"
        elif drop < th["head_drop_minor"]: return "轻微低头"
        else: return "端正"

    def _calc_body_sway(self, ls, rs, lh, rh):
        return abs((ls.x + rs.x) / 2 - (lh.x + rh.x) / 2) * POSE_THRESHOLDS["coord_scale"]

    def _evaluate_body_stability(self, sl, sway):
        th = POSE_THRESHOLDS
        if sl > th["shoulder_level_major"] or sway > th["body_sway_major"]: return "明显晃动"
        elif sl > th["shoulder_level_minor"] or sway > th["body_sway_minor"]: return "轻微晃动"
        else: return "稳定"

    def _calc_ear(self, face_landmarks):
        """基于 Face Mesh 关键点计算 EAR（眼纵横比），闭眼时 EAR 接近 0"""
        ear_default = POSE_THRESHOLDS["ear_default"]
        if face_landmarks is None:
            return ear_default
        def _ear(p1, p2, p3, p4, p5, p6):
            v1 = math.hypot(p2.x - p6.x, p2.y - p6.y)
            v2 = math.hypot(p3.x - p5.x, p3.y - p5.y)
            h = math.hypot(p1.x - p4.x, p1.y - p4.y)
            if h < 1e-6: return ear_default
            return (v1 + v2) / (2.0 * h)
        left = _ear(
            face_landmarks[33], face_landmarks[160], face_landmarks[158],
            face_landmarks[133], face_landmarks[153], face_landmarks[144])
        right = _ear(
            face_landmarks[362], face_landmarks[385], face_landmarks[387],
            face_landmarks[263], face_landmarks[373], face_landmarks[380])
        return (left + right) / 2.0

    def _evaluate_fatigue(self, ear):
        """基于 EAR 判断疲劳等级：低于 ear_fatigue_major 明显疲劳，低于 ear_fatigue_minor 轻度疲劳"""
        if ear is None or ear <= 0: return "未知"
        th = POSE_THRESHOLDS
        if ear < th["ear_fatigue_major"]: return "明显疲劳"
        if ear < th["ear_fatigue_minor"]: return "轻度疲劳"
        return "精力充沛"

    def _calc_iris_offset(self, face_landmarks):
        """虹膜中心相对眼角水平中心的偏移比例（0 正对屏幕）。返回左右眼平均值 0-1"""
        if face_landmarks is None:
            return 0.5
        def _one(inner_idx, outer_idx, iris_idx):
            inner = face_landmarks[inner_idx]
            outer = face_landmarks[outer_idx]
            iris = face_landmarks[iris_idx]
            eye_w = abs(outer.x - inner.x)
            if eye_w < 1e-6: return 0.0
            return abs(iris.x - (inner.x + outer.x) / 2) / eye_w
        left = _one(33, 133, 468)
        right = _one(362, 263, 473)
        return (left + right) / 2.0

    # ========== 姿态稳定性评估 ==========
    def _calculate_pose_stability(self, head_tilt, shoulder_level):
        """计算姿态稳定性（窗口与归一系数取自 POSE_STABILITY）"""
        ps = POSE_STABILITY
        self.head_tilt_history.append(head_tilt)
        self.shoulder_history.append(shoulder_level)

        if len(self.head_tilt_history) > ps["history_size"]:
            self.head_tilt_history.pop(0)
        if len(self.shoulder_history) > ps["history_size"]:
            self.shoulder_history.pop(0)

        if len(self.head_tilt_history) < ps["min_history"]:
            return 1.0

        tilt_variance = np.var(self.head_tilt_history)
        shoulder_variance = np.var(self.shoulder_history)
        stability = max(0, 1 - (tilt_variance / ps["tilt_var_scale"] + shoulder_variance * ps["shoulder_var_scale"]))
        return stability

    # ========== 时序平滑 ==========
    def _smooth_focus_score(self, raw_score):
        """对专注度分数进行指数平滑，减少短期波动（参数取自 SCORE_SMOOTHING）"""
        sm = SCORE_SMOOTHING
        self.focus_score_history.append(raw_score)
        if len(self.focus_score_history) > sm["history_size"]:
            self.focus_score_history.pop(0)
        if len(self.focus_score_history) < sm["min_history"]:
            return raw_score
        # 指数衰减权重，让近期分数权重更高
        weights = [sm["decay"] ** (len(self.focus_score_history) - 1 - i) for i in range(len(self.focus_score_history))]
        weights = [w / sum(weights) for w in weights]
        smoothed = sum(s * w for s, w in zip(self.focus_score_history, weights))
        # 增加惯性：如果新分数与历史均值差距很小，更倾向于保持当前趋势
        if len(self.focus_score_history) >= sm["inertia_window"]:
            recent_avg = sum(self.focus_score_history[-sm["inertia_window"]:]) / sm["inertia_window"]
            if abs(smoothed - recent_avg) < sm["inertia_gap"]:
                smoothed = sm["inertia_blend"] * smoothed + (1 - sm["inertia_blend"]) * recent_avg
        return round(smoothed)

    # ========== 动作小分类检测 ==========
    def _classify_action(self, landmarks, head_pose_status, eye_focus_status, fatigue_status, body_stability_status, ear_value=0.3):
        """
        检测具体动作类型（仅在 distracted 状态下分类）
        分类：focused / sleep / playphone / distracted
        基于姿态关键点规则判断：
        - sleep: 头部严重下垂 + 眼睛闭合(EAR低) + 身体不动
        - playphone: 双手腕在胸前/腹部区域 + 肘部弯曲 + 头部略微低头
        - distracted: 其他分心行为
        """
        if head_pose_status == "端正" and eye_focus_status == "注视屏幕" and fatigue_status == "精力充沛":
            return "focused"
        # P3-2 配套：人脸未检出时眼神/疲劳为中性的「未检测」，此时不应据此判为分心；
        # 只要头姿端正且身体稳定，即按专注处理，避免短暂转头触发误判。
        if eye_focus_status == "未检测" and head_pose_status == "端正" and body_stability_status == "稳定":
            return "focused"

        # 获取关键点
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        torso_mid_y = (shoulder_center_y + hip_center_y) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        at = ACTION_THRESHOLDS

        # ===== 睡觉检测 =====
        # 头部严重下垂（鼻子y坐标远低于肩膀）+ 眼睛闭合 + 身体稳定
        head_drop_ratio = nose.y - shoulder_center_y
        is_eyes_closed = fatigue_status == "明显疲劳" or eye_focus_status == "闭眼/走神"
        is_body_still = body_stability_status == "稳定"

        if head_drop_ratio > at["sleep_head_drop_ratio"] and is_eyes_closed and is_body_still:
            return "sleep"

        # 备选：头部略微下垂 + 眼睛闭合 + 长期不动也判定为睡觉
        if head_drop_ratio > at["sleep_head_drop_ratio_loose"] and is_eyes_closed and is_body_still and fatigue_status == "明显疲劳":
            return "sleep"

        # ===== 玩手机检测 =====
        # 手腕在躯干前方区域（允许在肩膀上方，因为举手机时手会抬高）
        left_wrist_in_torso = (left_wrist.y > shoulder_center_y - at["wrist_torso_above_shoulder"] and
                               left_wrist.y < hip_center_y + at["wrist_torso_below_hip"])
        right_wrist_in_torso = (right_wrist.y > shoulder_center_y - at["wrist_torso_above_shoulder"] and
                                right_wrist.y < hip_center_y + at["wrist_torso_below_hip"])

        # 手腕靠近身体中心线（水平距离小于肩宽的给定比例，更宽松）
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        left_wrist_near_center = abs(left_wrist.x - shoulder_center_x) < shoulder_width * at["wrist_center_ratio"]
        right_wrist_near_center = abs(right_wrist.x - shoulder_center_x) < shoulder_width * at["wrist_center_ratio"]

        # 肘部弯曲检测：手臂收起时，手腕与肘部的y坐标差距不大（无论上下）
        # 或者手腕在肘部上方（举着手机）
        left_elbow_bent = abs(left_wrist.y - left_elbow.y) < at["elbow_bend_dy"] or left_wrist.y < left_elbow.y
        right_elbow_bent = abs(right_wrist.y - right_elbow.y) < at["elbow_bend_dy"] or right_wrist.y < right_elbow.y

        # 综合判断：至少一只手在胸前区域 + 肘部弯曲
        # 玩手机的典型姿势：手在胸前/脸前，手臂弯曲收起
        left_hand_valid = left_wrist_in_torso and left_wrist_near_center
        right_hand_valid = right_wrist_in_torso and right_wrist_near_center
        any_hand_in_torso = left_hand_valid or right_hand_valid
        any_elbow_bent = left_elbow_bent or right_elbow_bent

        # 调试输出（帮助诊断）
        debug_info = f"[DEBUG] hand_L={left_hand_valid} hand_R={right_hand_valid} elbow_L={left_elbow_bent} elbow_R={right_elbow_bent} lw_y={left_wrist.y:.3f} le_y={left_elbow.y:.3f}"
        if any_hand_in_torso and any_elbow_bent:
            logger.debug("%s -> playphone", debug_info)
            return "playphone"
        elif any_hand_in_torso or any_elbow_bent:
            logger.debug("%s -> distracted (partial match)", debug_info)

        # ===== 默认：普通分心 =====
        return "distracted"

    def _smooth_action(self, action):
        """对动作分类进行时序平滑，避免频繁跳变"""
        at = ACTION_THRESHOLDS
        self.action_history.append(action)
        if len(self.action_history) > at["action_history_size"]:
            self.action_history.pop(0)
        if len(self.action_history) < at["action_vote_min"]:
            return action
        # 取最近若干帧的众数
        recent = self.action_history[-at["action_vote_window"]:]
        most_common = Counter(recent).most_common(1)[0]
        return most_common[0]

    # ========== 评分算法 ==========
    def _calculate_focus_score(self, hp, bs, fat, ef, scene, stability=1.0):
        """规则引擎评分（可解释、可单测）：所有扣分/加分均取自 SCORE_PENALTIES

        P1-8 / P3-2：数值与重构前硬编码字面量完全一致，行为零漂移；
        「未检测」类中性状态（人脸未检出的疲劳/眼神）不参与扣分。
        """
        p = SCORE_PENALTIES
        s = 100
        if hp == "明显偏移": s -= p["head_pose_major"]
        elif hp == "轻微偏移": s -= p["head_pose_minor"]
        elif hp == "明显低头": s -= p["head_down_major"]
        elif hp == "轻微低头": s -= p["head_down_minor"]
        if bs == "明显晃动": s -= p["body_major"]
        elif bs == "轻微晃动": s -= p["body_minor"]
        if fat == "明显疲劳": s -= p["fatigue_major"]
        elif fat == "轻度疲劳": s -= p["fatigue_minor"]
        # 眼神状态：闭眼/走神最严重，视线模糊次之，偶尔偏离（正常眨眼）扣分很少
        if ef == "闭眼/走神": s -= p["eye_closed"]
        elif ef == "视线模糊": s -= p["eye_blur"]
        elif ef == "频繁偏离": s -= p["eye_deviation_frequent"]
        elif ef == "偶尔偏离": s -= p["eye_deviation_occasional"]
        # 「未检测」不命中任何分支 → 不加不减
        if stability > p["stability_high"]: s += p["stability_bonus"]
        elif stability < p["stability_low"]: s -= p["stability_penalty"]
        if scene == "onetoone": s = min(s + p["scene_onetoone_bonus"], p["score_max"])
        elif scene == "online": s = max(s - p["scene_online_penalty"], p["score_min"])
        return max(p["score_min"], min(p["score_max"], s))

    def _generate_suggestion(self, fs, hp, fat, scene):
        # P1-8：分档阈值统一取 WARNING_LEVELS（excellent / mild / moderate），低于 moderate 视为严重分心
        lv = WARNING_LEVELS
        if fs >= lv["excellent"]: return random.choice(["✅ 专注度极佳，保持当前学习节奏！", "✅ 学习状态非常好，继续保持！", "✅ 你的专注力令人印象深刻！"])
        elif fs >= lv["mild"]:
            if "偏移" in hp: return "💡 注意调整坐姿，保持头部端正有助于集中注意力"
            if "低头" in hp: return "💡 检测到轻微低头，适当调整坐姿，保持平视"
            if fat == "轻度疲劳": return "💡 检测到轻微疲劳，建议调整呼吸，做几次深呼吸"
            return "💡 状态良好，建议适时调整坐姿，预防疲劳"
        elif fs >= lv["moderate"]:
            if fat == "明显疲劳": return "⚠️ 检测到疲劳信号，建议闭眼休息30秒再继续"
            if "偏移" in hp: return "⚠️ 头部偏移较大，请调整坐姿，保持正视屏幕"
            if "低头" in hp: return "⚠️ 低头时间较长，建议抬抬头活动颈椎"
            return "⚠️ 专注度下降，建议起身活动2分钟，喝口水"
        else: return random.choice(["🚨 严重分心！建议暂停学习，休息5分钟后再继续", "🚨 注意力严重分散！请先放松一下，调整好状态再学习", "🚨 检测到多项异常指标，强烈建议休息调整"])

    def _fallback_analysis(self, scene, image_base64=None):
        """无人脸时：不返回虚假专注度，携带原始画面以便前端显示摄像头"""
        return {
            "focus_score": None,
            "distraction_count": self.total_distractions,
            "head_pose": "未检测到", "body_stability": "未检测到",
            "fatigue_level": "未知", "eye_focus": "未知",
            "suggestion": "未检测到人体，请确保摄像头能拍摄到上半身",
            "pose_detected": False,
            "annotated_image": image_base64,
            "model_prediction": "", "model_enabled": self.model_loaded,
            # P1-4：与正常分支保持同一套双轨字段契约，便于前端统一处理
            "model_available": False, "model": None,
            "score_source": "rule", "rule_score": None,
            "pose_stability": 0.0,
        }


analyzer = MediaPipePoseAnalyzer()

# ========== 学生实时视频帧缓存（P2-3：TTL + 容量上限，防 OOM）==========
# 视频帧：10 秒无更新即视为离线自动过期；容量上限 200，超出按 LRU 淘汰
student_video_cache = TTLCache(maxsize=200, ttl=10)
# 语音提醒：学生端一次性消费，5 分钟未取自动过期，避免堆积
student_notifications = TTLCache(maxsize=1000, ttl=300)
# 在线学生状态：60 秒无心跳视为离线
online_students = TTLCache(maxsize=500, ttl=60)

def update_student_video_cache(student_id: int, annotated_image: str, focus_score: int,
                                 action_label="unknown", head_pose="", eye_focus="", fatigue_level="",
                                 name=""):
    """更新学生视频帧缓存（TTLCache 内部加锁，可被推理工作线程安全调用）"""
    student_video_cache.set(student_id, {
        "image": annotated_image,
        "timestamp": time.time(),
        "focus_score": focus_score,
        "action_label": action_label,
        "head_pose": head_pose,
        "eye_focus": eye_focus,
        "fatigue_level": fatigue_level,
        "name": name
    })

def get_student_video_cache(student_id: int):
    """获取学生视频帧缓存（过期或不存在返回 None）"""
    return student_video_cache.get(student_id)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    # ===== P0-1：握手阶段完成 Origin 校验与 JWT 鉴权，全部在 accept 之前 =====
    origin = websocket.headers.get("origin")
    if origin and "*" not in CORS_ALLOW_ORIGINS and origin not in CORS_ALLOW_ORIGINS:
        # 4403：来源不在白名单
        await websocket.close(code=4403)
        return

    token = websocket.query_params.get("token")
    if not token:
        # 4401：缺少令牌
        await websocket.close(code=4401)
        return

    try:
        payload = verify_jwt_token(token)
    except HTTPException:
        # 4401：令牌无效或已过期
        await websocket.close(code=4401)
        return

    if payload.get("role") != "student":
        # 4403：仅学生可建立分析连接
        await websocket.close(code=4403)
        return

    # 身份只取自令牌，后续任何客户端消息都无法改变当前学生身份
    student_id = payload.get("user_id")
    if not student_id:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    logger.info("客户端已连接（student_id=%s）", student_id)
    current_student_id = student_id

    # ===== P1-5：握手鉴权通过后只查一次数据库，取真实姓名缓存，帧循环内直接复用 =====
    student_name = "学生" + str(student_id)
    try:
        student_info = get_student_by_id(student_id)
        if student_info and student_info.get("name"):
            student_name = student_info["name"]
    except Exception:
        # 查询失败时降级为默认显示名，且不影响监控主流程；完整堆栈只进日志
        logger.warning("获取学生姓名失败（student_id=%s）", student_id, exc_info=True)

    # 若该学生此前已有在线记录，同步刷新姓名，避免继续显示旧的占位名
    cached_online = online_students.get(student_id)
    if cached_online:
        cached_online["name"] = student_name
        online_students.set(student_id, cached_online)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if "image" in msg:
                scene = msg.get("scene", "selfstudy")
                # P0-1：身份已由令牌绑定，忽略消息体中的 student_id

                # ===== P2-2：单学生背压——上一帧尚未算完则直接丢弃当前帧 =====
                # 学生端 1 秒 1 帧，正常情况推理远快于 1 秒；一旦某帧卡住，
                # 直接返回 skipped 让前端跳过而不排队，避免帧缓冲无限积压。
                if student_id in _inflight_students:
                    await websocket.send_json({"skipped": True, "reason": "上一帧仍在处理中"})
                    continue

                _inflight_students.add(student_id)
                try:
                    # ===== P2-2：推理移入有界线程池，不阻塞主事件循环 =====
                    # 同一学生同一时刻最多一个推理任务在跑，因此不会与其自身状态竞争；
                    # 不同学生会被调度到不同工作线程，各自持有 threading.local 的
                    # MediaPipe 实例（P2-1），互不串台。
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        _INFER_POOL, analyzer.analyze, msg["image"], scene, student_id
                    )
                finally:
                    # 无论推理成功、异常还是被取消，都必须释放背压标记，
                    # 否则该学生后续帧会被永久丢弃。
                    _inflight_students.discard(student_id)

                # 缓存学生视频帧（供教师实时查看）
                if student_id and result.get("annotated_image"):
                    update_student_video_cache(
                        student_id,
                        result["annotated_image"],
                        result.get("focus_score", 0),
                        action_label=result.get("action_label", "unknown"),
                        head_pose=result.get("head_pose", ""),
                        eye_focus=result.get("eye_focus", ""),
                        fatigue_level=result.get("fatigue_level", ""),
                        name=student_name
                    )
                    result["student_id"] = student_id
                    # 更新在线学生列表（TTLCache：60 秒无心跳自动过期）
                    online_students.set(student_id, {
                        "name": student_name,
                        "focus_score": result.get("focus_score", 0),
                        "status": result.get("action_label", "unknown"),
                        "last_seen": time.time()
                    })

                # ===== P2-6：下行瘦身——不再回传整帧 base64 大图 =====
                # 原始画面已存入 student_video_cache，教师端需要看图时走
                # GET /api/teacher/students/{id}/snapshot.jpg 拉二进制 JPEG。
                # 这里只保留分数/状态/警告等轻量字段，单条消息控制在 1KB 以内。
                result.pop("annotated_image", None)
                await websocket.send_json(result)
            elif msg.get("action") == "encouragement":
                logger.info("收到鼓励（student_id=%s）: %s", student_id, msg.get("message", ""))
            elif msg.get("action") == "asr_result":
                logger.info("语音识别（student_id=%s）: %s", student_id, msg.get("text", ""))
    except WebSocketDisconnect:
        # 正常断开（前端关闭页面/刷新），只记一条 info
        logger.info("客户端断开连接（student_id=%s）", student_id)
    except Exception:
        # P2-5：异常断开完整堆栈进日志，不向客户端回传异常文本
        logger.exception("WebSocket 会话异常（student_id=%s）", student_id)
    finally:
        # 断开时从在线列表移除，并清掉可能残留的背压标记
        if current_student_id:
            online_students.pop(current_student_id, None)
            _inflight_students.discard(current_student_id)


# ========== 前端页面（已抽离至 templates/index.html） ==========


# 调试接口（P0-3）：默认不注册；仅当环境变量 AI_STUDY_DEBUG_API == "1" 时注册，且仅教师可访问
if os.getenv("AI_STUDY_DEBUG_API") == "1":
    @app.get("/api/debug/cache")
    async def debug_cache(user: dict = Depends(get_current_user)):
        """调试接口：查看所有学生视频缓存（需教师权限）"""
        if user.get("role") != "teacher":
            raise HTTPException(status_code=403, detail="仅教师可访问")
        result = {}
        for sid, cache in student_video_cache.items():
            result[sid] = {
                "has_image": cache.get("image") is not None,
                "image_length": len(cache.get("image", "")),
                "timestamp": cache.get("timestamp"),
                "name": cache.get("name"),
                "focus_score": cache.get("focus_score")
            }
        return {
            "total_students": len(student_video_cache),
            "caches": result
        }


@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))
@app.get("/api/student/notifications")
async def get_student_notifications(user: dict = Depends(get_current_user)):
    """学生端：获取自己的提醒（P0-2：身份只取自令牌，不再接受 student_id 参数）"""
    if user.get("role") != "student":
        raise HTTPException(status_code=403, detail="仅学生可访问")
    student_id = user.get("user_id")
    # 原子读取并删除，避免并发重复提醒与 KeyError
    notification = student_notifications.pop(student_id, None)
    if notification:
        return {"success": True, "notification": notification}
    return {"success": True, "notification": None}


# ========== P0-10：学生本人数据删除接口（未成年人隐私合规最小集）==========
@app.delete("/api/student/my-data")
async def delete_my_student_data(user: dict = Depends(get_current_user)):
    """学生端：删除本人全部行为记录与内存缓存

    - 身份只取自 JWT 令牌，不接受任何外部传入的 student_id（避免越权删除他人数据）
    - 非 student 角色直接返回 403
    """
    if user.get("role") != "student":
        raise HTTPException(status_code=403, detail="仅学生可删除本人数据")

    student_id = user.get("user_id")

    # 1) 删除数据库中的全部行为记录（with 自动 commit / rollback / close）
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM behavior_records WHERE student_id = ?", (student_id,))
        deleted_records = cursor.rowcount or 0

    # 2) 清理内存缓存：实时画面、在线状态、待推送提醒
    student_video_cache.pop(student_id, None)
    online_students.pop(student_id, None)
    student_notifications.pop(student_id, None)

    logger.info("学生 %s 已删除本人数据，共 %s 条行为记录", student_id, deleted_records)
    return {
        "success": True,
        "deleted_records": deleted_records,
        "message": "本人行为记录与实时缓存已删除"
    }

if __name__ == "__main__":
    # P1-6：建表、迁移与模型加载已统一由 lifespan 负责，此处不再重复调用 init_db()
    # P1-8：启动主机与端口改为读取 config 中的 HOST / PORT（此前为硬编码，配置项形同虚设）
    logger.info("=" * 50)
    logger.info("AI学伴 · 专注度智能监护系统")
    logger.info("访问地址: http://%s:%s", HOST, PORT)
    logger.info("学生: 注册后登录 → 进入专注度监控")
    logger.info("=" * 50)
    uvicorn.run(app, host=HOST, port=PORT)
