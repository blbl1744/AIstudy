"""
配置层（依赖层级最底层，不 import 本包内任何其他模块）

职责：
1. 定义 BASE_DIR（指向项目根目录）与日志目录；
2. 初始化结构化日志 logger（固定名称 aistudy）；
3. 按绝对路径加载项目根目录的用户自建 config.py，缺失项回落到与历史完全一致的默认值；
4. 阈值/扣分/动作判定等业务常量；
5. 安全配置加固（JWT 密钥、CORS 白名单）。
"""
import copy
import importlib.util
import logging
import logging.handlers
import os
import secrets
import sys

# ========== 路径基准 ==========
# 注意：本文件位于 app/ 包内，BASE_DIR 必须上跳一级指向项目根目录，
# 否则数据库、模型、日志与前端模板的相对路径会全部落到 app/ 下。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# ========== 根目录 config.py 加载（避免与 app/config.py 同名遮蔽） ==========
# 说明：项目根目录存在用户自建的 config.py（存放密钥等不入库配置），本文件同在 app 包内，
# 若直接 `import config`，不同启动方式（python app/main.py / uvicorn app.main:app）会解析到不同文件。
# 因此这里按绝对路径显式加载根目录 config.py；任何一项缺失或加载失败都回落到历史默认值，
# 保证「接入配置项前后判定阈值零漂移」。
_DEFAULT_CONFIG = {
    "VOLC_ACCESS_KEY": "YOUR_ACCESS_KEY",
    "VOLC_SECRET_KEY": "YOUR_SECRET_KEY",
    "VOLC_TOKEN_URL": "https://open.volcengineapi.com/api/v3/auth/token",
    "VOLC_ASR_ENDPOINT": "https://openspeech.bytedance.com/api/v1/auc/recognize",
    "HOST": "0.0.0.0",
    "PORT": 8001,
    # ===== P1-8：以下默认值已逐项对齐「当前业务代码实际生效的运行值」=====
    "HEAD_DOWN_THRESHOLD": 0.15,        # 明显低头阈值（头部下沉比，越小越严格）
    "BODY_SWAY_LIMIT": 15,              # 身体摆动上限（×100 百分比刻度）
    "EYE_CLOSED_FRAMES": 60,            # 确认闭眼所需连续帧数（约 2 秒 @30fps）
    "DISTRACTION_INTERVAL": 3,          # 连续低专注帧数达到该值计一次分心
    "WARNING_LEVELS": {"excellent": 80, "mild": 60, "moderate": 40},
    "ASR_CONFIG": {"language": "zh-CN", "sample_rate": 16000, "silence_timeout": 120,
                   "enable_volcengine": True, "enable_browser_asr": True},
    "TTS_CONFIG": {"enabled": True, "language": "zh-CN", "rate": 0.9, "pitch": 1.1,
                   "volume": 0.8, "cooldown": 30},
    "POSE_CONFIG": {"static_image_mode": False, "model_complexity": 2,
                    "min_detection_confidence": 0.4, "min_tracking_confidence": 0.3},
    "POSE_THRESHOLDS": {
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
    },
    # P0-5：此处不再内置任何可用密钥，留空后交由下方「安全配置加固」段处理
    "JWT_SECRET_KEY": "",
    "JWT_ALGORITHM": "HS256",
    "JWT_EXPIRE_DAYS": 7,
}


def _load_root_config() -> dict:
    """按绝对路径加载项目根目录 config.py，返回其中与默认键同名的部分（失败返回空字典）"""
    path = os.path.join(BASE_DIR, "config.py")
    if not os.path.isfile(path):
        logger.info("未发现根目录 config.py，全部配置使用内置默认值")
        return {}
    try:
        spec = importlib.util.spec_from_file_location("aistudy_root_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        # 配置模块本身有语法/运行错误时不能让服务起不来，降级为默认值并留日志
        logger.warning("根目录 config.py 加载失败，全部配置回落默认值", exc_info=True)
        return {}
    return {key: getattr(module, key) for key in _DEFAULT_CONFIG if hasattr(module, key)}


def _cfg(key: str):
    """取配置值：根目录 config.py 优先，缺失则深拷贝默认值（防止默认字典被后续归一化改写）"""
    if key in _ROOT_CONFIG:
        return copy.deepcopy(_ROOT_CONFIG[key])
    return copy.deepcopy(_DEFAULT_CONFIG[key])


_ROOT_CONFIG = _load_root_config()

VOLC_ACCESS_KEY = _cfg("VOLC_ACCESS_KEY")
VOLC_SECRET_KEY = _cfg("VOLC_SECRET_KEY")
VOLC_TOKEN_URL = _cfg("VOLC_TOKEN_URL")
VOLC_ASR_ENDPOINT = _cfg("VOLC_ASR_ENDPOINT")
HOST = _cfg("HOST")
PORT = _cfg("PORT")
HEAD_DOWN_THRESHOLD = _cfg("HEAD_DOWN_THRESHOLD")
BODY_SWAY_LIMIT = _cfg("BODY_SWAY_LIMIT")
EYE_CLOSED_FRAMES = _cfg("EYE_CLOSED_FRAMES")
DISTRACTION_INTERVAL = _cfg("DISTRACTION_INTERVAL")
WARNING_LEVELS = _cfg("WARNING_LEVELS")
ASR_CONFIG = _cfg("ASR_CONFIG")
TTS_CONFIG = _cfg("TTS_CONFIG")
POSE_CONFIG = _cfg("POSE_CONFIG")
POSE_THRESHOLDS = _cfg("POSE_THRESHOLDS")
JWT_SECRET_KEY = _cfg("JWT_SECRET_KEY")
JWT_ALGORITHM = _cfg("JWT_ALGORITHM")
JWT_EXPIRE_DAYS = _cfg("JWT_EXPIRE_DAYS")


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
