"""
P3-3 测试夹具（conftest）

职责：
1. 在任何 app 模块被导入之前，向 sys.modules 注入 cv2 / mediapipe 轻量桩模块，
   使测试在未安装 OpenCV / MediaPipe（含其 C++ 运行库）的机器上也能直接运行；
2. 在导入 app 之前写死测试专用环境变量（JWT 密钥、CORS 白名单、数据库路径等），
   保证配置层读取到的是测试值而非生产默认值；
3. 提供 client 夹具：按用例动态隔离 SQLite 数据库、触发完整 lifespan，
   并在用例前后清理登录限速状态与三类内存缓存，避免用例互相污染；
4. 提供 fake_landmarks 夹具：生成符合 MediaPipe 33 点结构的模拟关键点。

说明：本文件只做测试环境搭建，不修改任何业务代码。
"""
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# ========== 0. 保证项目根目录可被 import（app 包所在目录） ==========
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ========== 1. 视觉依赖桩（必须在 import app 之前完成） ==========
# MediaPipe 官方 33 点顺序（下标即真实下标，仅用于桩模块的属性映射）
_POSE_LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]


def _build_cv2_stub() -> types.ModuleType:
    """构造 cv2 桩：所有图像解码/编码一律返回「解码失败」，绘制类调用全部空实现

    这样 analyzer.analyze 会走到「未检测到人体」的兜底分支（focus_score=None），
    与真实环境中传入非图片数据的行为完全一致。
    """
    stub = types.ModuleType("cv2")

    # 常量：数值与 OpenCV 官方保持一致，避免桩模块改变业务分支
    stub.IMREAD_COLOR = 1
    stub.COLOR_BGR2RGB = 4
    stub.IMWRITE_JPEG_QUALITY = 1
    stub.LINE_AA = 16
    stub.FILLED = -1

    def _imdecode(*_args, **_kwargs):
        return None

    def _imread(*_args, **_kwargs):
        return None

    def _imencode(*_args, **_kwargs):
        return False, b""

    def _cvtColor(frame, _code):
        return frame

    def _noop(*_args, **_kwargs):
        return None

    stub.imdecode = _imdecode
    stub.imread = _imread
    stub.imencode = _imencode
    stub.cvtColor = _cvtColor
    stub.circle = _noop
    stub.line = _noop
    stub.rectangle = _noop
    stub.putText = _noop
    stub.resize = _cvtColor
    return stub


def _build_mediapipe_stub() -> types.ModuleType:
    """构造 mediapipe 桩：提供 solutions.pose / face_mesh / drawing_utils 三个命名空间

    Pose.process / FaceMesh.process 恒定返回「未检测到关键点」的结果对象，
    使推理链路在无视觉库环境下仍可完整走通到兜底分支。
    """
    class Pose:  # noqa: D401 - 桩类，仅需可实例化
        def __init__(self, **_kwargs):
            pass

        def process(self, _frame):
            return SimpleNamespace(pose_landmarks=None)

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class FaceMesh:
        def __init__(self, **_kwargs):
            pass

        def process(self, _frame):
            return SimpleNamespace(multi_face_landmarks=None)

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class PoseLandmark:
        """33 点枚举的等价替身（属性值即关键点下标）"""

    for _idx, _name in enumerate(_POSE_LANDMARK_NAMES):
        setattr(PoseLandmark, _name, _idx)

    pose_ns = SimpleNamespace(Pose=Pose, PoseLandmark=PoseLandmark, POSE_CONNECTIONS=tuple())
    face_mesh_ns = SimpleNamespace(FaceMesh=FaceMesh)

    def _noop(*_args, **_kwargs):
        return None

    drawing_ns = SimpleNamespace(
        draw_landmarks=_noop,
        DrawingSpec=lambda *a, **k: None,
        POSE_CONNECTIONS=tuple(),
    )

    stub = types.ModuleType("mediapipe")
    stub.solutions = SimpleNamespace(
        pose=pose_ns,
        face_mesh=face_mesh_ns,
        drawing_utils=drawing_ns,
    )
    return stub


def _install_visual_stubs() -> None:
    """仅在真实依赖缺失时注入桩模块，装了真库的机器仍走真实实现"""
    for module_name, builder in (("cv2", _build_cv2_stub), ("mediapipe", _build_mediapipe_stub)):
        if module_name in sys.modules:
            continue
        try:
            __import__(module_name)
        except Exception:
            sys.modules[module_name] = builder()


_install_visual_stubs()


# ========== 2. 测试环境变量（必须在 import app 之前生效） ==========
# 引导库路径：使用进程级临时目录，保证「导入期」就有可用的数据库位置；
# 每个用例仍会通过 client 夹具把 db.DB_FILE 指向各自 tmp_path，实现用例级隔离。
_BOOTSTRAP_DIR = Path(tempfile.mkdtemp(prefix="aistudy_pytest_bootstrap_"))

os.environ["AI_STUDY_DB"] = str(_BOOTSTRAP_DIR / "bootstrap.db")
# JWT 密钥长度不低于 32 位，避免配置层判定为不安全密钥后随机生成（否则令牌无法跨模块校验）
os.environ["AI_STUDY_JWT_SECRET"] = "pytest-only-secret-key-0123456789abcdef0123456789"
# CORS 白名单：包含 TestClient 默认 Host 与真实前端来源，供 WebSocket Origin 校验使用
os.environ["AI_STUDY_CORS_ORIGINS"] = "http://localhost:8001,http://127.0.0.1:8001,http://testserver"
# 强制纯规则引擎模式：测试不依赖已训练的模型文件
os.environ["AI_STUDY_ENABLE_MODEL"] = "0"
# 降级日志级别，避免测试输出被 INFO 日志淹没
os.environ["AI_STUDY_LOG_LEVEL"] = "WARNING"
# 默认关闭调试接口（与生产默认一致），用于验证「默认调试接口不可达」
os.environ.pop("AI_STUDY_DEBUG_API", None)
# 不注入管理员账号，教师账号由用例自行注册
os.environ.pop("AI_STUDY_ADMIN_PASSWORD", None)


# ========== 3. 运行时状态清理工具 ==========
def _reset_runtime_state() -> None:
    """清理登录限速记录、四类内存缓存、在途推理标记与评分历史"""
    from app import auth as auth_module
    from app import state as state_module
    from app.analyzer import analyzer

    with auth_module._login_fail_lock:
        auth_module._login_fail.clear()

    for cache in (
        state_module.student_video_cache,
        state_module.online_students,
        state_module.student_notifications,
        # 扩展二：持续分心预警状态也必须逐用例清空，否则上个用例的预警会飘到下个用例
        state_module.student_alert_state,
    ):
        with cache._lock:
            cache._store.clear()

    state_module._inflight_students.clear()

    for attr in ("focus_score_history", "head_tilt_history", "shoulder_history", "action_history"):
        history = getattr(analyzer, attr, None)
        if isinstance(history, list):
            history.clear()
    analyzer.eyes_closed_frames = 0
    analyzer.confirmed_eyes_closed = False
    analyzer.consecutive_low_focus = 0
    analyzer.total_distractions = 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离数据库 + 完整 lifespan 的 TestClient

    隔离手段：
    1. 环境变量 AI_STUDY_DB 指向 tmp_path，供配置层/子进程读取；
    2. 直接改写 app.db.DB_FILE（模块已导入，环境变量不会二次生效），确保连接指向临时库；
    3. 为每个用例重建推理线程池：lifespan 退出时会 shutdown 全局池，复用会导致
       后续用例的 WebSocket 推理无法提交任务。
    """
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    import app.db as db_module
    import app.routers.student_routes as student_routes_module
    import app.state as state_module
    from app.main import app as application

    db_file = tmp_path / "test_ai_study.db"
    monkeypatch.setenv("AI_STUDY_DB", str(db_file))
    monkeypatch.setattr(db_module, "DB_FILE", str(db_file), raising=True)

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pytest-infer")
    monkeypatch.setattr(state_module, "_INFER_POOL", pool, raising=True)
    monkeypatch.setattr(student_routes_module, "_INFER_POOL", pool, raising=True)

    _reset_runtime_state()
    try:
        with TestClient(application) as test_client:
            yield test_client
    finally:
        _reset_runtime_state()
        pool.shutdown(wait=False)


# ========== 4. 模拟关键点 ==========
def build_landmarks(nose_y=0.35, hip_center_x=0.50):
    """按 MediaPipe 33 点结构生成关键点列表（默认值即端正坐姿）

    Args:
        nose_y: 鼻尖纵坐标（0 在画面顶部，越大越靠下），用于模拟低头
        hip_center_x: 髋部中心横坐标，用于模拟身体晃动
    """
    shoulder_y = 0.60
    hip_y = 0.85
    points = []

    def _add(x, y, z=0.0):
        points.append(SimpleNamespace(x=float(x), y=float(y), z=float(z), visibility=1.0))

    # 逐一下标填充：0 鼻、2/5 双眼、7/8 双耳、11/12 双肩、23/24 双髋，其余补中性坐标
    base = {
        0: (0.50, nose_y),
        1: (0.47, nose_y - 0.04),
        2: (0.47, nose_y - 0.04),
        3: (0.45, nose_y - 0.04),
        4: (0.53, nose_y - 0.04),
        5: (0.53, nose_y - 0.04),
        6: (0.55, nose_y - 0.04),
        7: (0.40, nose_y - 0.02),
        8: (0.60, nose_y - 0.02),
        9: (0.48, nose_y + 0.06),
        10: (0.52, nose_y + 0.06),
        11: (0.42, shoulder_y),
        12: (0.58, shoulder_y),
        13: (0.38, shoulder_y + 0.15),
        14: (0.62, shoulder_y + 0.15),
        15: (0.36, shoulder_y + 0.28),
        16: (0.64, shoulder_y + 0.28),
        17: (0.35, shoulder_y + 0.30),
        18: (0.65, shoulder_y + 0.30),
        19: (0.35, shoulder_y + 0.31),
        20: (0.65, shoulder_y + 0.31),
        21: (0.36, shoulder_y + 0.32),
        22: (0.64, shoulder_y + 0.32),
        23: (hip_center_x - 0.06, hip_y),
        24: (hip_center_x + 0.06, hip_y),
        25: (hip_center_x - 0.07, hip_y + 0.15),
        26: (hip_center_x + 0.07, hip_y + 0.15),
        27: (hip_center_x - 0.08, hip_y + 0.30),
        28: (hip_center_x + 0.08, hip_y + 0.30),
        29: (hip_center_x - 0.09, hip_y + 0.31),
        30: (hip_center_x + 0.09, hip_y + 0.31),
        31: (hip_center_x - 0.09, hip_y + 0.33),
        32: (hip_center_x + 0.09, hip_y + 0.33),
    }
    for idx in range(33):
        x, y = base[idx]
        _add(x, y)
    return points


@pytest.fixture
def fake_landmarks():
    """符合 MediaPipe 33 点结构的模拟关键点（端正坐姿）"""
    return build_landmarks()
