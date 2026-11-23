"""
专注度模型加载层：13 维特征契约常量、契约断言、SHA256 校验、模型/标准化器加载与推理入口。

本层不依赖 FastAPI、数据库、路由与 MediaPipe（特征抽取中的 MediaPipe 部分在 analyzer.py），
因此可被独立单测。
"""
import hashlib
import hmac
import os

import joblib
import numpy as np

from .config import BASE_DIR, logger


# ========== 模型加载（全局缓存，只加载一次）==========
# P0-7：模型反序列化加固 —— 彻底移除不安全的反序列化调用，改用 joblib；且必须通过 SHA256 白名单校验
# P1-1：模型目录真实名称为 model/（历史代码误写为 models/），并支持环境变量覆盖
MODEL_DIR = os.path.join(BASE_DIR, "model")
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




def model_is_loaded() -> bool:
    """当前进程是否已成功加载专注度模型（供 /healthz 只读查询，不触发加载）"""
    return bool(_cached_model is not None)
