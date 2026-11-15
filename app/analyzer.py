"""
姿态分析层：threading.local 的 MediaPipe 懒加载 + MediaPipePoseAnalyzer 几何规则打分，
以及「摄像头帧 → 13 维特征」的抽取封装。

设计约束：
1. 本模块不 import FastAPI/路由，便于被推理线程池直接调用与单元测试；
2. MediaPipe 实例按线程隔离（threading.local），避免多线程共享同一 graph 造成状态串台。
"""
import base64
import math
import random
import threading
import time
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

from .config import (
    ACTION_THRESHOLDS,
    EYE_CLOSED_FRAMES,
    POSE_CONFIG,
    POSE_STABILITY,
    POSE_THRESHOLDS,
    SCORE_PENALTIES,
    SCORE_SMOOTHING,
    WARNING_LEVELS,
    logger,
)
from .model_loader import (
    FEATURE_DIM,
    MODEL_MIN_CONFIDENCE,
    extract_model_features,
    load_trained_model,
    predict_focus,
)
from .state import student_notifications


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
        except Exception:
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
