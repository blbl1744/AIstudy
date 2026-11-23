"""
低头姿态误判修复专项测试（姿态算法迭代）

覆盖场景：
1. 正常坐姿看屏幕不误判；
2. 低头阅读/写作业不扣分（含真实关键点分级）；
3. 持续低头不累计走神、不消耗机会；
4. 低头后短暂抬头无状态跳变（防抖）；
5. 关键点丢失 → UNKNOWN 中性，不扣分不判走神；
6. 明显转头须叠加独立信号才判走神；
7. 低头不判疲劳，可靠长时间闭眼独立升级 FATIGUE 并判走神；
8. 正常→低头平稳过渡（防抖切换）；
9. 正常路径与 fallback 路径对「无法确认」处理一致；
10. 真正偏移/晃动等异常仍正常识别（未关闭走神检测）。

说明：本文件只构造可控模拟关键点/信号，不声称验证真实识别准确率。
每个测试使用新建的 MediaPipePoseAnalyzer 实例，天然隔离内部防抖状态，
不依赖 conftest 对模块级单例的清理。
"""
import copy

from app.analyzer import MediaPipePoseAnalyzer


def _new_engine():
    return MediaPipePoseAnalyzer()


# ---------- 1. 正常坐姿 ----------
def test_normal_posture_is_normal_and_not_distracted():
    eng = _new_engine()
    raw = eng._classify_attention_state("端正", "精力充沛", "注视屏幕", True, False)
    assert raw == "NORMAL"
    assert eng._is_true_distraction(raw, "端正", "精力充沛", "注视屏幕", "稳定", False) is False


# ---------- 2. 低头不扣分（相对比较，不依赖具体阈值） ----------
def test_head_down_does_not_lower_score():
    eng = _new_engine()
    base = dict(bs="稳定", fat="精力充沛", ef="注视屏幕", scene="selfstudy", stability=1.0)
    upright = eng._calculate_focus_score("端正", **base)
    head_down = eng._calculate_focus_score("明显低头", **base)
    slight_down = eng._calculate_focus_score("轻微低头", **base)
    assert head_down == upright, "明显低头不得扣分"
    assert slight_down == upright, "轻微低头不得扣分"


# ---------- 2b. 用模拟关键点验证低头分级链路 ----------
def test_head_down_landmarks_classified(fake_landmarks):
    eng = _new_engine()

    def _pose(lm):
        return eng._evaluate_head_pose(
            eng._calc_head_tilt(lm[0], lm[11], lm[12]),
            eng._calc_head_drop(lm[0], lm[11], lm[12]),
        )

    assert _pose(fake_landmarks) == "端正"
    down = copy.deepcopy(fake_landmarks)
    down[0].y = 0.595  # 鼻尖显著下移 → 肩鼻落差变小 → 明显低头
    assert _pose(down) == "明显低头"
    # 低头姿态归一化后为 HEAD_DOWN，且多信号判定不判走神
    raw = eng._classify_attention_state("明显低头", "精力充沛", "注视屏幕", True, False)
    assert raw == "HEAD_DOWN"
    assert eng._is_true_distraction(raw, "明显低头", "精力充沛", "注视屏幕", "稳定", False) is False


# ---------- 3. 持续低头不累计走神 ----------
def test_sustained_head_down_never_accumulates_distraction():
    eng = _new_engine()
    for _ in range(40):
        raw = eng._classify_attention_state("明显低头", "精力充沛", "注视屏幕", True, False)
        state = eng._debounce_attention(raw)
        d_raw = eng._is_true_distraction(state, "明显低头", "精力充沛", "注视屏幕", "稳定", False)
        flag = eng._debounce_distraction(d_raw)
        assert flag is False, "任何低头帧都不得判走神"
    assert state == "HEAD_DOWN"
    assert eng.is_distracted is False
    assert eng.consecutive_low_focus == 0, "低头不得累计低专注帧"
    assert eng.total_distractions == 0, "低头不得累计走神次数"


# ---------- 4. 低头后短暂抬头无跳变 ----------
def test_brief_lookup_after_head_down_no_jump():
    eng = _new_engine()
    for _ in range(6):
        raw = eng._classify_attention_state("明显低头", "精力充沛", "注视屏幕", True, False)
        state = eng._debounce_attention(raw)
    assert state == "HEAD_DOWN"
    # 单帧抬头：防抖要求连续 N 帧一致，单帧不应翻转状态或触发走神
    raw = eng._classify_attention_state("端正", "精力充沛", "注视屏幕", True, False)
    state_after = eng._debounce_attention(raw)
    assert state_after == "HEAD_DOWN"
    d = eng._debounce_distraction(False)
    assert d is False


# ---------- 5. 关键点丢失 → UNKNOWN 中性 ----------
def test_missing_landmarks_unknown_and_neutral():
    eng = _new_engine()
    raw = eng._classify_attention_state("未检测到", "未知", "未知", False, False)
    assert raw == "UNKNOWN"
    assert eng._is_true_distraction(raw, "未检测到", "未知", "未知", "未检测", False) is False
    fb = eng._fallback_analysis("selfstudy")
    assert fb["focus_score"] is None
    assert fb["attention_state"] == "UNKNOWN"
    assert fb["is_distracted"] is False
    assert fb["pose_detected"] is False


# ---------- 6. 明显转头须叠加独立信号 ----------
def test_head_turned_requires_independent_signal():
    eng = _new_engine()
    raw = eng._classify_attention_state("明显偏移", "精力充沛", "注视屏幕", True, False)
    assert raw == "HEAD_TURNED"
    # 仅转头（正视、身体稳定）不判走神
    assert eng._is_true_distraction(raw, "明显偏移", "精力充沛", "注视屏幕", "稳定", False) is False
    # 转头 + 视线频繁偏离 → 判走神
    assert eng._is_true_distraction(raw, "明显偏移", "精力充沛", "频繁偏离", "稳定", False) is True
    # 转头 + 身体明显晃动 → 判走神
    assert eng._is_true_distraction(raw, "明显偏移", "精力充沛", "注视屏幕", "明显晃动", False) is True


# ---------- 7. 疲劳独立：低头不判疲劳，可靠闭眼独立升级 ----------
def test_fatigue_independent_from_head_down():
    eng = _new_engine()
    # 低头但无可靠闭眼 → HEAD_DOWN，不是 FATIGUE（即使疲劳等级字段异常也不误升级）
    assert eng._classify_attention_state("明显低头", "明显疲劳", "注视屏幕", True, False) == "HEAD_DOWN"
    # 可靠长时间闭眼 → FATIGUE（低头姿态下也独立成立）
    assert eng._classify_attention_state("明显低头", "精力充沛", "注视屏幕", True, True) == "FATIGUE"
    # FATIGUE 且确认闭眼 → 判走神
    assert eng._is_true_distraction("FATIGUE", "明显低头", "精力充沛", "注视屏幕", "稳定", True) is True
    # 关键点丢失 → UNKNOWN，不判疲劳/走神
    assert eng._classify_attention_state("未检测", "未检测", "未检测", False, False) == "UNKNOWN"


# ---------- 8. 正常→低头平稳过渡（防抖切换） ----------
def test_normal_to_head_down_smooth_transition():
    eng = _new_engine()
    for _ in range(3):
        state = eng._debounce_attention("NORMAL")
    assert state == "NORMAL"
    # 单帧低头不翻转
    assert eng._debounce_attention("HEAD_DOWN") == "NORMAL"
    assert eng._debounce_attention("HEAD_DOWN") == "NORMAL"
    # 连续第 3 帧低头才切换
    assert eng._debounce_attention("HEAD_DOWN") == "HEAD_DOWN"


# ---------- 9. 正常路径与 fallback 路径一致 ----------
def test_fallback_and_normal_path_consistent():
    eng = _new_engine()
    fb = eng._fallback_analysis("selfstudy")
    for key in ("attention_state", "is_distracted", "focus_score"):
        assert key in fb, "fallback 返回须与正常路径同构"
    # 两路径对「无法确认」都用 UNKNOWN + 不判走神，处理一致
    assert fb["attention_state"] == "UNKNOWN"
    assert fb["is_distracted"] is False
    raw = eng._classify_attention_state("未检测到", "未知", "未知", False, False)
    assert raw == fb["attention_state"]
    assert eng._is_true_distraction(raw, "未检测到", "未知", "未知", "未检测", False) == fb["is_distracted"]


# ---------- 10. 真正异常仍识别（未关闭走神检测） ----------
def test_real_anomalies_still_detected():
    eng = _new_engine()
    base = dict(bs="稳定", fat="精力充沛", ef="注视屏幕", scene="selfstudy", stability=1.0)
    upright = eng._calculate_focus_score("端正", **base)
    # 头部明显偏移仍扣分
    assert eng._calculate_focus_score("明显偏移", **base) < upright
    # 身体明显晃动仍扣分
    assert eng._calculate_focus_score("端正", bs="明显晃动", fat="精力充沛", ef="注视屏幕",
                                      scene="selfstudy", stability=1.0) < upright
    # 闭眼/走神的视线状态仍扣分
    assert eng._calculate_focus_score("端正", bs="稳定", fat="精力充沛", ef="闭眼/走神",
                                      scene="selfstudy", stability=1.0) < upright
