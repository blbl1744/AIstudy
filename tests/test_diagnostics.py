"""姿态诊断日志专项测试（第二阶段：真实摄像头数据验证机制）。

覆盖：
1. 诊断开关默认关闭：log_frame 返回 False、不产生记录、不落盘；
2. 开启后字段完整：每条记录包含全部白名单字段，timestamp 自动生成；
3. 不影响评分/惩罚：开/关两种状态下 _calculate_focus_score 与 _is_true_distraction 结果一致；
4. 导出格式正确：JSONL 每行可 json.loads，CSV 表头与白名单一致；
5. 兜底帧记录为 UNKNOWN 中性；
6. 字段白名单不含敏感字段与原始图像。

说明：本文件只验证诊断机制的工程正确性，不声称验证真实识别准确率。
"""
import csv
import json

import pytest

from app.analyzer import MediaPipePoseAnalyzer
from app.diagnostics import (
    DECISION_FIELDS,
    DIAGNOSTIC_FIELDS,
    FORBIDDEN_FIELDS,
    binary_metrics,
    build_decision_reason,
    diagnostics,
)
from app.routers.room_routes import _resolve_distraction


@pytest.fixture
def diag_env(tmp_path):
    """隔离诊断单例状态：每例前后保存/恢复，日志写入 tmp_path。"""
    saved = {
        "enabled": diagnostics.enabled,
        "fmt": diagnostics.fmt,
        "dir": diagnostics.dir,
        "path": diagnostics.path,
        "decision_path": diagnostics.decision_path,
        "records": diagnostics.records,
        "decision_records": diagnostics.decision_records,
    }
    diagnostics.set_enabled(False)
    diagnostics.fmt = "jsonl"
    diagnostics.dir = str(tmp_path)
    diagnostics.path = str(tmp_path / "pose_diagnostics.jsonl")
    diagnostics.decision_path = str(tmp_path / "pose_decisions.jsonl")
    diagnostics.clear()
    yield diagnostics
    diagnostics.set_enabled(saved["enabled"])
    diagnostics.fmt = saved["fmt"]
    diagnostics.dir = saved["dir"]
    diagnostics.path = saved["path"]
    diagnostics.decision_path = saved["decision_path"]
    with diagnostics._lock:
        diagnostics._records = list(saved["records"])
        diagnostics._decision_records = list(saved["decision_records"])


_SAMPLE = {
    "attention_state": "HEAD_DOWN",
    "is_distracted": False,
    "rule_score": 96,
    "focus_score": 96,
    "distraction_count": 0,
    "head_pose_status": "明显低头",
    "head_tilt": 2.1,
    "head_drop": 0.12,
    "fatigue_status": "精力充沛",
    "eye_focus_status": "注视屏幕",
    "ear_value": 0.31,
    "confirmed_eyes_closed": False,
    "body_stability_status": "稳定",
    "body_sway": 3.2,
    "shoulder_level": 0.01,
    "pose_stability": 0.9,
    "pose_detected": True,
    "face_detected": True,
    "decision_reason": "状态=HEAD_DOWN;低头属正常学习姿态，不判走神;走神=否",
}


# ---------- 1. 默认关闭零副作用 ----------
def test_diagnostics_default_off(diag_env):
    assert diag_env.enabled is False
    before = diag_env.count
    assert diag_env.log_frame(_SAMPLE) is False
    assert diag_env.count == before
    import os
    assert not os.path.exists(diag_env.path), "关闭时不得落盘"


# ---------- 2. 开启后字段完整 ----------
def test_fields_complete_when_enabled(diag_env):
    diag_env.set_enabled(True)
    assert diag_env.log_frame(_SAMPLE) is True
    assert diag_env.count == 1
    rec = diag_env.records[0]
    for key in DIAGNOSTIC_FIELDS:
        assert key in rec, "记录缺字段: %s" % key
    assert rec["timestamp"], "timestamp 须自动生成"
    assert rec["attention_state"] == "HEAD_DOWN"
    assert rec["is_distracted"] is False


# ---------- 3. 不影响评分/惩罚 ----------
def test_no_effect_on_score_and_distraction(diag_env):
    eng = MediaPipePoseAnalyzer()
    base = dict(hp="端正", bs="稳定", fat="精力充沛", ef="注视屏幕",
                scene="selfstudy", stability=1.0)
    score_off = eng._calculate_focus_score(**base)
    dist_off = eng._is_true_distraction("NORMAL", "端正", "精力充沛", "注视屏幕", "稳定", False)

    diag_env.set_enabled(True)
    for _ in range(5):
        diag_env.log_frame(_SAMPLE)
    score_on = eng._calculate_focus_score(**base)
    dist_on = eng._is_true_distraction("NORMAL", "端正", "精力充沛", "注视屏幕", "稳定", False)

    assert score_on == score_off, "诊断开启不得改变评分"
    assert dist_on == dist_off, "诊断开启不得改变走神判定"
    assert eng.total_distractions == 0, "诊断不得触碰走神累计"


# ---------- 4a. JSONL 导出格式 ----------
def test_export_jsonl(diag_env, tmp_path):
    diag_env.set_enabled(True)
    diag_env.log_frame(_SAMPLE)
    diag_env.log_frame(_SAMPLE)
    out = diag_env.export(path=str(tmp_path / "out.jsonl"), fmt="jsonl")
    with open(out, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert set(DIAGNOSTIC_FIELDS).issubset(obj.keys())


# ---------- 4b. CSV 导出格式 ----------
def test_export_csv(diag_env, tmp_path):
    diag_env.set_enabled(True)
    diag_env.log_frame(_SAMPLE)
    out = diag_env.export(path=str(tmp_path / "out.csv"), fmt="csv")
    with open(out, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == DIAGNOSTIC_FIELDS
    assert len(rows) == 2


# ---------- 5. 兜底帧记录为 UNKNOWN ----------
def test_fallback_frame_logged_as_unknown(diag_env):
    diag_env.set_enabled(True)
    eng = MediaPipePoseAnalyzer()
    result = eng._fallback_analysis("selfstudy")
    assert result["attention_state"] == "UNKNOWN"
    assert result["is_distracted"] is False
    assert diag_env.count == 1
    rec = diag_env.records[0]
    assert rec["attention_state"] == "UNKNOWN"
    assert rec["pose_detected"] is False
    assert rec["is_distracted"] is False


# ---------- 6. 白名单不含敏感字段 ----------
def test_no_sensitive_or_image_fields():
    fl = set(DIAGNOSTIC_FIELDS)
    assert fl.isdisjoint(FORBIDDEN_FIELDS), "诊断字段含敏感/图像字段"
    for bad in ("image", "base64", "password", "token", "secret", "student_id"):
        assert not any(bad in f.lower() for f in fl), "诊断字段疑似含: %s" % bad


# ---------- 7. 判定原因纯函数 ----------
def test_decision_reason_pure():
    r = build_decision_reason("HEAD_DOWN", "精力充沛", "注视屏幕", "稳定", False, False)
    assert "HEAD_DOWN" in r and "走神=否" in r
    r2 = build_decision_reason("HEAD_TURNED", "精力充沛", "频繁偏离", "稳定", False, True)
    assert "视线偏离" in r2 and "走神=是" in r2


# ========== 第二阶段补充审查新增测试 ==========

# ---------- 8. analyzer 正常提供字段 → decision_source=analyzer ----------
def test_analyzer_field_records_correct_source(diag_env):
    diag_env.set_enabled(True)
    final, source, analyzer_val, fallback = _resolve_distraction({"is_distracted": True}, 95)
    assert (final, source, analyzer_val, fallback) == (True, "analyzer", True, False)
    assert diag_env.log_decision({
        "decision_source": source,
        "final_is_distracted": final,
        "analyzer_is_distracted": analyzer_val,
        "legacy_fallback_used": fallback,
    }) is True
    rec = diag_env.decision_records[0]
    assert rec["decision_source"] == "analyzer"
    assert rec["final_is_distracted"] is True
    assert rec["analyzer_is_distracted"] is True
    assert rec["legacy_fallback_used"] is False
    assert rec["timestamp"]


# ---------- 9. analyzer 缺失字段 → 旧阈值回退 ----------
def test_missing_field_records_legacy_fallback(diag_env):
    diag_env.set_enabled(True)
    # score 低于阈值 → 回退判走神
    final_lo, source, analyzer_val, fallback = _resolve_distraction({}, 40)
    assert (final_lo, source, analyzer_val, fallback) == (True, "legacy_fallback", None, True)
    # score 为 None → UNKNOWN（analyzer 推理异常场景，绝不回退为「确认走神」）
    final_none, source_none, analyzer_none, fallback_none = _resolve_distraction({}, None)
    assert (final_none, source_none, analyzer_none, fallback_none) == (None, "unknown", None, False)
    # score 高于阈值 → 回退判不走神
    final_hi, _, _, _ = _resolve_distraction({}, 95)
    assert final_hi is False
    assert diag_env.log_decision({
        "decision_source": source,
        "final_is_distracted": final_lo,
        "analyzer_is_distracted": analyzer_val,
        "legacy_fallback_used": fallback,
    }) is True
    rec = diag_env.decision_records[0]
    assert rec["decision_source"] == "legacy_fallback"
    assert rec["analyzer_is_distracted"] is None
    assert rec["legacy_fallback_used"] is True
    assert rec["final_is_distracted"] is True


# ---------- 10. 诊断开/关不改变最终裁决结果 ----------
def test_diag_toggle_does_not_change_final_decision(diag_env):
    payloads = [{"is_distracted": True}, {"is_distracted": False}, {}, {"is_distracted": "not-bool"}]
    scores = [95, 40, None, 60]
    diag_env.set_enabled(False)
    off = [_resolve_distraction(p, s)[0] for p, s in zip(payloads, scores)]
    diag_env.set_enabled(True)
    on = [_resolve_distraction(p, s)[0] for p, s in zip(payloads, scores)]
    assert off == on, "诊断开关不得改变最终裁决"
    # 末例 score=60 不<阈值→False；第 3 例缺字段且 score=None→UNKNOWN(None)
    assert off == [True, False, None, False]


# ---------- 11. 裁决字段关闭时零落盘 ----------
def test_decision_logging_default_off_zero_cost(diag_env):
    import os
    before = diag_env.decision_count
    assert diag_env.log_decision({
        "decision_source": "analyzer",
        "final_is_distracted": True,
    }) is False
    assert diag_env.decision_count == before
    assert not os.path.exists(diag_env.decision_path), "关闭时裁决日志不得落盘"


# ---------- 12. 裁决导出 JSONL / CSV 字段完整 ----------
def test_decision_export_complete(diag_env, tmp_path):
    diag_env.set_enabled(True)
    for i in range(2):
        diag_env.log_decision({
            "decision_source": "analyzer" if i == 0 else "legacy_fallback",
            "final_is_distracted": bool(i),
            "analyzer_is_distracted": True if i == 0 else None,
            "legacy_fallback_used": bool(i),
        })
    jl = diag_env.export(path=str(tmp_path / "dec.jsonl"), fmt="jsonl", kind="decision")
    with open(jl, encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh.read().splitlines() if ln.strip()]
    assert len(rows) == 2
    for row in rows:
        assert set(DECISION_FIELDS).issubset(row.keys())
    cs = diag_env.export(path=str(tmp_path / "dec.csv"), fmt="csv", kind="decision")
    with open(cs, encoding="utf-8-sig", newline="") as fh:
        csv_rows = list(csv.reader(fh))
    assert csv_rows[0] == DECISION_FIELDS
    assert len(csv_rows) == 3


# ---------- 13. 裁决字段与敏感字段白名单不相交 ----------
def test_decision_fields_no_sensitive():
    fl = set(DECISION_FIELDS)
    assert fl.isdisjoint(FORBIDDEN_FIELDS)
    for bad in ("image", "base64", "password", "token", "secret", "student_id", "user_id"):
        assert not any(bad in f.lower() for f in fl), "裁决字段疑似含: %s" % bad


# ---------- 14. UNKNOWN 不参与二分类，计入覆盖率 ----------
def test_unknown_excluded_from_binary_metrics():
    # (ground_truth, predicted)：2 正 1 负可标注 + 2 个 UNKNOWN（None）
    samples = [(True, True), (False, False), (True, False), (None, True), (True, None)]
    m = binary_metrics(samples, task="attention")
    assert m["total"] == 5
    assert m["labelable"] == 3          # 只有 3 个可标注样本进入二分类
    assert m["unknown_count"] == 2
    assert round(m["coverage"], 4) == round(3 / 5, 4)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 0, 1, 1)
    # UNKNOWN 既不算误报也不算漏报：fp/fn 合计只来自可标注样本
    assert m["fp"] + m["fn"] == 1


# ---------- 15. 疲劳与走神各自独立评价 ----------
def test_fatigue_and_attention_independent_metrics():
    # 同一批样本：注意力偏离与疲劳分别标注、分别评价，指标互不串用
    attention_samples = [(True, True), (False, False), (True, True), (False, False)]
    fatigue_samples = [(True, False), (False, False), (True, True), (False, True)]
    a = binary_metrics(attention_samples, task="attention")
    f = binary_metrics(fatigue_samples, task="fatigue")
    assert a["task"] == "attention" and f["task"] == "fatigue"
    assert (a["tp"], a["fp"], a["tn"], a["fn"]) == (2, 0, 2, 0)   # 注意力全对
    assert a["tpr"] == 1.0 and a["fpr"] == 0.0
    # 疲劳识别有 1 漏报(fn)、1 误报(fp)，与注意力指标不同 → 证明两套指标独立
    assert (f["tp"], f["fp"], f["tn"], f["fn"]) == (1, 1, 1, 1)
    assert f["tpr"] == 0.5 and f["fpr"] == 0.5
    assert a["tpr"] != f["tpr"], "疲劳与走神须用独立评价指标"
    assert a["coverage"] == f["coverage"] == 1.0


# ---------- 16. 全 UNKNOWN 时分母为 0 安全返回 None ----------
def test_binary_metrics_all_unknown():
    m = binary_metrics([(None, None), (None, True)])
    assert m["labelable"] == 0
    assert m["coverage"] == 0.0
    assert m["tpr"] is None and m["fpr"] is None
    assert m["precision"] is None and m["recall"] is None


# ========== 最终修复：UNKNOWN 中性裁决（analyzer 推理异常 result=None） ==========

def _base_room(**overrides):
    """构造最小房间字典；预设非零连续走神累计，用于验证 UNKNOWN 不清零。"""
    room = {
        "room_id": "r-unknown-neutral",
        "strikes_left": 2,
        "distracted_streak_sec": 5.0,
        "is_locked": False,
        "punishment_type": "",
        "grace_until": 0.0,
        "watchers": {},
        "score_history": [],
    }
    room.update(overrides)
    return room


def _drive_frame(monkeypatch, room, result, *, strike_spy=None, saved=None,
                 broadcasts=None, now=1000.0):
    """在隔离环境驱动一次 _on_frame_done，spy 全部外部依赖，无需 client/DB。"""
    import asyncio
    import app.routers.room_routes as rr
    from app import state as app_state

    rid = room["room_id"]
    # active_rooms 是 TTLCache（不支持 item 赋值），整体替换为普通 dict 以便隔离
    monkeypatch.setattr(app_state, "active_rooms", {rid: room})
    monkeypatch.setattr(app_state, "_inflight_students", {1})
    monkeypatch.setattr(rr, "_now_ts", lambda: now)
    monkeypatch.setattr(
        rr, "save_behavior_record",
        lambda uid, data, session_id=None: (
            saved.append((uid, dict(data), session_id)) if saved is not None else None
        ),
    )

    async def _fake_broadcast(room_id, payload, exclude_user_id=None):
        if broadcasts is not None:
            broadcasts.append((room_id, dict(payload)))

    async def _fake_dialogue(*args, **kwargs):
        return None

    monkeypatch.setattr(rr, "_broadcast", _fake_broadcast)
    monkeypatch.setattr(rr, "_broadcast_dialogue", _fake_dialogue)

    real_strike = rr.evaluate_strike

    def _spy(r, is_distracted, delta):
        if strike_spy is not None:
            strike_spy.append((is_distracted, delta))
        return real_strike(r, is_distracted, delta)

    monkeypatch.setattr(rr, "evaluate_strike", _spy)
    asyncio.run(rr._on_frame_done(rid, 1, "ZmFrZQ==", result))
    return room


# ---------- T1. analyzer=True 识别走神（bool 优先于分数） ----------
def test_t1_analyzer_true_wins(monkeypatch):
    assert _resolve_distraction({"is_distracted": True}, 95)[:2] == (True, "analyzer")
    # 即便分数很高，analyzer 明确走神也以 analyzer 为准
    assert _resolve_distraction({"is_distracted": True}, 99)[0] is True
    # 集成路径：analyzer=True 仍正常调用 evaluate_strike 并累计走神
    spy = []
    room = _base_room(distracted_streak_sec=2.0, last_frame_ts=999.0)  # now=1000 → delta=1.0
    _drive_frame(monkeypatch, room, {"is_distracted": True, "focus_score": 95}, strike_spy=spy)
    assert spy and spy[0][0] is True, "analyzer=True 须正常进入惩罚引擎"
    assert room["distracted_streak_sec"] > 2.0, "analyzer 走神须正常累计"


# ---------- T2. analyzer=False 识别未走神（bool 优先于低分） ----------
def test_t2_analyzer_false_wins():
    assert _resolve_distraction({"is_distracted": False}, 40)[:2] == (False, "analyzer")
    # 即便分数低于阈值，analyzer 明确未走神也不误判
    assert _resolve_distraction({"is_distracted": False}, 10)[0] is False


# ---------- T3. 缺字段 + score=50 旧阈值有效（走神） ----------
def test_t3_missing_field_score50_legacy():
    r = _resolve_distraction({}, 50)
    assert (r[0], r[1], r[3]) == (True, "legacy_fallback", True)
    assert r[2] is None


# ---------- T4. 缺字段 + score=80 旧阈值有效（未走神） ----------
def test_t4_missing_field_score80_legacy():
    r = _resolve_distraction({}, 80)
    assert (r[0], r[1], r[3]) == (False, "legacy_fallback", True)
    assert r[2] is None


# ---------- T5. 缺字段 + score=None 进入 UNKNOWN ----------
def test_t5_missing_field_score_none_unknown():
    r = _resolve_distraction({}, None)
    assert r == (None, "unknown", None, False)


# ---------- T6. UNKNOWN 不调用 evaluate_strike ----------
def test_t6_unknown_skips_evaluate_strike(monkeypatch):
    spy = []
    saved = []
    room = _base_room()
    _drive_frame(monkeypatch, room, None, strike_spy=spy, saved=saved)
    assert spy == [], "UNKNOWN(result=None) 不得调用 evaluate_strike"
    assert len(saved) == 1, "跳过惩罚不得影响行为记录保存"


# ---------- T7. UNKNOWN 不改变连续走神计数及惩罚状态 ----------
def test_t7_unknown_preserves_punishment_state(monkeypatch):
    room = _base_room(strikes_left=2, distracted_streak_sec=5.0,
                      is_locked=False, punishment_type="")
    before = dict(room)
    _drive_frame(monkeypatch, room, None, now=before["distracted_streak_sec"] / 2 + 1000.0)
    for key in ("strikes_left", "distracted_streak_sec", "is_locked", "punishment_type"):
        assert room[key] == before[key], "UNKNOWN 不得改变惩罚状态字段: %s" % key
    assert "strikes_used" not in room, "UNKNOWN 不得累加真实扣血次数"


# ---------- T8. UNKNOWN 诊断字段正确（不静默转 False） ----------
def test_t8_unknown_decision_fields(monkeypatch, diag_env):
    diag_env.set_enabled(True)
    broadcasts = []
    room = _base_room()
    _drive_frame(monkeypatch, room, None, broadcasts=broadcasts)
    assert len(diag_env.decision_records) == 1
    rec = diag_env.decision_records[0]
    assert rec["decision_source"] == "unknown"
    assert rec["final_is_distracted"] is None
    assert rec["analyzer_is_distracted"] is None
    assert rec["legacy_fallback_used"] is False
    # 广播照常，且走神时长用房间原值（中性），不把 None 转成 0 或误报走神
    assert any(p.get("type") == "analysis_update" for _, p in broadcasts)


# ---------- T9. 诊断开/关实际裁决一致 ----------
def test_t9_diag_toggle_same_verdict_integration(monkeypatch, diag_env):
    def _run(enabled):
        diag_env.set_enabled(enabled)
        spy = []
        room = _base_room()
        broadcasts = []
        _drive_frame(monkeypatch, room, None, strike_spy=spy, broadcasts=broadcasts)
        payload = next(p for _, p in broadcasts if p.get("type") == "analysis_update")
        return (tuple(spy), room["strikes_left"], room["distracted_streak_sec"],
                payload["focus_score"], payload["distracted_streak_sec"])

    assert _run(False) == _run(True), "诊断开/关不得改变裁决与广播"


# ---------- T10. analyzer 推理异常后房间仍能正常运行 ----------
def test_t10_room_still_works_after_analyzer_exception(monkeypatch, diag_env):
    diag_env.set_enabled(True)
    spy = []
    saved = []
    broadcasts = []
    room = _base_room()
    # analyzer 推理异常 → result=None → payload={} → score=None → UNKNOWN
    _drive_frame(monkeypatch, room, None, strike_spy=spy, saved=saved, broadcasts=broadcasts)
    assert spy == [], "异常帧不得触发惩罚"
    assert saved and saved[0][1]["focus_score"] is None, "保存照常（含 None 分数）"
    types = [p.get("type") for _, p in broadcasts]
    assert "analysis_update" in types, "WS 分析结果广播照常"
    assert not any(t in types for t in ("punishment_required", "punishment_cleared")), \
        "UNKNOWN 不得下发惩罚消息"
    assert room["is_locked"] is False, "房间未被误锁"
    assert room["strikes_left"] == 2, "生命值未被误扣"


# ============================================================
# 最终修复验收：UNKNOWN 语义统一 + 哨兵预警当前帧有效性约束
# 覆盖：裁决逻辑 / 惩罚状态机 / 哨兵预警 / 诊断记录 四组
# 全部为纯 Python 单测，不依赖真实 DB/WS/线程池，不落盘用户数据
# ============================================================

# ---------- 组1：裁决逻辑 _resolve_distraction ----------

def test_fix_resolve_fallback_unknown_false_score_none_yields_none():
    """fallback 帧 attention_state=UNKNOWN + is_distracted=False + score=None → None（问题A核心）"""
    payload = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    assert _resolve_distraction(payload, None) == (None, "unknown", None, False)


def test_fix_resolve_thread_exception_none_yields_none():
    """线程池异常 result=None → payload={} → UNKNOWN(None)"""
    assert _resolve_distraction({}, None) == (None, "unknown", None, False)


def test_fix_resolve_normal_false_and_true():
    """正常 NORMAL + 有效 bool → 直接采用 bool"""
    assert _resolve_distraction({"attention_state": "NORMAL", "is_distracted": False}, 90)[:2] == (False, "analyzer")
    assert _resolve_distraction({"attention_state": "NORMAL", "is_distracted": True}, 30)[:2] == (True, "analyzer")


def test_fix_resolve_explicit_unknown_with_true_not_distraction():
    """显式 UNKNOWN 即使伴 is_distracted=True 也不得当有效走神（优先级1）"""
    payload = {"attention_state": "UNKNOWN", "is_distracted": True, "focus_score": None}
    assert _resolve_distraction(payload, None) == (None, "unknown", None, False)


def test_fix_resolve_legacy_numeric_threshold_without_attention_state():
    """旧版 analyzer 无 attention_state：数值阈值兼容不变；analyzer_is_distracted=None，legacy=True"""
    assert _resolve_distraction({"focus_score": 50}, 50) == (True, "legacy_fallback", None, True)
    assert _resolve_distraction({"focus_score": 80}, 80) == (False, "legacy_fallback", None, True)


def test_fix_resolve_score_none_not_coerced_to_zero():
    """score=None 不得转 0；无 attention_state、无 bool 时仍为 UNKNOWN，不走数值阈值"""
    final, source, aval, legacy = _resolve_distraction({"focus_score": None}, None)
    assert final is None and source == "unknown" and legacy is False
    assert final is not False  # 不得把 None 当 False（确认未走神）


# ---------- 组2：惩罚状态机（fallback UNKNOWN 不动任何状态） ----------

def test_fix_punishment_fallback_unknown_skips_evaluate_strike(monkeypatch):
    """fallback UNKNOWN 帧不调用 evaluate_strike，惩罚状态机完全不动（问题A修复验证）"""
    spy = []
    room = _base_room(distracted_streak_sec=5.0, strikes_left=2)
    before = dict(room)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False,
                "focus_score": None, "annotated_image": "ZmFrZQ=="}
    _drive_frame(monkeypatch, room, fallback, strike_spy=spy)
    assert spy == [], "UNKNOWN 不得调用 evaluate_strike"
    for key in ("distracted_streak_sec", "strikes_left", "is_locked", "punishment_type"):
        assert room[key] == before[key], f"UNKNOWN 不得改变 {key}"
    assert "strikes_used" not in room, "UNKNOWN 不得增加真实扣血次数"


def test_fix_punishment_consecutive_unknown_preserves_streak(monkeypatch):
    """连续多次 UNKNOWN（fallback 与 result=None 交替）：不清零、不增加累计"""
    spy = []
    room = _base_room(distracted_streak_sec=5.0, strikes_left=2)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    for now in (1000.0, 1001.0, 1002.0, 1003.0):
        result = fallback if int(now) % 2 == 0 else None
        _drive_frame(monkeypatch, room, result, strike_spy=spy, now=now)
    assert spy == []
    assert room["distracted_streak_sec"] == 5.0
    assert room["strikes_left"] == 2


def test_fix_punishment_recovers_after_unknown(monkeypatch):
    """UNKNOWN 后恢复正常确认走神：惩罚状态机正常工作（累计追加）"""
    spy = []
    room = _base_room(distracted_streak_sec=2.0, strikes_left=2, last_frame_ts=999.0)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    _drive_frame(monkeypatch, room, fallback, strike_spy=spy, now=1000.0)
    assert spy == []
    assert room["distracted_streak_sec"] == 2.0
    # 恢复：确认走神，delta=1.0 → 累计 3.0
    _drive_frame(monkeypatch, room, {"is_distracted": True, "focus_score": 40},
                 strike_spy=spy, now=1001.0)
    assert spy and spy[0][0] is True, "恢复后应正常进入惩罚引擎"
    assert room["distracted_streak_sec"] == 3.0


# ---------- 组3：哨兵预警当前帧有效性约束 ----------

def _drive_frame_with_watchers(monkeypatch, room, result, *, now=1000.0, alert_sec=3.0,
                               cooldown_sec=30.0, sentinel_spy=None, strike_spy=None):
    """在 _drive_frame 基础上注入督学官、压缩哨兵阈值并捕获哨兵单播消息。"""
    import asyncio
    import app.routers.room_routes as rr
    from app import state as app_state

    rid = room["room_id"]
    monkeypatch.setattr(app_state, "active_rooms", {rid: room})
    monkeypatch.setattr(app_state, "_inflight_students", {1})
    monkeypatch.setattr(rr, "_now_ts", lambda: now)
    monkeypatch.setattr(rr, "SENTINEL_ALERT_SEC", alert_sec)
    monkeypatch.setattr(rr, "SENTINEL_COOLDOWN_SEC", cooldown_sec)
    monkeypatch.setattr(rr, "save_behavior_record", lambda *a, **k: None)

    async def _fake_broadcast(*a, **k):
        return None

    async def _fake_dialogue(*a, **k):
        return None

    async def _fake_send(room_id, user_id, payload):
        if sentinel_spy is not None:
            sentinel_spy.append((user_id, dict(payload)))

    monkeypatch.setattr(rr, "_broadcast", _fake_broadcast)
    monkeypatch.setattr(rr, "_broadcast_dialogue", _fake_dialogue)
    monkeypatch.setattr(rr, "_send_to", _fake_send)

    real_strike = rr.evaluate_strike

    def _spy(r, is_distracted, delta):
        if strike_spy is not None:
            strike_spy.append((is_distracted, delta))
        return real_strike(r, is_distracted, delta)

    monkeypatch.setattr(rr, "evaluate_strike", _spy)
    asyncio.run(rr._on_frame_done(rid, 1, "ZmFrZQ==", result))
    return room


def _alerts(spy):
    return [p for _, p in spy if p.get("type") == "sentinel_alert"]


def _room_with_watcher(**overrides):
    base = dict(distracted_streak_sec=5.0, strikes_left=3, watchers={99: {"nickname": "督学官A"}})
    base.update(overrides)
    return _base_room(**base)


def test_fix_sentinel_unknown_with_history_does_not_alert(monkeypatch):
    """场景A：历史累计达阈值 + 当前帧 fallback UNKNOWN + 督学官在线 → 不触发新预警（问题B核心）"""
    spy = []
    room = _room_with_watcher(distracted_streak_sec=5.0)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    _drive_frame_with_watchers(monkeypatch, room, fallback, sentinel_spy=spy)
    assert _alerts(spy) == [], "UNKNOWN 帧不得用历史累计触发哨兵预警"
    assert room["distracted_streak_sec"] == 5.0, "UNKNOWN 帧历史累计不被清零"
    assert "_last_sentinel_alert_ts" not in room, "UNKNOWN 帧不得刷新冷却时间戳"


def test_fix_sentinel_watcher_joins_during_unknown_no_stale_alert(monkeypatch):
    """场景A补充：UNKNOWN 帧期间督学官才加入 → 仍不触发过期预警"""
    spy = []
    room = _base_room(distracted_streak_sec=5.0, strikes_left=3, watchers={})
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    _drive_frame_with_watchers(monkeypatch, room, fallback, sentinel_spy=spy)
    # 帧处理期间注入督学官（模拟此时加入），哨兵判定读的是当前 room
    room["watchers"] = {99: {"nickname": "督学官A"}}
    assert _alerts(spy) == []


def test_fix_sentinel_unknown_then_recover_distraction_alerts(monkeypatch):
    """场景B：累计→UNKNOWN→恢复且确认走神 → 只依据恢复后的有效结果触发预警"""
    spy = []
    room = _room_with_watcher(distracted_streak_sec=3.0)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    # UNKNOWN 帧：不触发、不清累计
    _drive_frame_with_watchers(monkeypatch, room, fallback, now=1000.0, sentinel_spy=spy)
    assert _alerts(spy) == []
    assert room["distracted_streak_sec"] == 3.0
    # 恢复帧：确认走神，delta=1.0 → 累计 4.0 ≥ alert_sec=3 → 触发（真实有效结果）
    _drive_frame_with_watchers(monkeypatch, room, {"is_distracted": True, "focus_score": 40},
                               now=1001.0, sentinel_spy=spy)
    assert len(_alerts(spy)) == 1, "恢复后有效走神应按正常规则触发哨兵"
    assert room["_last_sentinel_alert_ts"] == 1001.0


def test_fix_sentinel_unknown_then_recover_not_distraction_no_alert(monkeypatch):
    """场景C：累计→UNKNOWN→恢复确认未走神 → 正常 False 清零累计，不预警"""
    spy = []
    room = _room_with_watcher(distracted_streak_sec=5.0)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    _drive_frame_with_watchers(monkeypatch, room, fallback, now=1000.0, sentinel_spy=spy)
    assert _alerts(spy) == []
    _drive_frame_with_watchers(monkeypatch, room, {"is_distracted": False, "focus_score": 90},
                               now=1001.0, sentinel_spy=spy)
    assert _alerts(spy) == [], "确认未走神不触发哨兵"
    assert room["distracted_streak_sec"] == 0.0, "正常 False 仍按既有逻辑清零累计"


def test_fix_sentinel_consecutive_unknown_no_alert(monkeypatch):
    """场景D：连续多次 UNKNOWN（fallback + result=None）→ 不产生新预警、不改累计"""
    spy = []
    room = _room_with_watcher(distracted_streak_sec=5.0)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    for i, now in enumerate((1000.0, 1001.0, 1002.0, 1003.0)):
        result = fallback if i % 2 == 0 else None
        _drive_frame_with_watchers(monkeypatch, room, result, now=now, sentinel_spy=spy)
    assert _alerts(spy) == [], "连续 UNKNOWN 不得触发任何预警"
    assert room["distracted_streak_sec"] == 5.0
    assert "_last_sentinel_alert_ts" not in room


def test_fix_sentinel_normal_sustained_distraction_still_alerts(monkeypatch):
    """回归保护：正常持续走神达阈值仍能触发哨兵，冷却期内不重复"""
    spy = []
    room = _room_with_watcher(distracted_streak_sec=3.0)
    frame = {"is_distracted": True, "focus_score": 40}
    _drive_frame_with_watchers(monkeypatch, room, frame, now=1000.0, sentinel_spy=spy)
    assert len(_alerts(spy)) == 1, "正常走神应触发哨兵"
    # 冷却期内再次走神：重置累计到阈值再次达标，但不预警
    room["distracted_streak_sec"] = 3.0
    _drive_frame_with_watchers(monkeypatch, room, frame, now=1005.0, sentinel_spy=spy,
                               cooldown_sec=30.0)
    assert len(_alerts(spy)) == 1, "冷却期内不得重复预警"


# ---------- 组4：诊断记录字段统一 ----------

def test_fix_diag_unknown_decision_fields(diag_env):
    """UNKNOWN 时 decision 记录：source=unknown / final=None / analyzer=None / legacy=False"""
    diag_env.set_enabled(True)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    final, source, aval, legacy = _resolve_distraction(fallback, None)
    diag_env.log_decision({
        "decision_source": source,
        "final_is_distracted": final,
        "analyzer_is_distracted": aval,
        "legacy_fallback_used": legacy,
    })
    rec = diag_env.decision_records[-1]
    assert rec["decision_source"] == "unknown"
    assert rec["final_is_distracted"] is None
    assert rec["analyzer_is_distracted"] is None
    assert rec["legacy_fallback_used"] is False


def test_fix_diag_raw_analyzer_output_preserved_in_frame_log(diag_env):
    """帧级日志按原始输出保留 is_distracted=False + attention_state=UNKNOWN，与裁决级 None 可区分"""
    diag_env.set_enabled(True)
    fallback = {"attention_state": "UNKNOWN", "is_distracted": False, "focus_score": None}
    diag_env.log_frame(fallback)
    frame_rec = diag_env.records[-1]
    assert frame_rec["attention_state"] == "UNKNOWN"
    assert frame_rec["is_distracted"] is False, "帧级日志保留 analyzer 原始兜底值，不丢失信息"
    # 裁决级：final=None，与帧级原始 False 明确区分
    assert _resolve_distraction(fallback, None)[0] is None


def test_fix_diag_jsonl_keeps_none_csv_not_zero(diag_env):
    """JSONL 保留 null；CSV 中 None→空串，不得转成 0/False"""
    import csv
    import io
    diag_env.set_enabled(True)
    diag_env.log_decision({
        "decision_source": "unknown",
        "final_is_distracted": None,
        "analyzer_is_distracted": None,
        "legacy_fallback_used": False,
    })
    from app.diagnostics import DECISION_FIELDS
    # JSONL：None 原样写 null
    diag_env.fmt = "jsonl"
    jpath = diag_env.export(kind="decision", fmt="jsonl")
    with open(jpath, "r", encoding="utf-8") as fh:
        jcontent = fh.read()
    assert '"final_is_distracted": null' in jcontent
    assert '"analyzer_is_distracted": null' in jcontent
    # CSV：None→空串，bool False→"0"（None 字段不得变 0/False）
    diag_env.fmt = "csv"
    cpath = diag_env.export(kind="decision", fmt="csv")
    with open(cpath, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    row = rows[-1]
    assert row["final_is_distracted"] == "", "CSV 中 UNKNOWN 的 final 必须为空，不得为 0/False"
    assert row["analyzer_is_distracted"] == "", "CSV 中 UNKNOWN 的 analyzer 值必须为空"
    assert "final_is_distracted" in DECISION_FIELDS
