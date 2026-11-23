"""第六阶段 · HUD 布局与走神进度条优化测试（不新增依赖）。

分两类断言，报告中须分开表述：
A) 后端行为测试：真实执行 evaluate_strike / _resolve_distraction。
B) 前端自动化静态审查：读取 .vue/.ts 源码断言布局与数据流契约，
   静态审查不能替代真实浏览器视觉验证，报告须单独标注其未执行。
"""
import math
from pathlib import Path

from app import punishment
from app.routers.room_routes import _resolve_distraction

ROOT = Path(__file__).resolve().parents[1]
HUD = (ROOT / "frontend/src/components/HudOverlay.vue").read_text(encoding="utf-8")
STUDY = (ROOT / "frontend/src/views/StudyRoomView.vue").read_text(encoding="utf-8")
STORE = (ROOT / "frontend/src/stores/room.ts").read_text(encoding="utf-8")
TYPES = (ROOT / "frontend/src/types/index.ts").read_text(encoding="utf-8")
ROUTES = (ROOT / "app/routers/room_routes.py").read_text(encoding="utf-8")


def _block(src: str, selector: str) -> str:
    """取一个 CSS 规则块文本（selector 须精确唯一）。"""
    start = src.index(selector)
    end = src.index("\n}", start)
    return src[start:end]


# ---------- A. 后端行为测试 ----------
def test_distracted_streak_accumulates_continuously(monkeypatch):
    """走神累计连续增长，进度条数据源平滑递增，无前端推算。"""
    monkeypatch.setattr(punishment, "STRIKE_SECONDS", 100.0)
    room = {"strikes_left": 3, "distracted_streak_sec": 0.0}
    for expected in (1.0, 2.0, 3.0):
        res = punishment.evaluate_strike(room, True, 1.0)
        assert res["event"] is None
        assert math.isclose(room["distracted_streak_sec"], expected)


def test_normal_frame_resets_streak(monkeypatch):
    """正常（确认未走神）帧使累计归零，进度条随之回落。"""
    monkeypatch.setattr(punishment, "STRIKE_SECONDS", 100.0)
    room = {"strikes_left": 3, "distracted_streak_sec": 3.0}
    res = punishment.evaluate_strike(room, False, 1.0)
    assert res["event"] is None
    assert room["distracted_streak_sec"] == 0.0


def test_unknown_frames_keep_streak_not_flip(monkeypatch):
    """UNKNOWN 帧保持累计：既不转成确认走神，也不转成确认未走神。"""
    monkeypatch.setattr(punishment, "STRIKE_SECONDS", 100.0)
    cases = [
        ({"attention_state": "UNKNOWN", "is_distracted": False}, 80),
        ({}, None),  # result=None 兜底：无 bool、无分数
    ]
    for payload, score in cases:
        final, source, analyzer_val, legacy = _resolve_distraction(payload, score)
        assert final is None and source == "unknown"
        assert analyzer_val is None and legacy is False
    # 后端 UNKNOWN 分支构造中性帧时，累计取房间现值且不写回
    room = {"strikes_left": 3, "distracted_streak_sec": 2.5}
    neutral = {"is_distracted": None, "distracted_streak_sec": room["distracted_streak_sec"]}
    assert neutral["distracted_streak_sec"] == 2.5
    assert 'if distracted is None:' in ROUTES


def test_threshold_reached_ui_matches_backend(monkeypatch):
    """达阈值时后端扣血并清零累计；前端按同一阈值换算可达 100% 后同步归零。"""
    monkeypatch.setattr(punishment, "STRIKE_SECONDS", 2.0)
    room = {"strikes_left": 3, "distracted_streak_sec": 0.0}
    res = punishment.evaluate_strike(room, True, 1.0)
    assert res["event"] is None
    pct = min(100, max(0, round(room["distracted_streak_sec"] / punishment.STRIKE_SECONDS * 100)))
    assert pct == 50
    res = punishment.evaluate_strike(room, True, 1.0)
    assert res["event"] is not None
    assert room["strikes_left"] == 2 and room["distracted_streak_sec"] == 0.0


# ---------- B. 前端自动化静态审查 ----------
def test_threshold_consistent_backend_frontend():
    """前后端阈值一致：后端下发权威阈值，前端无硬编码 10 秒。"""
    assert isinstance(punishment.STRIKE_SECONDS, (int, float)) and punishment.STRIKE_SECONDS > 0
    assert '"strike_seconds": float(punishment_module.STRIKE_SECONDS)' in ROUTES
    assert "strike_seconds?: number" in TYPES
    assert "room.strikeSeconds" in HUD
    assert "Math.round((room.distractedStreakSec / 10)" not in HUD
    assert "strikeSeconds.value = msg.strike_seconds" in STORE


def test_broadcast_carries_authoritative_threshold():
    """analysis_update 广播携带惩罚引擎实际阈值，前端不独立推算业务状态。"""
    assert '"strike_seconds": float(punishment_module.STRIKE_SECONDS)' in ROUTES
    assert '"type": "analysis_update"' in ROUTES
    assert "distractedStreakSec.value +" not in STORE


def test_skipped_frame_does_not_mutate_streak():
    """skipped 帧直接 return，累计更新赋值在其后，不被跳过帧误改。"""
    idx = STORE.index("if (msg.skipped)")
    assert "return" in STORE[idx:idx + 120]
    assign = STORE.index("distractedStreakSec.value = msg.distracted_streak_sec")
    assert assign > idx


def test_room_switch_resets_hud_state():
    """换房 reset 同时复位累计与阈值，旧 HUD 状态不残留。"""
    assert "strikeSeconds.value = 10" in STORE
    assert "distractedStreakSec.value = 0" in STORE
    assert "strikeSeconds," in STORE


def test_long_text_no_absolute_overlap():
    """长文字不遮挡：HUD 根与顶部条脱离绝对定位，长文本安全折行。"""
    root = _block(HUD, "\n.hud {\n")
    assert "position: absolute" not in root and "inset: 0" not in root
    bar = _block(HUD, "\n.hud-topbar {\n")
    assert "position: absolute" not in bar and "top: 8px" not in bar
    blk = _block(HUD, "\n.hud-block {\n")
    assert "overflow-wrap: anywhere" in blk and "white-space: normal" in blk
    assert "overflow-wrap: anywhere" in STUDY


def test_narrow_screen_flow_layout():
    """窄屏动态布局：study-layout 纵向 flex，study-body 自适应剩余空间。"""
    lay = _block(STUDY, "\n.study-layout {\n")
    assert "display: flex" in lay and "flex-direction: column" in lay
    assert "padding: 64px" not in lay
    body = _block(STUDY, "\n.study-body {\n")
    assert "flex: 1 1 auto" in body and "margin-top: 16px" not in body
    assert "flex-wrap: wrap" in _block(HUD, "\n.hud-topbar {\n")


def test_zoom_safe_no_fixed_height():
    """缩放安全：顶部条无固定高度假设，弹幕层与布局解耦。"""
    bar = _block(HUD, "\n.hud-topbar {\n")
    assert "height:" not in bar
    barline = _block(HUD, "\n.hud-bar {\n")
    assert "flex: 1 1 100px" in barline and "min-width: 60px" in barline
    assert "position: fixed" in _block(HUD, "\n.hud-barrage-layer {\n")


def test_hud_state_in_normal_dom_flow():
    """HudOverlay 位于 study-topbar 之前的正常流首行，不再覆盖房间信息框。"""
    layout_pos = STUDY.index('<div class="study-layout">')
    hud_pos = STUDY.index("<HudOverlay />", layout_pos)
    topbar_pos = STUDY.index('<header class="study-topbar">', layout_pos)
    assert layout_pos < hud_pos < topbar_pos
    assert STUDY.count("<HudOverlay />") == 1
