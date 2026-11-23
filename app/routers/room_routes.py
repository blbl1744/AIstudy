"""
路由层 · 督学房间管理与统一房间 WebSocket 网关（M1-B）。

对外提供：
- POST /api/rooms                     登录用户创建房间
- GET  /api/rooms/plaza               公告广场（公开且非单机房间）
- GET  /api/rooms/{room_id}/snapshot.jpg  房主最新快照（鉴权）
- WS   /ws/room/{room_id}?token=...   统一督学房间网关

设计要点：
1. 房间状态存于 state.active_rooms，连接存于 state.room_connections，
   二者均为进程内唯一真源，房间 TTL 由 janitor 统一回收；
2. 推理走共享线程池 state._INFER_POOL，背压键为学习者 user_id，
   推理完成后经事件循环回写行为记录、广播分析结果并判定哨兵预警；
3. 哨兵阈值与时钟均为模块级可替换对象，便于 pytest 用可控时钟压缩长时延；
4. 所有注释使用中文，本模块不落盘任何报告文件。
"""
import asyncio
import base64
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from .. import punishment as punishment_module
from .. import state
from ..ai_commander import stream_commander_dialogue
from ..analyzer import analyzer
from ..auth import get_current_user, verify_jwt_token
from ..config import CORS_ALLOW_ORIGINS, logger
from ..db import calc_rank_title, get_db, save_behavior_record
from ..diagnostics import diagnostics
from ..punishment import (
    evaluate_strike,
    verify_punishment_submission,
    verify_punishment_with_llm,
)
from ..scene_generator import compile_scene, list_presets

router = APIRouter()

# ===== 哨兵预警参数（模块级常量，测试可 monkeypatch 压缩阈值）=====
SENTINEL_ALERT_SEC = 8.0         # 连续走神达到该秒数触发预警
SENTINEL_COOLDOWN_SEC = 30.0     # 同一房间两次预警的最小间隔（防刷屏）
DISTRACTED_SCORE = 60            # 缺 is_distracted 时的旧阈值：focus_score 低于该值视为走神；score=None 归 UNKNOWN
SOUND_WHITELIST = {"knock_board", "clear_throat", "alarm", "cheer"}

# ===== 建设性惩罚提示文案（三次机会耗尽时下发给学习者）=====
_PUNISHMENT_PROMPT = {
    "feynman_recall": "任务失守：请用费曼技巧把本轮目标讲清楚（不少于15字有效内容）以解除锁定。",
    "emergency_sprint": "任务失守：请保持专注冲刺满 5 分钟后提交，以解除锁定。",
    "goal_breakdown": "任务失守：请把本轮目标拆解为至少 2 个具体微行动步骤后提交，以解除锁定。",
}


def _punishment_prompt(ptype: str) -> str:
    """按惩罚类型返回解锁提示（非法类型回退费曼输出文案）"""
    return _PUNISHMENT_PROMPT.get(ptype, _PUNISHMENT_PROMPT["feynman_recall"])


def _resolve_distraction(payload, score):
    """确定最终走神裁决及其来源（纯函数，三态裁决）。

    返回四元组 (final, decision_source, analyzer_is_distracted, legacy_fallback_used)：
    裁决优先级（自高而低，最终修复统一）：
      1) analyzer 显式 attention_state == "UNKNOWN"（引擎降级/关键点缺失兜底）→ 一律 UNKNOWN，
         无论 is_distracted 是 True/False，绝不把兜底伪 False 当「确认未走神」（否则会清零累计）；
      2) 有效 bool 型 is_distracted → 采用该值（True 确认走神 / False 确认未走神），
         source=analyzer，analyzer_is_distracted=该值，legacy_fallback_used=False；
      3) 无有效 bool 但 score 是有效数值 → 沿用旧阈值 score < DISTRACTED_SCORE，
         source=legacy_fallback，analyzer_is_distracted=None，legacy_fallback_used=True；
      4) 以上均不可用（score=None 且无有效 bool，典型：线程池推理异常 result=None）→ UNKNOWN
         (None, "unknown", None, False)。

    UNKNOWN 两种情况（显式 attention_state=UNKNOWN 与 result=None）统一返回
    (None, "unknown", None, False)：analyzer_is_distracted 按字段定义记 None（未经有效性确认），
    analyzer 原始兜底值不在此字段体现，而在帧级诊断 log_frame 按原始输出保留，二者可区分且不丢信息。
    三态语义：True=确认走神 / False=确认未走神 / None=UNKNOWN（跳过惩罚、不动任何状态）。

    该函数只做「读 payload + 判定来源」的纯计算，不写库、不落盘、不改变任何状态。
    """
    # 第一优先：显式 UNKNOWN 必须先看 attention_state，再看 is_distracted。
    # 兜底帧的 is_distracted=False 是硬编码伪值，命中 bool 分支会误判「确认未走神」并清零累计。
    if payload.get("attention_state") == "UNKNOWN":
        return None, "unknown", None, False
    if isinstance(payload.get("is_distracted"), bool):
        final = payload["is_distracted"]
        return final, "analyzer", final, False
    if score is None:
        # 缺失 bool 且无分数：无法判断，中性 UNKNOWN，绝不回退为「确认走神」或「确认未走神」
        return None, "unknown", None, False
    legacy = score < DISTRACTED_SCORE
    return legacy, "legacy_fallback", None, True


# WebSocket 握手失败关闭码
WS_CLOSE_BAD_TOKEN = 4401        # JWT 缺失或无效
WS_CLOSE_FORBIDDEN = 4403        # Origin 非法 / 私密房间 / 房间满员
WS_CLOSE_NOT_FOUND = 4404        # 房间不存在


def _now_ts() -> float:
    """可控时钟：默认单调时钟，测试内可 monkeypatch 推进虚拟时间"""
    return time.monotonic()


# ========== 请求模型 ==========
class CreateRoomRequest(BaseModel):
    """创建房间请求：对目标文本做长度与 XSS 双重约束"""
    mode: Literal["ai_solo", "peer_1v1", "peer_group"]
    is_public: bool = True
    scene_id: str = ""
    work_name: str = ""
    scene_title: str = ""
    goal_text: str = Field(..., min_length=1, max_length=100)
    max_watchers: int = Field(default=3, ge=1, le=20)
    punishment_types: list[str] = Field(default_factory=list)

    @field_validator("goal_text")
    @classmethod
    def _validate_goal(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError("学习目标不能为空")
        # 防 XSS：拒绝常见危险字符，杜绝存储型脚本注入
        if any(ch in text for ch in "<>&\"'"):
            raise ValueError("学习目标包含非法字符")
        return text

    @field_validator("scene_title")
    @classmethod
    def _validate_scene_title(cls, v: str) -> str:
        text = (v or "").strip()
        if any(ch in text for ch in "<>&\"'"):
            raise ValueError("场景标题包含非法字符")
        return text


class PunishmentSubmitRequest(BaseModel):
    """惩罚提交 HTTP 兜底请求：字段与 WS punishment_submit 严格对齐"""
    punishment_type: str = ""
    content: str = ""


def _current_user_id(user: dict) -> int:
    """从 JWT 载荷取统一账号 ID；缺失则视为无效凭证"""
    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="无效的Token")
    return int(user_id)


async def _resolve_scene_config(scene_id: str, work_name: str) -> dict:
    """异步解析房间剧本：work_name 调编译器，scene_id 匹配内置预设，均无则空字典"""
    work_name = (work_name or "").strip()
    if work_name:
        try:
            return await compile_scene(work_name)
        except Exception as exc:
            logger.warning("房间剧本编译失败，回退空剧本：%s", exc)
            return {}
    sid = (scene_id or "").strip()
    if sid:
        for preset in list_presets():
            if preset.get("scene_id") == sid:
                return preset
    return {}


# ========== REST：创建房间 ==========
@router.post("/api/rooms")
async def create_room(request: CreateRoomRequest, user: dict = Depends(get_current_user)):
    """登录用户创建房间；goal_text 已由模型完成长度与 XSS 校验"""
    user_id = _current_user_id(user)
    room_id = uuid.uuid4().hex
    # M2-B：解析剧本——work_name 优先调编译器；否则按 scene_id 匹配预设；
    # 两者都取不到时留空字典，前端仍可正常渲染（对话指挥官会回退通用称谓）。
    scene_config = await _resolve_scene_config(request.scene_id, request.work_name)
    room = {
        "room_id": room_id,
        "host_id": user_id,
        "host_nickname": user.get("nickname", ""),
        "mode": request.mode,
        "is_public": request.is_public,
        "scene_id": request.scene_id,
        "work_name": (request.work_name or "").strip(),
        "scene_title": request.scene_title,
        "goal_text": request.goal_text,
        "max_watchers": request.max_watchers,
        "punishment_types": list(request.punishment_types),
        "created_at": _now_ts(),
        "strikes_left": 3,
        "distracted_streak_sec": 0.0,
        # ===== M2-B：剧本与流式对话状态 =====
        "scene_config": scene_config,          # 当前房间挂载的完整 SceneConfig
        "dialogue_history": [],                # 最近 8 轮指挥官对话
        "is_locked": False,                    # 惩罚锁房标记（锁房时忽略常规推理）
        "strikes_used": 0,                     # 本局累计扣血次数（如实写战报）
        "active_punishment": "",               # 当前触发的建设性惩罚类型
        "punishment_type": "",                 # evaluate_strike 回写的惩罚类型（供其读取）
        "last_frame_ts": None,
        "last_frame_b64": None,
        # 独立的“画面到达”时间戳：专供快照新鲜度/过期识别，不复用 last_frame_ts
        # （后者被推理帧间隔与哨兵走神累计依赖，属冻结逻辑不可扰动）
        "last_frame_received_ts": None,
        "score_history": [],
        "watchers": {},
        # 房主在线标记：暂离时置 False 仅保留房间，结算时才真正销毁
        "host_online": True,
        # 哨兵上次预警时间（内部字段，不对外回显）
        "_last_sentinel_alert_ts": 0.0,
    }
    state.active_rooms.set(room_id, room)
    # 回显房间信息，剔除内部下划线字段
    return {k: v for k, v in room.items() if not k.startswith("_")}


# ========== REST：公告广场 ==========
@router.get("/api/rooms/plaza")
async def plaza(user: dict = Depends(get_current_user)):
    """公开广场快照：本人房间无论模式/公开/在线状态一律可见（供暂离返回），他人仅公开非单机房"""
    user_id = _current_user_id(user)  # 鉴权并取当前用户 id
    now = _now_ts()
    result = []
    for _, room in state.active_rooms.items():
        if not isinstance(room, dict):
            continue
        is_mine = room.get("host_id") == user_id
        # 他人房间：仅展示公开且非 ai_solo 的；本人房间无条件保留（含 ai_solo / 私密 / 暂离）
        if not is_mine and (not room.get("is_public") or room.get("mode") == "ai_solo"):
            continue
        result.append({
            "room_id": room.get("room_id"),
            "host_id": room.get("host_id"),
            "host_nickname": room.get("host_nickname", ""),
            "mode": room.get("mode"),
            "scene_title": room.get("scene_title", ""),
            "goal_text": room.get("goal_text", ""),
            "elapsed_seconds": max(0, int(now - (room.get("created_at") or now))),
            "watcher_count": len(room.get("watchers") or {}),
            "max_watchers": room.get("max_watchers", 0),
            "host_online": bool(room.get("host_online", True)),
        })
    return {"rooms": result}


# ========== REST：房主快照 ==========
@router.get("/api/rooms/{room_id}/snapshot.jpg")
async def snapshot(room_id: str, user: dict = Depends(get_current_user)):
    """鉴权获取房主最新 image/jpeg 快照；房间/画面不存在 404，私密房间非本人 403"""
    user_id = _current_user_id(user)
    room = state.active_rooms.get(room_id)
    if not isinstance(room, dict):
        raise HTTPException(status_code=404, detail="房间不存在")
    if not room.get("is_public") and room.get("host_id") != user_id:
        raise HTTPException(status_code=403, detail="无权访问私密房间")
    frame = room.get("last_frame_b64")
    if not frame:
        raise HTTPException(status_code=404, detail="暂无画面")
    try:
        content = base64.b64decode(frame, validate=True)
    except Exception:
        raise HTTPException(status_code=404, detail="画面数据无效")
    if not content:
        raise HTTPException(status_code=404, detail="暂无画面")
    # 回传画面到达时间戳（供观看端识别“房主已停推/快照过期”），并禁用缓存
    # 无画面时 last_frame_received_ts 为 None，回 0 表示从未发布
    frame_ts = room.get("last_frame_received_ts")
    if frame_ts is None:
        frame_ts = room.get("last_frame_ts") or 0
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"X-Frame-Ts": str(frame_ts), "Cache-Control": "no-store"},
    )


# ========== REST：房间元信息（督学官进入校验归属） ==========
@router.get("/api/rooms/{room_id}")
async def room_meta(room_id: str, user: dict = Depends(get_current_user)):
    """返回房间元信息与现场快照：供督学官校验归属，也供房主暂离后恢复现场；不存在返回 404"""
    room = state.active_rooms.get(room_id)
    if not isinstance(room, dict):
        raise HTTPException(status_code=404, detail="房间不存在")
    return {
        "room_id": room.get("room_id"),
        "host_id": room.get("host_id"),
        "host_nickname": room.get("host_nickname", ""),
        "scene_title": room.get("scene_title", ""),
        "goal_text": room.get("goal_text", ""),
        "mode": room.get("mode"),
        "is_public": room.get("is_public"),
        # 暂离/返回闭环所需现场字段
        "host_online": bool(room.get("host_online", True)),
        "scene_config": room.get("scene_config", {}),
        "strikes_left": room.get("strikes_left", 3),
        "is_locked": bool(room.get("is_locked", False)),
        "active_punishment": room.get("active_punishment", ""),
    }


@router.post("/api/rooms/{room_id}/punishment/submit")
async def punishment_submit_http(
    room_id: str, request: PunishmentSubmitRequest,
    user: dict = Depends(get_current_user),
):
    """WS 未连接时的惩罚提交兜底：复用统一校验与解锁逻辑，同步房间状态并全房推 punishment_cleared。

    通过/拒绝均以 200 回执结构化结果，绝不因语义服务不可用挂起请求。
    """
    user_id = _current_user_id(user)
    room = state.active_rooms.get(room_id)
    if not isinstance(room, dict):
        raise HTTPException(status_code=404, detail="房间不存在")
    if user_id != room.get("host_id"):
        raise HTTPException(status_code=403, detail="仅房主可提交惩罚解锁")
    verdict = await _run_punishment_verdict(
        room, request.punishment_type, request.content,
        str(room.get("goal_text", "") or ""),
    )
    if verdict["passed"]:
        await _broadcast(room_id, {
            "type": "punishment_cleared",
            "shield_saved": False,
            "strikes_left": room.get("strikes_left", 3),
        })
    return {
        "passed": bool(verdict["passed"]),
        "reason": verdict.get("reason", ""),
        "strikes_left": room.get("strikes_left", 3),
    }


# ========== 广播工具 ==========
async def _broadcast(room_id: str, payload: dict, exclude_user_id: int | None = None) -> None:
    """向房间内全部连接广播消息（可选排除某用户）"""
    with state.room_connections_lock:
        conns = list(state.room_connections.get(room_id, {}).items())
    for uid, ws in conns:
        if exclude_user_id is not None and uid == exclude_user_id:
            continue
        try:
            await ws.send_json(payload)
        except Exception:
            logger.debug("广播消息到用户 %s 失败", uid)


async def _send_to(room_id: str, user_id: int, payload: dict) -> None:
    """向房间内指定用户单播消息"""
    with state.room_connections_lock:
        ws = state.room_connections.get(room_id, {}).get(user_id)
    if ws is None:
        return
    try:
        await ws.send_json(payload)
    except Exception:
        logger.debug("发送消息到用户 %s 失败", user_id)


async def _notify_watcher_update(room_id: str, room: dict) -> None:
    """向房主广播当前督学官人数与名单"""
    watchers = room.get("watchers") or {}
    await _send_to(room_id, room.get("host_id"), {
        "type": "watcher_update",
        "watcher_count": len(watchers),
        "watchers": [{"user_id": uid, "nickname": name} for uid, name in watchers.items()],
    })


async def _run_punishment_verdict(room: dict, punishment_type: str, content, goal_text: str) -> dict:
    """惩罚提交统一入口：复用带 8 秒超时降级的校验解锁函数，异常也必有回执。"""
    try:
        return await verify_punishment_with_llm(
            punishment_type, content, goal_text, room=room, timeout=8.0,
        )
    except Exception:
        logger.exception("惩罚校验解锁内部异常，按未通过回执处理")
        return {
            "passed": False,
            "reason": "服务处理异常，请重新提交",
            "punishment_type": punishment_type,
            "unlocked": False,
        }


async def _broadcast_dialogue(room_id: str, room: dict, trigger: str,
                              user_input: str = "", target_user_id=None) -> None:
    """驱动一轮指挥官流式对话：先发 start，逐块广播 chunk/audio，最后发 end。

    target_user_id 为 None 时广播全房；指定时仅单播给目标用户（如 reply 仅回求助者）。
    对话文本落库由 ai_commander 负责，本函数只负责 WS 消息编排。
    """
    await _broadcast(room_id, {"type": "ai_dialogue_start", "trigger": trigger})
    try:
        async for piece in stream_commander_dialogue(room, trigger, user_input):
            if target_user_id is None:
                await _broadcast(room_id, piece)
            else:
                await _send_to(room_id, target_user_id, piece)
    except Exception as exc:
        logger.warning("指挥官对话生成异常（room_id=%s）：%s", room_id, exc)
    await _broadcast(room_id, {"type": "ai_dialogue_end", "trigger": trigger})


# ========== 推理回调：事件循环内回写状态与广播 ==========
async def _on_frame_done(room_id: str, user_id: int, image_b64: str, result) -> None:
    """推理完成后的统一收尾：写行为记录、更新分数/走神时长、广播结果并判定哨兵"""
    state._inflight_students.discard(user_id)
    room = state.active_rooms.get(room_id)
    if not isinstance(room, dict):
        return

    payload = result if isinstance(result, dict) else {}
    score = payload.get("focus_score")

    # 1) 行为采样落库（记录关联房间，便于战报回溯）
    try:
        save_behavior_record(user_id, {
            "focus_score": score,
            "distraction_count": payload.get("distraction_count"),
            "head_pose": payload.get("head_pose"),
            "body_stability": payload.get("body_stability"),
            "fatigue_level": payload.get("fatigue_level"),
            "eye_focus": payload.get("eye_focus"),
            "suggestion": payload.get("suggestion"),
            "model_prediction": payload.get("model_prediction"),
        }, session_id=room_id)
    except Exception:
        logger.exception("行为采样写入失败（room_id=%s）", room_id)

    # 2) 更新分数历史并计算帧间隔（时钟可注入，便于单测压缩长时延）
    room.setdefault("score_history", []).append(score)
    now = _now_ts()
    prev = room.get("last_frame_ts")
    delta = (now - prev) if prev is not None else 0.0
    room["last_frame_ts"] = now
    # 本次姿态算法迭代：走神判定优先采用 analyzer 的「多信号综合 + 防抖」结果 is_distracted，
    # 使正常低头学习（HEAD_DOWN）不再因 focus_score 波动或无检测被误判为走神、累计扣血。
    # 兼容回退：若上游 analyzer 未提供该字段（旧版本/第三方实现），沿用原有分数阈值逻辑，
    # 保证 WS 消息结构与既有行为零漂移。
    # 裁决来源追踪（第二阶段补充审查）：_resolve_distraction 三态裁决——
    # analyzer 有效 bool 优先；缺 bool 但有数值走 DISTRACTED_SCORE 阈值兼容；
    # 两者皆缺（analyzer 推理异常 result=None）返回 UNKNOWN，不再误判走神。
    # 诊断关闭时 log_decision 立即返回，零落盘、零状态变更。
    distracted, decision_source, analyzer_is_distracted, legacy_fallback_used = _resolve_distraction(payload, score)
    diagnostics.log_decision({
        "decision_source": decision_source,
        "final_is_distracted": distracted,
        "analyzer_is_distracted": analyzer_is_distracted,
        "legacy_fallback_used": legacy_fallback_used,
    })

    # 2.5) 惩罚引擎统一裁决：内部维护走神累计（走神累加、端正清零）、扣血与锁房状态
    # UNKNOWN（analyzer 推理异常 result=None）：不做惩罚裁决——传 False 会清零连续走神累计、
    # 恢复护盾并改变房间状态，传 True 更会误扣血。这里跳过 evaluate_strike，构造中性结果：
    # event=None 不触发任何惩罚联动，streak 取房间当前值且完全不写回 room；
    # 后续行为记录保存、诊断日志、WS 广播与哨兵判定全部照常执行，仅惩罚状态机保持原样。
    if distracted is None:
        strike = {
            "event": None,
            "strikes_left": room.get("strikes_left", 3),
            "shield_saved": False,
            "punishment_type": room.get("punishment_type", ""),
            "distracted_streak_sec": float(room.get("distracted_streak_sec", 0.0) or 0.0),
        }
    else:
        strike = evaluate_strike(room, distracted, delta)
    streak_sec = float(strike.get("distracted_streak_sec") or 0.0)

    # 3) 分析结果广播给全房（生命值与走神时长均取引擎回写后的最新值）
    await _broadcast(room_id, {
        "type": "analysis_update",
        "focus_score": score,
        "action_label": payload.get("action_label"),
        "annotated_image": payload.get("annotated_image"),
        "strikes_left": room.get("strikes_left", 3),
        "distracted_streak_sec": round(streak_sec, 2),
        # 第六阶段：下发惩罚引擎实际阈值（可被环境变量/测试覆盖），
        # 前端进度条据此换算，杜绝硬编码 10 秒导致的前后端阈值漂移。
        "strike_seconds": float(punishment_module.STRIKE_SECONDS),
    })

    # 3.5) 惩罚联动：按引擎事件累加真实扣血次数并广播战术纠偏/惩罚消息
    event = strike.get("event")
    if event in ("strike_1", "strike_2"):
        room["strikes_used"] = int(room.get("strikes_used", 0) or 0) + 1
        await _broadcast_dialogue(room_id, room, "distraction")
    elif event in ("strike_3", "shield_saved"):
        # 第 3 次耗尽：无论护盾抵消与否，都记一次真实扣血
        room["strikes_used"] = int(room.get("strikes_used", 0) or 0) + 1
        if strike.get("shield_saved"):
            await _broadcast(room_id, {
                "type": "punishment_cleared",
                "shield_saved": True,
                "strikes_left": room.get("strikes_left", 1),
            })
        else:
            ptype = strike.get("punishment_type") or ""
            room["active_punishment"] = ptype
            await _broadcast(room_id, {
                "type": "punishment_required",
                "punishment_type": ptype,
                "prompt": _punishment_prompt(ptype),
            })

    # 4) 哨兵预警：有督学官在线、连续走神达标且已过冷却期时，仅向督学官推送。
    # 最终修复（哨兵当前帧有效性约束）：必须【当前帧最终裁决明确为 True】才允许新的哨兵预警。
    # 核心=只用当前帧确认走神这一有效证据触发，不得借用历史累计：
    #   - UNKNOWN（引擎降级兜底 attention_state=UNKNOWN / 线程池异常 result=None）时
    #     distracted is None，不满足 is True，故不触发新预警；且 UNKNOWN 帧走中性 strike，
    #     完全不写回 room，distracted_streak_sec 历史累计不清零、不增长，原样保留；
    #   - 确认未走神（False）也不触发，且惩罚引擎会清零累计（既有正常逻辑不变）；
    #   - 恢复有效走神（True）后从保留的累计继续按正常规则判定，可正常触发预警。
    # 复用当前帧裁决值 distracted 与既有冷却戳 _last_sentinel_alert_ts 即可实现，
    # 无需新增独立哨兵有效性状态，也不复用/篡改惩罚计数器的写回流程。
    watchers = room.get("watchers") or {}
    last_alert = room.get("_last_sentinel_alert_ts") or 0.0
    if distracted is True and watchers and room["distracted_streak_sec"] >= SENTINEL_ALERT_SEC \
            and (now - last_alert) >= SENTINEL_COOLDOWN_SEC:
        room["_last_sentinel_alert_ts"] = now
        for wid in list(watchers.keys()):
            await _send_to(room_id, wid, {
                "type": "sentinel_alert",
                "message": "学习者已连续走神，请及时提醒",
                "distracted_streak_sec": round(room["distracted_streak_sec"], 2),
            })


# ========== 结算：写战报并更新成长字段 ==========
def _settle_session(room: dict, user_id: int, duration_minutes: int,
                    avg_score, strikes_used: int) -> dict:
    """结束一局：写 study_sessions(completed) 并累加专注时长、军衔、干净局连击与护盾"""
    # 惩罚类型如实记录：优先本局实际触发的类型，未触发时回退开局勾选组合
    actual_ptype = room.get("active_punishment") or room.get("punishment_type") or ""
    punishment = actual_ptype or ",".join(room.get("punishment_types") or [])
    with get_db() as conn:
        conn.execute(
            "INSERT INTO study_sessions "
            "(user_id, mode, scene_title, goal_text, duration_minutes, avg_score, "
            " strikes_used, punishment_type, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed')",
            (user_id, room.get("mode", "ai_solo"), room.get("scene_title", ""),
             room.get("goal_text", ""), duration_minutes, avg_score, strikes_used, punishment),
        )
        row = conn.execute(
            "SELECT total_focus_minutes, streak_clean_sessions, shield_count FROM users WHERE id=?",
            (user_id,),
        ).fetchone()
        total = int((row["total_focus_minutes"] if row else 0) or 0) + duration_minutes
        streak = int((row["streak_clean_sessions"] if row else 0) or 0)
        shield = int((row["shield_count"] if row else 0) or 0)
        # 干净局：零罚站且时长达标，连击 +1；每满 3 局奖励 1 枚护盾
        if strikes_used == 0 and duration_minutes >= 25:
            streak += 1
            if streak % 3 == 0:
                shield += 1
        conn.execute(
            "UPDATE users SET total_focus_minutes=?, rank_title=?, "
            "streak_clean_sessions=?, shield_count=? WHERE id=?",
            (total, calc_rank_title(total), streak, shield, user_id),
        )
    return {
        "total_focus_minutes": total,
        "rank_title": calc_rank_title(total),
        "streak_clean_sessions": streak,
        "shield_count": shield,
    }


# ========== 第三阶段：结束链路 ACK / 幂等 / 清理辅助 ==========
# 结算状态码（写入 session_finish_ack.status，前端据此给出明确反馈）
SETTLE_OK = "success"            # 本次真正完成结算
SETTLE_DUPLICATE = "duplicate"   # 已完成，重复请求命中幂等缓存（不重复扣分/保存/广播）
SETTLE_IN_PROGRESS = "in_progress"  # 并发：别的请求正在结算，等待其完成
SETTLE_DB_FAILED = "db_failed"   # 数据库保存失败（事务已回滚，可重试）
SETTLE_FORBIDDEN = "forbidden"   # 无权：仅房主可结束整房
SETTLE_NOT_FOUND = "not_found"   # 房间不存在（含重复点击/旧请求命中已销毁房间）
SETTLE_INTERNAL = "internal_error"  # 后端内部异常


async def _send_finish_ack(room_id: str, user_id: int, status: str,
                           message: str = "", result: dict | None = None,
                           request_id=None) -> None:
    """向发起结束请求的用户单播结算回执（ACK）。

    status 语义与 SETTLE_* 常量一致；result 仅在 success/duplicate 时携带战报。
    request_id 原样回传，便于前端把回执与某次点击关联（旧客户端忽略该字段即可，兼容）。
    ACK 发送失败只记日志，不影响结算主流程（结算状态已由 settlement_records 定案）。
    """
    payload = {
        "type": "session_finish_ack",
        "status": status,
        "message": message,
        "request_id": request_id,
        "room_id": room_id,
    }
    if result is not None:
        payload["result"] = result
    try:
        await _send_to(room_id, user_id, payload)
    except Exception:
        logger.exception("结算 ACK 发送失败（room=%s user=%s）", room_id, user_id)


def _cleanup_finished_room(room_id: str, room: dict) -> bool:
    """幂等清理房间全部资源：仅当 active_rooms 中仍是目标房间对象时才销毁，返回是否实际清理。

    身份守卫确保：并发/重复/旧请求不会误清新房间；重复调用安全（第二次返回 False）。
    """
    if state.active_rooms.get(room_id) is not room:
        return False
    state.active_rooms.pop(room_id, None)
    with state.room_connections_lock:
        state.room_connections.pop(room_id, None)
    host_id = room.get("host_id")
    if host_id is not None:
        state._inflight_students.discard(host_id)
    # 锁随房间销毁一并回收，避免长期驻留（新请求若再来会按需新建）
    state.purge_settlement_lock(room_id)
    return True


def _build_settle_result(duration_minutes: int, avg_score, settled: dict) -> dict:
    """组装战报结果（与既有 session_finished 字段保持一致）"""
    return {
        "duration_minutes": duration_minutes,
        "avg_score": avg_score,
        "total_focus_minutes": settled.get("total_focus_minutes", 0),
        "rank_title": settled.get("rank_title", ""),
        "streak_clean_sessions": settled.get("streak_clean_sessions", 0),
        "shield_count": settled.get("shield_count", 0),
    }


async def _settle_room_once(room_id: str, room: dict, user_id: int,
                            request_id=None, broadcast: bool = True) -> dict:
    """幂等结算核心：同一房间只实际结算一次，返回统一结构的结果描述。

    返回 dict: {"status": SETTLE_*, "message": str, "result": dict|None}
    并发控制：per-room asyncio.Lock 内完成「查记录→写库→落定状态」，只有一个协程真正写库；
    数据保护：DB 事务（get_db 上下文）保证写库异常整体回滚，失败状态可安全重试不产生重复行；
    幂等命中：status=done 直接返回缓存战报，不重复扣分/保存/广播（只有 broadcast 分支首次广播）。
    """
    lock = state.get_settlement_lock(room_id)
    async with lock:
        cached = state.settlement_records.get(room_id)
        if cached and cached.get("status") == SETTLE_OK:
            # 已完成：直接回放既有战报，绝不重复结算
            logger.info("结算幂等命中：房间 %s 已完成，跳过重复结算（request_id=%s）", room_id, request_id)
            return {"status": SETTLE_DUPLICATE,
                    "message": "本局已结算，已返回原有战报",
                    "result": cached.get("result")}

        # 计算本局指标（在锁内基于房间对象快照，避免并发修改）
        now = _now_ts()
        created = float(room.get("created_at") or now)
        duration_minutes = int(max(0.0, now - created) // 60)
        scores = [s for s in (room.get("score_history") or [])
                  if isinstance(s, (int, float)) and not isinstance(s, bool)]
        avg_score = round(sum(scores) / len(scores)) if scores else None
        strikes_used = int(room.get("strikes_used", 0) or 0)

        # 标记结算中：并发的其它协程若也进到锁，会在上面命中 settling 之外的路径
        state.settlement_records.set(room_id, {
            "status": SETTLE_IN_PROGRESS, "request_id": request_id,
        })
        try:
            settled = _settle_session(room, user_id, duration_minutes, avg_score, strikes_used)
        except Exception as exc:
            # 事务已回滚；落 failed 状态允许后续重试，不残留半成品数据
            state.settlement_records.set(room_id, {
                "status": SETTLE_DB_FAILED, "request_id": request_id, "error": str(exc),
            })
            logger.exception("房间 %s 结算写库失败（request_id=%s）", room_id, request_id)
            return {"status": SETTLE_DB_FAILED,
                    "message": "结算保存失败，请稍后重试（本局未重复扣分）",
                    "result": None}

        result = _build_settle_result(duration_minutes, avg_score, settled)
        state.settlement_records.set(room_id, {
            "status": SETTLE_OK, "request_id": request_id, "result": result,
        })

    # 锁外广播：仅首次真正结算时通知全体；重复请求不会走到这里（上面已 return duplicate）
    if broadcast:
        try:
            await _broadcast(room_id, {"type": "session_finished", **result})
        except Exception:
            logger.exception("session_finished 广播失败（room=%s），结算已完成不受影响", room_id)
    return {"status": SETTLE_OK, "message": "结算成功", "result": result}


# ========== 统一房间 WebSocket 网关 ==========
@router.websocket("/ws/room/{room_id}")
async def room_ws(websocket: WebSocket, room_id: str):
    """统一房间网关：JWT / Origin / 房间存在三级握手校验 + 双角色消息分发"""
    # 1) JWT 校验（从 query 取 token）
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=WS_CLOSE_BAD_TOKEN)
        return
    try:
        user = verify_jwt_token(token)
        user_id = int(user["user_id"])
        nickname = user.get("nickname", "")
    except Exception:
        await websocket.close(code=WS_CLOSE_BAD_TOKEN)
        return

    # 2) Origin 校验：仅当客户端带 Origin 且不在白名单时拒绝
    origin = websocket.headers.get("origin")
    if origin and origin not in CORS_ALLOW_ORIGINS:
        await websocket.close(code=WS_CLOSE_FORBIDDEN)
        return

    # 3) 房间存在校验
    room = state.active_rooms.get(room_id)
    if not isinstance(room, dict):
        await websocket.close(code=WS_CLOSE_NOT_FOUND)
        return

    is_host = user_id == room.get("host_id")
    if not is_host:
        # 督学官：私密房间或满员一律拒绝
        if not room.get("is_public") or len(room.get("watchers") or {}) >= int(room.get("max_watchers", 0)):
            await websocket.close(code=WS_CLOSE_FORBIDDEN)
            return

    await websocket.accept()
    with state.room_connections_lock:
        state.room_connections.setdefault(room_id, {})[user_id] = websocket
    if not is_host:
        room.setdefault("watchers", {})[user_id] = nickname
    # 连接建立后通知房主当前督学官人数
    await _notify_watcher_update(room_id, room)
    loop = asyncio.get_running_loop()

    if is_host:
        # 房主连入即视为在线（暂离重连会由此恢复在线标记）
        room["host_online"] = True
        # 下发当前发布态：房主刷新/重连后以服务端现场校准，不凭本地假设“已发布”
        await _send_to(room_id, user_id, {
            "type": "stream_status",
            "published": bool(room.get("last_frame_b64")),
            "ts": room.get("last_frame_received_ts"),
        })
        # 分流：已开过局（有历史对话或已发过开局播报）认定为暂离重连，恢复现场而非重复开局
        has_history = bool(room.get("dialogue_history"))
        if room.get("_proactive_sent") or has_history:
            # 向本房主单发现场快照，恢复目标/机会/锁房/惩罚/场景与历史对话
            await _send_to(room_id, user_id, {
                "type": "room_state_sync",
                "goal_text": room.get("goal_text", ""),
                "strikes_left": room.get("strikes_left", 3),
                "is_locked": bool(room.get("is_locked", False)),
                "active_punishment": room.get("active_punishment", ""),
                "scene_config": room.get("scene_config", {}),
                "dialogue_history": room.get("dialogue_history", []),
            })
            # 仅向在线督学官广播房主已返回（排除房主本人）
            await _broadcast(room_id, {"type": "host_status", "online": True},
                             exclude_user_id=user_id)
        else:
            # 首次连入：触发一次 proactive 流式对话（房间级标记防重连重复触发）
            room["_proactive_sent"] = True
            await _broadcast_dialogue(room_id, room, "proactive")

    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")
            try:
                # ===== 房间公共互动消息（房主与督学官均可发；身份取 JWT，绝不信任客户端伪造字段） =====
                # 第五阶段修复⑦：弹幕/音效此前仅在督学官分支处理，房主发出的消息被静默丢弃，
                # 造成双向通信不完整。此处提升为房间级公共消息，并附带服务端生成或原样回传的
                # message_id，供所有端幂等去重（本地回显与广播、重连、重复投递均不重复展示）。
                if mtype in ("barrage", "sound_effect"):
                    _raw_mid = msg.get("message_id")
                    if isinstance(_raw_mid, (str, int)) and str(_raw_mid).strip():
                        message_id = str(_raw_mid).strip()[:128]
                    else:
                        message_id = str(uuid.uuid4())
                    if mtype == "barrage":
                        # 与前端一致的字符清洗：保留换行与可打印字符，strip 后按 1~60 字校验。
                        text = "".join(
                            ch for ch in str(msg.get("text", ""))
                            if ch == "\n" or (ord(ch) >= 32 and ord(ch) != 127)
                        ).strip()
                        if 1 <= len(text) <= 60:
                            # from 恒取 JWT 身份，忽略客户端可能伪造的 sender_id / from；
                            # 另附 user_id 供前端精确识别发送者，不依赖可重名的昵称。
                            await _broadcast(room_id, {
                                "type": "room_interaction", "kind": "barrage",
                                "from": nickname, "user_id": user_id,
                                "text": text, "message_id": message_id,
                            })
                    else:
                        sound = msg.get("sound")
                        if sound in SOUND_WHITELIST:
                            await _broadcast(room_id, {
                                "type": "room_interaction", "kind": "sound_effect",
                                "from": nickname, "user_id": user_id,
                                "sound": sound, "message_id": message_id,
                            })
                    # 公共互动已广播给全房（含发送者本人），不再进入下方角色分支，避免重复处理。
                    continue
                if is_host:
                    # ===== 学习者消息 =====
                    if mtype == "frame":
                        image = msg.get("image")
                        # 房间身份守卫：房间已结束/重建时，旧连接的迟到帧不得写入新房间
                        if state.active_rooms.get(room_id) is not room:
                            await websocket.send_json({"type": "frame_ack", "ok": False, "reason": "stale_room"})
                            continue
                        if not image:
                            # 空帧不更新画面，明确回执失败：区分“消息发出”与“后端确认写入”
                            await websocket.send_json({"type": "frame_ack", "ok": False, "reason": "empty_frame"})
                            continue
                        # 快照画面即时保存：锁房态也更新，保证监督端画面不中断
                        room["last_frame_b64"] = image
                        # 独立的画面到达时间戳，专供快照新鲜度识别；不扰动推理所用的 last_frame_ts
                        room["last_frame_received_ts"] = _now_ts()
                        # 只有后端确实写入有效画面才回 ok=True，房主据此从 PUBLISHING 进入 STREAM_PUBLISHED
                        await websocket.send_json({
                            "type": "frame_ack",
                            "ok": True,
                            "frame_id": msg.get("frame_id"),
                            "ts": room["last_frame_received_ts"],
                            "analysis_skipped": user_id in state._inflight_students,
                        })
                        # 锁房态：仍更新画面，但跳过常规推理，仅下发解锁提示（惩罚提交走独立分支）
                        if room.get("is_locked"):
                            await websocket.send_json({"type": "room_locked", "msg": "请先完成惩罚解锁"})
                            continue
                        # 背压：同一学习者上一帧未算完则丢弃（画面已确认写入，仅跳过推理）
                        if user_id in state._inflight_students:
                            await websocket.send_json({"type": "analysis_update", "skipped": True})
                            continue
                        state._inflight_students.add(user_id)
                        scene = msg.get("scene", "selfstudy")

                        def _done(fut, rid=room_id, uid=user_id, frame=image, app_loop=loop):
                            """推理线程回调：把回写逻辑调度回事件循环执行（默认参数锁定当帧变量）"""
                            try:
                                result = fut.result()
                            except Exception:
                                logger.exception("房间帧推理异常（room_id=%s）", rid)
                                result = None
                            asyncio.run_coroutine_threadsafe(_on_frame_done(rid, uid, frame, result), app_loop)

                        future = state._INFER_POOL.submit(analyzer.analyze, image, scene=scene, student_id=user_id)
                        future.add_done_callback(_done)

                    elif mtype == "user_message":
                        # 学习者求助/互动：1~200 字校验 + 控制字符清洗，触发 chat 流式回复
                        raw = msg.get("text", "")
                        text = "".join(ch for ch in str(raw) if ch == "\n" or (ord(ch) >= 32 and ord(ch) != 127))
                        text = text.strip()
                        if 1 <= len(text) <= 200:
                            await _broadcast_dialogue(room_id, room, "chat", user_input=text)

                    elif mtype == "punishment_submit":
                        # 惩罚提交：统一校验 + 8 秒超时降级；通过/拒绝/异常均有回执，绝不卡住前端
                        try:
                            verdict = await _run_punishment_verdict(
                                room,
                                str(msg.get("punishment_type", "") or ""),
                                msg.get("content", ""),
                                str(room.get("goal_text", "") or ""),
                            )
                            if verdict["passed"]:
                                await _broadcast(room_id, {
                                    "type": "punishment_cleared",
                                    "shield_saved": False,
                                    "strikes_left": room.get("strikes_left", 3),
                                })
                            else:
                                await websocket.send_json({
                                    "type": "punishment_rejected",
                                    "reason": verdict.get("reason", "校验未通过"),
                                })
                        except Exception:
                            logger.exception("处理惩罚提交异常（room_id=%s）", room_id)
                            await websocket.send_json({
                                "type": "punishment_rejected",
                                "reason": "服务处理异常，请重新提交",
                            })

                    elif mtype == "finish_session":
                        # 生命周期 A（房主结束整房）与 C（AI 单机结束）共用幂等结算核心：
                        # 只有 host 能走到此分支；按 room_id 幂等，重复/并发/旧请求只结算一次。
                        # 生命周期 B（普通学员/房主暂离）走下方 leave_room 分支，保留房间不结算。
                        req_id = msg.get("request_id")
                        outcome = await _settle_room_once(
                            room_id, room, user_id, request_id=req_id, broadcast=True)
                        status = outcome["status"]
                        result = outcome.get("result")
                        if status in (SETTLE_OK, SETTLE_DUPLICATE):
                            # ACK 必须在清理房间前发送（_send_to 依赖 room_connections）。
                            # success 首次已由 _settle_room_once 广播 session_finished；
                            # duplicate 仅回放战报，不重复广播、不重复扣分。
                            await _send_finish_ack(
                                room_id, user_id, status, outcome["message"],
                                result=result, request_id=req_id)
                            _cleanup_finished_room(room_id, room)
                            break
                        elif status == SETTLE_DB_FAILED:
                            # 事务已回滚，明确告知失败，保留房间与连接允许前端重试
                            await _send_finish_ack(
                                room_id, user_id, SETTLE_DB_FAILED, outcome["message"],
                                request_id=req_id)
                        else:
                            await _send_finish_ack(
                                room_id, user_id, SETTLE_INTERNAL,
                                "结算发生内部异常，请稍后重试", request_id=req_id)

                    elif mtype == "leave_room":
                        # 房主暂时离席：跳出循环，由 finally 标记离线并广播 host_status（房间保留）
                        break
                else:
                    # ===== 督学官消息（barrage/sound_effect 已在公共互动分支统一处理） =====
                    if mtype == "finish_session":
                        # 督学官无权结束整房：明确回执 forbidden，避免前端无限等待
                        await _send_finish_ack(
                            room_id, user_id, SETTLE_FORBIDDEN,
                            "只有房主可以结束本局", request_id=msg.get("request_id"))
            except Exception as exc:
                logger.exception(
                    "房间 %s 单条消息处理异常，已隔离避免断连（用户 %s）：%s",
                    room_id, user_id, exc,
                )
    finally:
        # 断开收尾：区分「结算销毁」与「暂离保留」两条路径
        if is_host:
            # 结算路径已在 finish_session 中 pop 房间并广播 session_finished，此处房间已不存在，无需再处理
            if state.active_rooms.get(room_id) is room:
                # 暂离路径：仅标记房主离线、清空画面缓存，房间与场景/惩罚/对话等现场完整保留
                room["host_online"] = False
                room["last_frame_b64"] = None
                room["last_frame_received_ts"] = None  # 画面已清空，时间戳同步归零供观看端识别
                # 向仍在线的督学官广播房主暂离（不主动断开督学官连接，待房主返回恢复）
                if room.get("watchers"):
                    await _broadcast(room_id, {
                        "type": "host_status",
                        "online": False,
                        "msg": "房主暂时离开，稍后回来",
                    }, exclude_user_id=user_id)
            # 仅从连接表移除房主自身连接，保留房间与督学官连接供重连恢复
            # 身份守卫：同用户重连会覆盖 room_connections[room_id][user_id]，
            # 旧连接延迟断开时映射里已是新 ws，不得误删新连接（问题②旧连接不污染新连接）
            with state.room_connections_lock:
                conns = state.room_connections.get(room_id, {})
                if conns.get(user_id) is websocket:
                    conns.pop(user_id, None)
                if not conns:
                    state.room_connections.pop(room_id, None)
        else:
            # 督学官断开：移除连接与 watcher 记录，同步通知房主最新名单
            # 同样加身份守卫，避免旧督学官连接延迟断开误删重连后的新连接
            with state.room_connections_lock:
                conns = state.room_connections.get(room_id, {})
                if conns.get(user_id) is websocket:
                    conns.pop(user_id, None)
                if not conns:
                    state.room_connections.pop(room_id, None)
            (room.get("watchers") or {}).pop(user_id, None)
            await _notify_watcher_update(room_id, room)
        state._inflight_students.discard(user_id)
