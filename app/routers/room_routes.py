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

from .. import state
from ..analyzer import analyzer
from ..auth import get_current_user, verify_jwt_token
from ..config import CORS_ALLOW_ORIGINS, logger
from ..db import calc_rank_title, get_db, save_behavior_record

router = APIRouter()

# ===== 哨兵预警参数（模块级常量，测试可 monkeypatch 压缩阈值）=====
SENTINEL_ALERT_SEC = 8.0         # 连续走神达到该秒数触发预警
SENTINEL_COOLDOWN_SEC = 30.0     # 同一房间两次预警的最小间隔（防刷屏）
DISTRACTED_SCORE = 60            # focus_score 低于该值或为 None 视为走神
SOUND_WHITELIST = {"knock_board", "clear_throat", "alarm", "cheer"}

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


def _current_user_id(user: dict) -> int:
    """从 JWT 载荷取统一账号 ID；缺失则视为无效凭证"""
    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="无效的Token")
    return int(user_id)


# ========== REST：创建房间 ==========
@router.post("/api/rooms")
async def create_room(request: CreateRoomRequest, user: dict = Depends(get_current_user)):
    """登录用户创建房间；goal_text 已由模型完成长度与 XSS 校验"""
    user_id = _current_user_id(user)
    room_id = uuid.uuid4().hex
    room = {
        "room_id": room_id,
        "host_id": user_id,
        "host_nickname": user.get("nickname", ""),
        "mode": request.mode,
        "is_public": request.is_public,
        "scene_id": request.scene_id,
        "scene_title": request.scene_title,
        "goal_text": request.goal_text,
        "max_watchers": request.max_watchers,
        "punishment_types": list(request.punishment_types),
        "created_at": _now_ts(),
        "strikes_left": 3,
        "distracted_streak_sec": 0.0,
        "last_frame_ts": None,
        "last_frame_b64": None,
        "score_history": [],
        "watchers": {},
        # 哨兵上次预警时间（内部字段，不对外回显）
        "_last_sentinel_alert_ts": 0.0,
    }
    state.active_rooms.set(room_id, room)
    # 回显房间信息，剔除内部下划线字段
    return {k: v for k, v in room.items() if not k.startswith("_")}


# ========== REST：公告广场 ==========
@router.get("/api/rooms/plaza")
async def plaza(user: dict = Depends(get_current_user)):
    """返回所有公开且非 ai_solo 的活跃房间快照"""
    now = _now_ts()
    result = []
    for _, room in state.active_rooms.items():
        if not isinstance(room, dict):
            continue
        if not room.get("is_public") or room.get("mode") == "ai_solo":
            continue
        result.append({
            "room_id": room.get("room_id"),
            "host_nickname": room.get("host_nickname", ""),
            "mode": room.get("mode"),
            "scene_title": room.get("scene_title", ""),
            "goal_text": room.get("goal_text", ""),
            "elapsed_seconds": max(0, int(now - (room.get("created_at") or now))),
            "watcher_count": len(room.get("watchers") or {}),
            "max_watchers": room.get("max_watchers", 0),
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
    return Response(content=content, media_type="image/jpeg")


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

    # 2) 更新分数历史与连续走神秒数（按帧间隔累加，可注入时钟）
    room.setdefault("score_history", []).append(score)
    now = _now_ts()
    prev = room.get("last_frame_ts")
    delta = (now - prev) if prev is not None else 0.0
    room["last_frame_ts"] = now
    distracted = (score is None) or (score < DISTRACTED_SCORE)
    if distracted:
        room["distracted_streak_sec"] = float(room.get("distracted_streak_sec") or 0.0) + delta
    else:
        room["distracted_streak_sec"] = 0.0

    # 3) 分析结果广播给全房
    await _broadcast(room_id, {
        "type": "analysis_update",
        "focus_score": score,
        "action_label": payload.get("action_label"),
        "annotated_image": payload.get("annotated_image"),
        "strikes_left": room.get("strikes_left", 3),
        "distracted_streak_sec": round(room["distracted_streak_sec"], 2),
    })

    # 4) 哨兵预警：有督学官在线、连续走神达标且已过冷却期时，仅向督学官推送
    watchers = room.get("watchers") or {}
    last_alert = room.get("_last_sentinel_alert_ts") or 0.0
    if watchers and room["distracted_streak_sec"] >= SENTINEL_ALERT_SEC \
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
    punishment = ",".join(room.get("punishment_types") or [])
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

    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")

            if is_host:
                # ===== 学习者消息 =====
                if mtype == "frame":
                    image = msg.get("image")
                    if not image:
                        continue
                    # 快照画面即时保存（推理未完成也可供鉴权快照读取）
                    room["last_frame_b64"] = image
                    # 背压：同一学习者上一帧未算完则丢弃
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

                elif mtype == "finish_session":
                    now = _now_ts()
                    created = float(room.get("created_at") or now)
                    duration_minutes = int(max(0.0, now - created) // 60)
                    scores = [s for s in (room.get("score_history") or [])
                              if isinstance(s, (int, float)) and not isinstance(s, bool)]
                    avg_score = round(sum(scores) / len(scores)) if scores else None
                    strikes_used = max(0, 3 - int(room.get("strikes_left", 3)))
                    settled = _settle_session(room, user_id, duration_minutes, avg_score, strikes_used)
                    await _broadcast(room_id, {
                        "type": "session_finished",
                        "duration_minutes": duration_minutes,
                        "avg_score": avg_score,
                        **settled,
                    })
                    # 结算后清理房间与全部连接
                    state.active_rooms.pop(room_id, None)
                    with state.room_connections_lock:
                        state.room_connections.pop(room_id, None)
                    state._inflight_students.discard(user_id)
                    break
            else:
                # ===== 督学官消息 =====
                if mtype == "barrage":
                    text = str(msg.get("text", "")).strip()
                    if 1 <= len(text) <= 60:
                        await _broadcast(room_id, {
                            "type": "room_interaction", "kind": "barrage",
                            "from": nickname, "text": text,
                        })
                elif mtype == "sound_effect":
                    sound = msg.get("sound")
                    if sound in SOUND_WHITELIST:
                        await _broadcast(room_id, {
                            "type": "room_interaction", "kind": "sound_effect",
                            "from": nickname, "sound": sound,
                        })
    finally:
        # 断开收尾：移除连接，督学官离开时同步通知房主
        with state.room_connections_lock:
            conns = state.room_connections.get(room_id, {})
            conns.pop(user_id, None)
            if not conns:
                state.room_connections.pop(room_id, None)
        if not is_host:
            (room.get("watchers") or {}).pop(user_id, None)
            await _notify_watcher_update(room_id, room)
        state._inflight_students.discard(user_id)
