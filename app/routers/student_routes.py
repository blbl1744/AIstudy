"""
路由层 · 学生端：行为上报、/ws 实时推理、语音提醒拉取、本人数据删除，
以及火山引擎语音识别封装（学生端语音链路）。
"""
import asyncio
import base64
import json
import time
from datetime import datetime, timezone

import requests
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from ..analyzer import analyzer
from ..auth import get_current_user, verify_jwt_token
from ..config import (
    AUTO_ALERT_COOLDOWN,
    AUTO_ALERT_SECONDS,
    CORS_ALLOW_ORIGINS,
    VOLC_ACCESS_KEY,
    VOLC_ASR_ENDPOINT,
    VOLC_SECRET_KEY,
    VOLC_TOKEN_URL,
    WARNING_LEVELS,
    logger,
)
from ..db import get_db, get_student_by_id, save_behavior_record
from ..state import (
    _INFER_POOL,
    _inflight_students,
    online_students,
    student_alert_state,
    student_notifications,
    student_video_cache,
    to_iso_utc,
    update_student_video_cache,
)

router = APIRouter()

# ========== feat/extensions 扩展二：持续分心自动预警引擎 ==========
# 单帧计时间隔上限：断线重连或页面切后台回来时，两帧之间的真实时间差可能达到几十秒，
# 若直接累加会把「一次分心」误算成「连续分心很久」，因此单帧最多按 3 秒计。
_MAX_FRAME_DELTA_SEC = 3.0


def _now_ts() -> float:
    """当前时间戳（抽成独立函数，便于测试注入可控时钟，避免用例真的等待十几秒）"""
    return time.time()


def _update_auto_alert_state(student_id: int, focus_score, now_ts: float) -> dict:
    """根据最新一帧的专注度更新「持续分心」状态，必要时自动下发温和提醒

    判定规则：
    - 分心帧：focus_score 为 None（未检测到人体/离座）或低于 WARNING_LEVELS["mild"]（60）；
    - 分心帧累加与上一帧的时间差（单帧上限 3 秒），专注帧则立刻清零并解除预警；
    - 连续分心时长达到 AUTO_ALERT_SECONDS：置 is_alerting=True 并记录原因（供教师端高亮）；
    - 自动提醒：距上次自动提醒超过 AUTO_ALERT_COOLDOWN 秒，且当前没有未读提醒时才下发，
      既避免刷屏，也避免覆盖教师刚手动发出的提醒。
    """
    state = student_alert_state.get(student_id) or {
        "distracted_streak_sec": 0.0,
        "last_frame_ts": now_ts,
        "last_alert_ts": 0.0,
        "is_alerting": False,
        "alert_reason": "",
    }

    last_frame_ts = state.get("last_frame_ts") or now_ts
    # 时间差做上下截断：上界防跳变误判，下界防同一时间戳或时钟回拨产生负数
    delta = min(max(now_ts - last_frame_ts, 0.0), _MAX_FRAME_DELTA_SEC)
    state["last_frame_ts"] = now_ts

    # 未检测到人体（None）或分数低于温和线，均记为分心帧
    is_distracted = focus_score is None or focus_score < WARNING_LEVELS["mild"]

    if not is_distracted:
        # 恢复端正坐姿/专注状态：连续分心清零，预警立即解除
        state["distracted_streak_sec"] = 0.0
        state["is_alerting"] = False
        state["alert_reason"] = ""
        student_alert_state.set(student_id, state)
        return state

    streak = float(state.get("distracted_streak_sec") or 0.0) + delta
    state["distracted_streak_sec"] = round(streak, 3)

    if streak < AUTO_ALERT_SECONDS:
        # 尚未达到阈值：只累积时长，不打扰学生
        student_alert_state.set(student_id, state)
        return state

    # 达到阈值：标记预警中，并区分「离座」与「在座但低头/分心」两类原因
    reason = "连续离开座位" if focus_score is None else "持续低头/分心"
    state["is_alerting"] = True
    state["alert_reason"] = reason

    cooldown_elapsed = now_ts - float(state.get("last_alert_ts") or 0.0) >= AUTO_ALERT_COOLDOWN
    has_unread = student_notifications.get(student_id) is not None
    if cooldown_elapsed and not has_unread:
        # 温和提醒：只提示、不批评，文案与教师手动提醒共用同一渲染通道
        student_notifications.set(student_id, {
            "message": f"已经{reason}超过 {AUTO_ALERT_SECONDS} 秒啦，先坐正、深呼吸一下，我们继续加油～",
            "level": "warning",
            "from_teacher": "智能督学助手",
            "created_at": to_iso_utc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
        })
        state["last_alert_ts"] = now_ts
        logger.info(
            "自动预警已下发：student_id=%s，原因=%s，连续分心=%s 秒（阈值 %s 秒，冷却 %s 秒）",
            student_id, reason, round(streak, 1), AUTO_ALERT_SECONDS, AUTO_ALERT_COOLDOWN,
        )

    student_alert_state.set(student_id, state)
    return state

# --- 学生保存自己的行为数据 ---
@router.post("/api/student/behavior")
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


@router.websocket("/ws")
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

                # ===== 扩展二：每帧推理后更新持续分心状态（教师端状态、自动提醒都在此产生）=====
                # 放在缓存更新之前，保证教师端拉列表时看到的是本帧判定后的最新预警状态。
                _update_auto_alert_state(student_id, result.get("focus_score"), _now_ts())

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

@router.get("/api/student/notifications")
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
@router.delete("/api/student/my-data")
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

    # 2) 清理内存缓存：实时画面、在线状态、待推送提醒、持续分心预警状态
    student_video_cache.pop(student_id, None)
    online_students.pop(student_id, None)
    student_notifications.pop(student_id, None)
    student_alert_state.pop(student_id, None)

    logger.info("学生 %s 已删除本人数据，共 %s 条行为记录", student_id, deleted_records)
    return {
        "success": True,
        "deleted_records": deleted_records,
        "message": "本人行为记录与实时缓存已删除"
    }
