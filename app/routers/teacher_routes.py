"""
路由层 · 教师端：学生列表、行为记录、实时状态、快照图片，以及受环境变量开关控制的调试接口。"""
import base64
import os
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from ..auth import get_current_user
from ..config import logger
from ..db import (
    get_all_students,
    get_latest_records_for_students,
    get_student_behavior_records,
    get_student_by_id,
)
from ..state import get_student_video_cache, student_video_cache

router = APIRouter()

# --- 教师获取所有学生列表 ---
@router.get("/api/teacher/students")
async def teacher_list_students(user: dict = Depends(get_current_user)):
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    students = get_all_students()
    # P2-4：先一次性批量取回所有学生的最新行为记录（查库次数恒为 1，彻底消除 N+1），
    # 再在内存里做归并，避免原实现「循环内逐个查库」在 20 个学生时触发 21 次查询。
    latest_map = get_latest_records_for_students([s["id"] for s in students])
    for s in students:
        # 附加最新行为记录（无记录则为 None，前端按空态渲染）
        s["latest_record"] = latest_map.get(s["id"])
        # 检查在线状态：视频帧缓存 TTL 为 10 秒，取得到即视为在线
        cache = get_student_video_cache(s["id"])
        if cache:
            s["online"] = True
            s["live_focus_score"] = cache["focus_score"]
        else:
            s["online"] = False
            s["live_focus_score"] = None
    return {"students": students}

# --- 教师查看某个学生的行为记录 ---
@router.get("/api/teacher/students/{student_id}/records")
async def teacher_view_records(student_id: int, user: dict = Depends(get_current_user)):
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    records = get_student_behavior_records(student_id)
    return {"student": student, "records": records}

# --- 教师获取学生实时视频帧 ---
@router.get("/api/teacher/students/{student_id}/live")
async def teacher_get_student_live(student_id: int, user: dict = Depends(get_current_user)):
    """教师获取学生的实时摄像头画面"""
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    
    cache = get_student_video_cache(student_id)
    if not cache:
        return {
            "student_id": student_id,
            "online": False,
            "message": "学生未在线或未开启监控"
        }
    
    # P2-3：视频帧缓存 TTL=10s，取到即说明最近 10 秒内有更新；此处再按时间戳二次校验
    is_online = (time.time() - cache.get("timestamp", 0)) < 10
    
    return {
        "student_id": student_id,
        "online": is_online,
        # P2-6：不再内联 base64 大图（单帧可达数百 KB，2 秒轮询会让教师端带宽爆掉），
        # 只回状态与分数；画面由 GET /api/teacher/students/{id}/snapshot.jpg 提供。
        "has_image": bool(cache.get("image")),
        "focus_score": cache["focus_score"],
        "timestamp": cache["timestamp"]
    }

# --- 教师拉取学生最新画面快照（P2-6）---
@router.get("/api/teacher/students/{student_id}/snapshot.jpg")
async def teacher_get_student_snapshot(student_id: int, user: dict = Depends(get_current_user)):
    """教师端拉取学生最新一帧画面（二进制 JPEG）

    背景：/ws 不再下行整帧 base64 图像，原始画面只留在服务端缓存里，
    教师端需要看图时走这个受鉴权保护的二进制接口，避免 base64 膨胀 33% 与 JSON 解析开销。
    无画面、缓存过期、解码失败统一返回 404，不向外暴露任何内部异常信息。
    """
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    cache = get_student_video_cache(student_id)
    if not cache:
        raise HTTPException(status_code=404, detail="暂无画面")
    image_b64 = cache.get("image") or ""
    if not image_b64:
        raise HTTPException(status_code=404, detail="暂无画面")
    try:
        raw = base64.b64decode(image_b64)
    except Exception:
        # 内部解码异常只进日志，对外统一 404
        logger.exception("学生 %s 画面快照解码失败", student_id)
        raise HTTPException(status_code=404, detail="暂无画面")
    if not raw:
        raise HTTPException(status_code=404, detail="暂无画面")
    # 画面实时变化，禁止浏览器/中间层缓存，否则教师端会看到旧帧
    return Response(content=raw, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ========== 前端页面（已抽离至 templates/index.html） ==========


# 调试接口（P0-3）：默认不注册；仅当环境变量 AI_STUDY_DEBUG_API == "1" 时注册，且仅教师可访问
if os.getenv("AI_STUDY_DEBUG_API") == "1":
    @router.get("/api/debug/cache")
    async def debug_cache(user: dict = Depends(get_current_user)):
        """调试接口：查看所有学生视频缓存（需教师权限）"""
        if user.get("role") != "teacher":
            raise HTTPException(status_code=403, detail="仅教师可访问")
        result = {}
        for sid, cache in student_video_cache.items():
            result[sid] = {
                "has_image": cache.get("image") is not None,
                "image_length": len(cache.get("image", "")),
                "timestamp": cache.get("timestamp"),
                "name": cache.get("name"),
                "focus_score": cache.get("focus_score")
            }
        return {
            "total_students": len(student_video_cache),
            "caches": result
        }
