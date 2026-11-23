"""
路由层 · 战报明细与 CSV 导出（M5 收官）。

对外提供：
- GET /api/stats/sessions/{session_id}   单局战报摘要 + 专注度采样点（需登录，仅本人可看）
- GET /api/stats/export.csv              当前用户全部战报的 CSV 导出（需登录，UTF-8 BOM）

设计要点：
1. 详情接口严格校验归属：战报不存在返回 404，非本人返回 403，杜绝越权读取他人曲线；
2. 采样点取自 behavior_records 中以 session_id 关联的记录，created_at 统一经 to_iso_utc 格式化；
3. CSV 写入 UTF-8 BOM 头，避免 Excel 打开中文乱码；
4. 所有文本列做公式注入转义：以 = + - @ 开头的值前置单引号，防止表格软件执行公式；
5. 所有注释使用中文，本模块不落盘任何报告文件。
"""
import csv
import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from ..auth import get_current_user
from ..db import get_db
from ..state import to_iso_utc

router = APIRouter()

# CSV 表头（严格对齐任务书给定的顺序）
CSV_HEADERS = [
    "结算时间(UTC)", "督学模式", "场景名称", "学习目标",
    "专注时长(分钟)", "平均专注分", "消耗机会次数", "惩罚类型", "结算状态",
]

# 危险公式前缀：单元格以此开头时前置单引号，防止 Excel / WPS 解析为公式执行
_DANGEROUS_PREFIXES = ("=", "+", "-", "@")


def _csv_escape(value):
    """把单个 CSV 单元格值转为安全文本：以危险字符开头时前置单引号。"""
    if value is None:
        return ""
    text = str(value)
    if text[:1] in _DANGEROUS_PREFIXES:
        return "'" + text
    return text


def _current_user_id(user: dict) -> int:
    """从 JWT 载荷取统一账号 ID；缺失则视为无效凭证"""
    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="无效的Token")
    return user_id


@router.get("/api/stats/sessions/{session_id}")
async def session_detail(session_id: int, user: dict = Depends(get_current_user)):
    """单局战报详情：先校验归属，再返回战报摘要与行为采样点列表供前端绘图。"""
    user_id = _current_user_id(user)
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, user_id, mode, scene_title, goal_text, duration_minutes, avg_score, "
            "strikes_used, punishment_type, status, created_at FROM study_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="战报不存在")
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="无权查看他人战报")

    # 行为采样点：session_id 在 behavior_records 中为 TEXT，统一用字符串比较并按时间升序
    with get_db() as conn:
        samples = conn.execute(
            "SELECT created_at, focus_score FROM behavior_records "
            "WHERE session_id = ? ORDER BY created_at ASC, id ASC",
            (str(session_id),),
        ).fetchall()
    points = [
        {"created_at": to_iso_utc(s["created_at"]), "focus_score": s["focus_score"]}
        for s in samples
    ]

    summary = {
        "id": row["id"],
        "mode": row["mode"],
        "scene_title": row["scene_title"],
        "goal_text": row["goal_text"],
        "duration_minutes": row["duration_minutes"],
        "avg_score": row["avg_score"],
        "strikes_used": row["strikes_used"],
        "punishment_type": row["punishment_type"],
        "status": row["status"],
        "created_at": to_iso_utc(row["created_at"]),
    }
    return {"summary": summary, "points": points}


@router.get("/api/stats/export.csv")
async def export_csv(user: dict = Depends(get_current_user)):
    """导出当前用户全部战报为 CSV：UTF-8 BOM 头 + 公式注入转义，可直接被 Excel 打开。"""
    user_id = _current_user_id(user)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT created_at, mode, scene_title, goal_text, duration_minutes, avg_score, "
            "strikes_used, punishment_type, status FROM study_sessions "
            "WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,),
        ).fetchall()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_HEADERS)
    for row in rows:
        writer.writerow([
            _csv_escape(to_iso_utc(row["created_at"])),
            _csv_escape(row["mode"]),
            _csv_escape(row["scene_title"]),
            _csv_escape(row["goal_text"]),
            row["duration_minutes"],
            row["avg_score"] if row["avg_score"] is not None else "",
            row["strikes_used"],
            _csv_escape(row["punishment_type"]),
            _csv_escape(row["status"]),
        ])
    # 前置 UTF-8 BOM，防止 Excel 以本地编码打开导致中文乱码
    content = "\ufeff" + buffer.getvalue()
    return Response(
        content=content.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="study_sessions.csv"'},
    )
