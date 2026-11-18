"""
路由层 · AI 剧本编译（M2-A）。

对外提供：
- GET  /api/scenes/presets        获取 4 套内置离线剧本列表（需登录）
- POST /api/scenes/compile        编译任意作品名剧本（需登录，优先缓存/在线生成/离线回退）

设计要点：
1. work_name 用 Pydantic 模型做 1~40 字长度与 XSS 双重约束，与房间路由的 goal_text 校验同策略；
2. 编译属于 CPU/IO 轻量逻辑，直接 async 调用 scene_generator.compile_scene，不占用推理线程池；
3. 所有注释使用中文，本模块不落盘任何报告文件。
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from ..auth import get_current_user
from ..scene_generator import compile_scene, list_presets

router = APIRouter()


class CompileSceneRequest(BaseModel):
    """剧本编译请求：作品名 1~40 字且拒绝危险字符，杜绝存储型 XSS"""
    work_name: str = Field(..., min_length=1, max_length=40)

    @field_validator("work_name")
    @classmethod
    def _validate_work_name(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError("作品名不能为空")
        # 防 XSS：拒绝常见危险字符，确保剧本标题/文案可安全回填前端
        if any(ch in text for ch in "<>&\"'"):
            raise ValueError("作品名包含非法字符")
        return text


@router.get("/api/scenes/presets")
async def scene_presets(user: dict = Depends(get_current_user)):
    """返回 4 套内置离线剧本（需登录），前端开局选场景时直接消费标准 SceneConfig"""
    return {"presets": list_presets()}


@router.post("/api/scenes/compile")
async def scene_compile(request: CompileSceneRequest, user: dict = Depends(get_current_user)):
    """编译作品名剧本：命中缓存直接返回，否则在线生成或平滑回退离线剧本，绝不阻断"""
    return await compile_scene(request.work_name)
