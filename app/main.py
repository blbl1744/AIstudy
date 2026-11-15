"""
应用装配层：创建 FastAPI 实例、生命周期钩子、CORS、全局脱敏异常处理器、
注册各路由，并提供首页与 /healthz 健康检查（P3-4 前置）。

本文件只做「接线」，不含业务逻辑，行数严格控制在 200 行以内。
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import model_loader
from .config import (
    BASE_DIR,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_ORIGINS,
    logger,
)
from .db import get_db, init_db
from .model_loader import load_trained_model
from .routers import auth_routes, student_routes, teacher_routes
from .state import (
    _INFER_POOL,
    _cache_janitor,
    _inflight_students,
    online_students,
    student_notifications,
    student_video_cache,
)

# 进程启动时刻（供 /healthz 计算运行时长）
_START_TS = time.time()


# 保证无论是 `python app.py` 还是 `uvicorn app:app` 启动都会自动执行一次。
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：启动阶段完成建表、历史表迁移与专注度模型校验加载，
    并启动缓存清理后台任务；关闭阶段优雅取消该任务并释放推理线程池。"""
    init_db()
    load_trained_model()
    # P2-3：启动后台缓存清理任务
    janitor_task = asyncio.create_task(_cache_janitor())
    logger.info("[启动] 数据库初始化与模型加载完成，缓存清理任务已启动")
    try:
        yield
    finally:
        # P2-3：优雅取消后台清理任务
        janitor_task.cancel()
        try:
            await janitor_task
        except asyncio.CancelledError:
            pass
        # P2-2：不等待在途推理完成，避免退出阻塞（守护语义）
        _INFER_POOL.shutdown(wait=False)
        logger.info("[退出] 应用已关闭，后台任务与推理线程池已释放")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,            # P0-6：来源白名单，由环境变量 AI_STUDY_CORS_ORIGINS 配置
    allow_credentials=CORS_ALLOW_CREDENTIALS,    # P0-6：来源含通配符时强制关闭凭证
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== P2-5：全局兜底异常处理器（对外脱敏） ==========
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    """兜底捕获未处理异常：日志留完整堆栈，响应只回通用安全提示

    严禁把 str(exc)、异常类型名、堆栈等内部细节写进响应体，避免泄露表名、
    文件路径、第三方接口地址等敏感实现细节。
    """
    logger.exception("未处理异常：%s %s", getattr(request, "method", "-"), getattr(request.url, "path", "-"))
    return JSONResponse(status_code=500, content={"detail": "服务暂时不可用"})

# ========== 路由注册（按角色拆分） ==========
app.include_router(auth_routes.router)
app.include_router(teacher_routes.router)
app.include_router(student_routes.router)


@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))

# ========== P3-4 前置：健康检查接口 ==========
@app.get("/healthz")
async def healthz():
    """健康检查：供负载均衡/容器探针直接调用，无需鉴权

    返回内容只包含运行状态与计数，不暴露路径、密钥、堆栈等内部细节：
    - 数据库：执行一次 SELECT 1 轻量探活，失败则整体置为 degraded（HTTP 503）；
    - 模型：只读当前加载状态，不触发重新加载；
    - 运行态：推理在途学生数与三类缓存条目数。
    """
    database_ok = True
    try:
        with get_db() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception:
        # 只记日志，不把异常细节写进响应
        logger.exception("[健康检查] 数据库探活失败")
        database_ok = False

    payload = {
        "status": "ok" if database_ok else "degraded",
        "uptime_seconds": round(time.time() - _START_TS, 1),
        "database": "ok" if database_ok else "error",
        "model": {
            "loaded": model_loader.model_is_loaded(),
            "feature_version": model_loader.FEATURE_VERSION,
            "feature_dim": model_loader.FEATURE_DIM,
        },
        "runtime": {
            "inflight_students": len(_inflight_students),
            "video_cache": len(student_video_cache),
            "online_students": len(online_students),
            "pending_notifications": len(student_notifications),
        },
    }
    return JSONResponse(status_code=200 if database_ok else 503, content=payload)


if __name__ == "__main__":
    # 兼容 `python -m app.main` 直接启动；建表/迁移/模型加载由 lifespan 统一负责
    import uvicorn as _uvicorn

    from .config import HOST, PORT

    _uvicorn.run(app, host=HOST, port=PORT)
