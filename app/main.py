"""
应用装配层：创建 FastAPI 实例、生命周期钩子、CORS、全局脱敏异常处理器、
注册各路由，并提供 /healthz 健康检查。

M1-A 剥离双角色模板与 HTML 首页；M1-B 挂载房间/公告广场 REST 与统一房间 WS 网关。
本文件只做「接线」，不含业务逻辑，行数严格控制在 200 行以内。
"""
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import model_loader
from .config import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_ORIGINS,
    logger,
)
from .db import get_db, init_db
from .model_loader import load_trained_model
from .routers import auth_routes, room_routes
from .state import (
    _INFER_POOL,
    _cache_janitor,
    _inflight_students,
    active_rooms,
    room_connections,
)

# 进程启动时刻（供 /healthz 计算运行时长）
_START_TS = time.time()


# 保证无论是 `python app.py` 还是 `uvicorn app:app` 启动都会自动执行一次。
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：启动阶段完成建表、迁移与模型校验加载，并启动缓存清理任务；
    关闭阶段优雅取消该任务并释放推理线程池。"""
    init_db()
    load_trained_model()
    janitor_task = asyncio.create_task(_cache_janitor())
    logger.info("[启动] 数据库初始化与模型加载完成，缓存清理任务已启动")
    try:
        yield
    finally:
        janitor_task.cancel()
        try:
            await janitor_task
        except asyncio.CancelledError:
            pass
        # 不等待在途推理完成，避免退出阻塞（守护语义）
        _INFER_POOL.shutdown(wait=False)
        logger.info("[退出] 应用已关闭，后台任务与推理线程池已释放")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,            # 来源白名单，由环境变量配置
    allow_credentials=CORS_ALLOW_CREDENTIALS,    # 来源含通配符时强制关闭凭证
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 全局兜底异常处理器（对外脱敏） ==========
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    """兜底捕获未处理异常：日志留完整堆栈，响应只回通用安全提示"""
    logger.exception("未处理异常：%s %s", getattr(request, "method", "-"), getattr(request.url, "path", "-"))
    return JSONResponse(status_code=500, content={"detail": "服务暂时不可用"})


# ========== 路由注册（M1-B：统一认证 + 房间/公告广场） ==========
app.include_router(auth_routes.router)
app.include_router(room_routes.router)


# ========== 健康检查接口 ==========
@app.get("/healthz")
async def healthz():
    """健康检查：只返回运行状态与计数，不暴露路径、密钥、堆栈等内部细节"""
    database_ok = True
    try:
        with get_db() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception:
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
            "active_rooms": len(active_rooms),
            "room_connections": sum(len(peers) for peers in room_connections.values()),
        },
    }
    return JSONResponse(status_code=200 if database_ok else 503, content=payload)


if __name__ == "__main__":
    import uvicorn as _uvicorn

    from .config import HOST, PORT

    _uvicorn.run(app, host=HOST, port=PORT)
