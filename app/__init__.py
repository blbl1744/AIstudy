"""
AI学伴 · 1对1专注度智能监护系统（app 包）

P3-1：原 2148 行单文件 app.py 已按职责拆分为本包，依赖层级自底向上为：
    config → state → db → auth → model_loader → analyzer → routers/* → main
下层模块严禁向上层模块 import，以保证无循环导入。

为兼容历史启动方式 `uvicorn app:app`，本包通过 PEP 562 模块级 __getattr__
惰性暴露 app 对象：仅在真正访问 app 属性时才导入 main，包导入期依旧保持轻量。
"""

__all__ = ["app"]


def __getattr__(name):
    """惰性导出 FastAPI 实例，兼容 uvicorn app:app，避免包导入期加载重量级依赖"""
    if name == "app":
        from .main import app as fastapi_app

        return fastapi_app
    raise AttributeError("模块 app 不存在属性 %r" % (name,))
