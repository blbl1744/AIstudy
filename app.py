"""AI学伴 · 兼容启动入口（P3-1 拆分后保留）

P3-1 已将原 2148 行单文件按职责拆分至 app/ 包，本文件仅作为兼容外壳保留，不承载业务逻辑：
1. `uvicorn app:app` —— 对外仍暴露 app 对象，历史启动命令无需任何修改；
2. `python app.py`   —— 本地调试仍可直接启动。

真实代码位置：app/config.py、state.py、db.py、auth.py、model_loader.py、analyzer.py、
app/routers/*.py、app/main.py。
"""
import uvicorn

from app.config import HOST, PORT, logger
from app.main import app  # noqa: F401  对外导出 app，供 uvicorn app:app 使用

__all__ = ["app"]


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("AI学伴 · 统一 API 服务（M1-A）")
    logger.info("接口文档: http://%s:%s/docs", HOST, PORT)
    logger.info("=" * 50)
    uvicorn.run(app, host=HOST, port=PORT)
