# ---------- AI 学习专注度分析系统 · 生产镜像 ----------
# 构建：docker build -t ai-study-buddy:latest .
# 运行：docker run -d --name ai-study -p 8001:8001 \
#         -e AI_STUDY_JWT_SECRET="<不少于32位的随机串>" \
#         -e AI_STUDY_CORS_ORIGINS="https://study.example.com" \
#         -v ai_study_data:/data -e AI_STUDY_DB=/data/ai_study_buddy.db \
#         ai-study-buddy:latest
# 说明：应用只监听容器内 8001 端口，对外一律经 Nginx 反向代理并启用 HTTPS/WSS。

FROM python:3.11-slim

# ---- 基础环境变量 ----
# PYTHONDONTWRITEBYTECODE：不在镜像内产生 __pycache__；
# PYTHONUNBUFFERED：日志实时输出到容器标准输出，便于 docker logs / 日志采集；
# PIP_NO_CACHE_DIR：镜像内不留 pip 缓存，减小体积；
# TZ：统一容器时区，保证日志与 UTC 存储的时间换算可追溯。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# ---- 系统依赖：OpenCV / MediaPipe 运行时必需的图形与 glib 库 ----
# libgl1       : 提供 libGL.so.1（opencv 导入时必须，缺失会报 ImportError: libGL.so.1）
# libglib2.0-0 : 提供 glib 运行库（mediapipe 依赖）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python 依赖：先只拷贝依赖清单，最大化利用镜像层缓存 ----
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- 业务代码与前端模板 ----
COPY . .

# ---- 运行期目录与数据卷挂载点 ----
# logs：应用日志目录（容器内以非 root 权限写入需要提前建好并授权）；
# /data：数据库推荐挂载点，配合 AI_STUDY_DB=/data/ai_study_buddy.db 让数据落在卷上，
#        容器重建后行为记录与账号不会丢失。
RUN mkdir -p /app/logs /data \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8001

# ---- 启动命令：显式指向重构后的 app.main:app，绕开兼容外壳 app.py ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
