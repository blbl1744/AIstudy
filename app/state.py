"""
运行时状态层：全局缓存、推理线程池、背压标记与时间格式工具。

设计约束：
1. 本模块只依赖 config，不依赖数据库/路由，供 db、analyzer、routers、main 共同引用，
   保证全局状态只有一份真源（不会因为拆分出现各自一份缓存）；
2. TTLCache 内部自带锁，可同时被事件循环线程与推理工作线程访问。
"""
import asyncio
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from .config import logger

# 导致其他学生的 REST 请求（列表/记录/提醒）排队超时。这里用固定 4 个工作线程的池承接推理，
# 线程名统一加 "infer" 前缀便于日志排查；进程退出时在 lifespan 中 shutdown。
_INFER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="infer")

# ========== P2-2：单学生背压集合 ==========
# 记录「当前已有帧正在推理中」的学生 id：同一学生上一帧未算完时，新帧直接丢弃，
# 避免弱网/低配机器上帧缓冲无限积压导致的雪崩。
# 该集合只在事件循环线程内增删（工作线程不触碰），因此无需额外加锁。
_inflight_students: set = set()


# ========== P2-3：带 TTL 与容量上限的线程安全内存缓存 ==========
class TTLCache:
    """轻量 TTL + LRU 容量上限缓存（防止长期运行的全局字典无限膨胀导致 OOM）

    设计要点：
    - 用 OrderedDict 维护插入顺序，配合 move_to_end 实现 LRU 淘汰；
    - 全部读写都在 RLock 保护下进行，可被事件循环线程与推理工作线程同时访问；
    - get / items / __contains__ / __len__ 均会顺带判断过期（懒惰淘汰），
      另有 cleanup() 供后台任务批量回收，返回本次实际清除的条数；
    - set 时若 key 已存在先移除再插入，保证 LRU 顺序与「最新一次写入」一致。
    """

    def __init__(self, maxsize=1000, ttl=60.0):
        self.maxsize = int(maxsize)
        self.ttl = float(ttl)
        self._store = OrderedDict()      # key -> (expire_at_ts, value)
        self._lock = threading.RLock()

    # ---------- 内部工具 ----------
    def _is_expired(self, expire_at) -> bool:
        """按单调时钟判断是否过期（避免系统时间回拨导致条目永不过期）"""
        return expire_at <= time.monotonic()

    def _purge_locked(self) -> int:
        """在已持锁前提下清除所有过期条目，返回清除数量"""
        now = time.monotonic()
        dead = [k for k, (exp, _) in self._store.items() if exp <= now]
        for k in dead:
            self._store.pop(k, None)
        return len(dead)

    def _evict_locked(self):
        """超容量时按 LRU 淘汰最旧条目"""
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    # ---------- 对外接口 ----------
    def set(self, key, value):
        """写入/覆盖一条记录，TTL 从本次写入时刻重新计时"""
        with self._lock:
            self._store.pop(key, None)
            self._store[key] = (time.monotonic() + self.ttl, value)
            self._evict_locked()

    def get(self, key, default=None):
        """读取记录；不存在或已过期一律返回 default"""
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return default
            expire_at, value = item
            if self._is_expired(expire_at):
                # 懒惰淘汰：读到过期项立即清除，保证教师端不会看到离线学生的旧画面
                self._store.pop(key, None)
                return default
            self._store.move_to_end(key)
            return value

    def pop(self, key, default=None):
        """取出并删除（用于提醒的一次性消费）"""
        with self._lock:
            item = self._store.pop(key, None)
            if item is None:
                return default
            expire_at, value = item
            if self._is_expired(expire_at):
                return default
            return value

    def cleanup(self) -> int:
        """清除全部过期条目，返回清除数量（供后台 janitor 定时调用）"""
        with self._lock:
            return self._purge_locked()

    def items(self):
        """返回未过期条目的快照列表（调试接口用，避免持锁遍历时被外部修改）"""
        with self._lock:
            now = time.monotonic()
            return [(k, v) for k, (exp, v) in self._store.items() if exp > now]

    def __contains__(self, key) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            self._purge_locked()
            return len(self._store)


# ========== P2-3：后台缓存清理协程 ==========
async def _cache_janitor():
    """每 60 秒清理一次内存缓存中的过期条目（防 OOM），含扩展二的分心预警状态"""
    while True:
        try:
            await asyncio.sleep(60)
            purged_video = student_video_cache.cleanup()
            purged_online = online_students.cleanup()
            purged_notice = student_notifications.cleanup()
            purged_alert = student_alert_state.cleanup()
            if purged_video or purged_online or purged_notice or purged_alert:
                logger.info(
                    "[缓存清理] 过期回收：视频帧 %s 条 / 在线状态 %s 条 / 提醒 %s 条 / 预警状态 %s 条",
                    purged_video, purged_online, purged_notice, purged_alert,
                )
        except asyncio.CancelledError:
            # 生命周期结束时被 cancel，正常退出
            raise
        except Exception:
            # 清理任务本身异常不能拖垮主服务
            logger.exception("[缓存清理] 后台清理任务执行异常")

# ========== 时间格式工具（P1-7） ==========
def to_iso_utc(v):
    """将 SQLite 存储的 UTC 时间字符串转为带 Z 后缀的 ISO-8601 格式"""
    if not v:
        return None
    s = str(v).replace(" ", "T")
    return s if s.endswith("Z") or "+" in s else s + "Z"

# ========== 学生实时视频帧缓存（P2-3：TTL + 容量上限，防 OOM）==========
# 视频帧：10 秒无更新即视为离线自动过期；容量上限 200，超出按 LRU 淘汰
student_video_cache = TTLCache(maxsize=200, ttl=10)
# 语音提醒：学生端一次性消费，5 分钟未取自动过期，避免堆积
student_notifications = TTLCache(maxsize=1000, ttl=300)
# 在线学生状态：60 秒无心跳视为离线
online_students = TTLCache(maxsize=500, ttl=60)
# 扩展二：持续分心自动预警状态（连续分心时长 / 上次提醒时间 / 是否预警中 / 预警原因）
# 5 分钟无更新自动回收，容量上限 500，避免学生长期离线后状态无限堆积
student_alert_state = TTLCache(maxsize=500, ttl=300)

def update_student_video_cache(student_id: int, annotated_image: str, focus_score: int,
                                 action_label="unknown", head_pose="", eye_focus="", fatigue_level="",
                                 name=""):
    """更新学生视频帧缓存（TTLCache 内部加锁，可被推理工作线程安全调用）"""
    student_video_cache.set(student_id, {
        "image": annotated_image,
        "timestamp": time.time(),
        "focus_score": focus_score,
        "action_label": action_label,
        "head_pose": head_pose,
        "eye_focus": eye_focus,
        "fatigue_level": fatigue_level,
        "name": name
    })

def get_student_video_cache(student_id: int):
    """获取学生视频帧缓存（过期或不存在返回 None）"""
    return student_video_cache.get(student_id)
