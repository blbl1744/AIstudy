"""
运行时状态层：进程内房间缓存、连接映射、推理线程池、背压标记与时间格式工具。

设计约束：
1. 本模块只依赖 config，不依赖数据库/路由，供 db、analyzer、routers、main 共同引用，
   保证全局状态只有一份真源（不会因为拆分出现各自一份缓存）；
2. TTLCache 内部自带锁，可同时被事件循环线程与推理工作线程访问；
3. main 生命周期钩子统一调度后台清理任务，避免多处启动 janitor；
4. M1-B：清理旧师生缓存，统一为 active_rooms（房间字典）+ room_connections（连接映射），
   房间 ttl=7200 秒，连接映射配线程锁并在 janitor 中回收失效连接。

注意：本模块只保存「进程内的临时状态」，数据库 CRUD 全部下沉到 db.py。
"""
import asyncio
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from .config import logger

# 全局共享推理线程池：analyzer 的视觉推理统一投递至此，避免每条 WS 各自建池
_INFER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="infer")

# 推理背压标记：记录当前有帧在推理中的学习者 user_id（房间制下即 host_id），
# 只在事件循环线程内增删，故用普通 set 即可。
_inflight_students: set = set()


class TTLCache:
    """带容量上限与过期时间的轻量缓存，接口对齐 cachetools.TTLCache 常用方法。

    内部以 OrderedDict 维护插入顺序，get/set 时刷新访问时间；expire 时按 TTL 淘汰。
    所有读写都受实例锁保护，可在事件循环线程与工作线程间安全共享。
    """

    def __init__(self, maxsize: int, ttl: float):
        self.maxsize = maxsize
        self.ttl = ttl
        self._store = OrderedDict()
        self._times = {}
        self._lock = threading.RLock()

    def _is_expired(self, key) -> bool:
        ts = self._times.get(key)
        return ts is None or (time.monotonic() - ts) > self.ttl

    def get(self, key, default=None):
        with self._lock:
            if key not in self._store or self._is_expired(key):
                self._store.pop(key, None)
                self._times.pop(key, None)
                return default
            self._times[key] = time.monotonic()
            self._store.move_to_end(key)
            return self._store[key]

    def set(self, key, value):
        with self._lock:
            self._store[key] = value
            self._times[key] = time.monotonic()
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                old, _ = self._store.popitem(last=False)
                self._times.pop(old, None)

    def pop(self, key, default=None):
        with self._lock:
            value = self._store.pop(key, default)
            self._times.pop(key, None)
            return value

    def clear(self):
        with self._lock:
            self._store.clear()
            self._times.clear()

    def cleanup(self) -> int:
        """清理全部过期条目，返回清理条数"""
        removed = 0
        now = time.monotonic()
        with self._lock:
            for key in list(self._store.keys()):
                ts = self._times.get(key)
                if ts is None or (now - ts) > self.ttl:
                    self._store.pop(key, None)
                    self._times.pop(key, None)
                    removed += 1
        return removed

    def __len__(self):
        with self._lock:
            return len(self._store)

    def __contains__(self, key):
        with self._lock:
            if key not in self._store or self._is_expired(key):
                return False
            return True

    def items(self):
        """返回未过期条目的快照，供遍历使用"""
        with self._lock:
            return [(k, v) for k, v in self._store.items() if not self._is_expired(k)]


# ========== M1-B：活跃房间缓存与连接映射 ==========
# 房间字典结构见 room_routes.create_room：room_id / host_id / mode / is_public /
# scene_id / scene_title / goal_text / max_watchers / punishment_types / created_at /
# strikes_left / distracted_streak_sec / last_frame_ts / last_frame_b64 /
# score_history / watchers(dict[int,str]) 等字段。
# M2-B 追加：scene_config(当前剧本) / dialogue_history(最近8轮对话) / is_locked(惩罚锁房) /
# strikes_used(本局累计扣血次数) / active_punishment(当前启用惩罚类型)。
active_rooms = TTLCache(maxsize=200, ttl=7200)

# 每个房间的实时连接：{room_id: {user_id: WebSocket}}。WebSocket 对象由事件循环线程
# 增删，janitor 也会读取，故配线程锁；键类型用 str 注解仅为可读性，运行时存 WebSocket。
room_connections: dict = {}
room_connections_lock = threading.RLock()


async def _cache_janitor():
    """每 60 秒清理一次活跃房间过期条目与失效连接（防 OOM、防幽灵连接）"""
    while True:
        try:
            await asyncio.sleep(60)
            purged_room = active_rooms.cleanup()
            purged_conn = 0
            now = time.monotonic()
            with room_connections_lock:
                for rid in list(room_connections.keys()):
                    peers = room_connections.get(rid)
                    # 房间已不存在或连接表为空：整体移除
                    if rid not in active_rooms or not peers:
                        room_connections.pop(rid, None)
                        purged_conn += 1
                        continue
                    # 逐条移除过期连接（用 TTL 时间戳近似：连接对象不记录时间，这里仅清理空映射）
            if purged_room or purged_conn:
                logger.info(
                    "[缓存清理] 过期回收：活跃房间 %s 条 / 失效连接房间 %s 个",
                    purged_room, purged_conn,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[缓存清理] 后台清理任务执行异常")


def to_iso_utc(v):
    """把 datetime / 时间戳 / ISO 字符串统一转成带 Z 的 UTC ISO-8601 字符串"""
    from datetime import datetime, timezone
    if v is None:
        return None
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, (int, float)):
        dt = datetime.fromtimestamp(v, tz=timezone.utc)
    elif isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return v
    else:
        return str(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def clear_user_room_cache(user_id: int) -> int:
    """按用户 ID 清除其创建的或作为督学官参与的活跃房间及其连接，返回清理房间数"""
    removed = 0
    with active_rooms._lock:
        for key, value in list(active_rooms._store.items()):
            if not isinstance(value, dict):
                continue
            watchers = value.get("watchers") or {}
            if value.get("host_id") == user_id or user_id in watchers:
                active_rooms._store.pop(key, None)
                active_rooms._times.pop(key, None)
                removed += 1
            with room_connections_lock:
                peers = room_connections.get(key)
                if peers is not None:
                    peers.pop(user_id, None)
                    if not peers:
                        room_connections.pop(key, None)
    return removed
