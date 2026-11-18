"""
数据访问层：SQLite 连接上下文管理器、建表/迁移/索引与统一账号相关 CRUD。

M1-A：合并师生双表为 users 统一账号，新增 study_sessions 战报表与 scene_cache 场景缓存；
behavior_records 保留原有字段并幂等补加 session_id 列。
依赖：config（BASE_DIR、日志）、state（时间格式化）。
说明：建号时需要 bcrypt 哈希，哈希实现归属认证层（auth.py），
      为避免 db ↔ auth 的导入期循环，这里以函数内延迟导入的方式单向下依赖 auth。
"""
import os
import sqlite3
from contextlib import contextmanager

from .config import BASE_DIR, logger
from .state import to_iso_utc


def _hash_password(password: str) -> str:
    """密码哈希（延迟导入 auth.hash_password，保证模块加载期依赖方向仍为 db ← auth）"""
    from .auth import hash_password
    return hash_password(password)


# ========== 数据库初始化 ==========
# P3-4：支持环境变量 AI_STUDY_DB 覆盖数据库文件路径（容器挂载卷 / 自动化测试隔离必需）
DB_FILE = (os.getenv("AI_STUDY_DB") or "").strip() or os.path.join(BASE_DIR, "ai_study_buddy.db")

# 军衔阈值（从小到大）：累计专注分钟数达到阈值即晋升
_RANK_THRESHOLDS = (
    (0, "见习学员"),
    (60, "列兵"),
    (300, "士官"),
    (1000, "少尉"),
    (3000, "王牌指挥官"),
)


def calc_rank_title(minutes) -> str:
    """按累计专注分钟数计算军衔（非法输入按 0 处理）"""
    try:
        value = int(minutes or 0)
    except (TypeError, ValueError):
        value = 0
    title = _RANK_THRESHOLDS[0][1]
    for threshold, name in _RANK_THRESHOLDS:
        if value >= threshold:
            title = name
        else:
            break
    return title


def init_db():
    """初始化数据库（建表 / 历史表迁移 / 索引创建 / 留存清理）"""
    with get_db() as conn:
        _init_db_schema(conn)
    logger.info("数据库初始化完成")


def _ensure_columns(cursor, table: str, columns: dict):
    """幂等补列：历史表缺少的字段按给定 DDL 补齐（表名与列名均为内部常量）"""
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            logger.info("%s 表已幂等补加字段 %s", table, name)


def _migrate_legacy_accounts(cursor):
    """把旧 students / teachers 账号并入 users（幂等；旧表保留不删除，便于回溯）"""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('students','teachers')")
    legacy = {row[0] for row in cursor.fetchall()}

    if "students" in legacy:
        cursor.execute("PRAGMA table_info(students)")
        cols = {row[1] for row in cursor.fetchall()}
        if {"id", "username", "password_hash"} <= cols:
            # 行为记录的历史外键指向 students.id，因此学生迁移保留原 id
            nickname_expr = "name" if "name" in cols else "username"
            cursor.execute(
                "INSERT OR IGNORE INTO users (id, username, nickname, password_hash, created_at) "
                f"SELECT id, username, COALESCE(NULLIF({nickname_expr}, ''), username), password_hash, created_at "
                "FROM students"
            )
            logger.info("旧 students 账号已合并至 users")

    if "teachers" in legacy:
        cursor.execute("PRAGMA table_info(teachers)")
        cols = {row[1] for row in cursor.fetchall()}
        if {"username", "password_hash"} <= cols:
            # 教师不再有独立角色，作为普通统一账号迁入；不指定 id，避免与学生 id 冲突
            cursor.execute(
                "INSERT OR IGNORE INTO users (username, nickname, password_hash, created_at) "
                "SELECT username, username, password_hash, created_at FROM teachers"
            )
            logger.info("旧 teachers 账号已合并至 users")


def _init_db_schema(conn):
    """init_db 的实际执行体（传入已开启 WAL 与事务的连接，提交与关闭由 get_db 统一负责）"""
    cursor = conn.cursor()

    # ===== M1-A：统一用户表 =====
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            nickname TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            total_focus_minutes INTEGER NOT NULL DEFAULT 0,
            rank_title TEXT NOT NULL DEFAULT '见习学员',
            shield_count INTEGER NOT NULL DEFAULT 0,
            streak_clean_sessions INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    _ensure_columns(cursor, "users", {
        "nickname": "TEXT NOT NULL DEFAULT ''",
        "total_focus_minutes": "INTEGER NOT NULL DEFAULT 0",
        "rank_title": "TEXT NOT NULL DEFAULT '见习学员'",
        "shield_count": "INTEGER NOT NULL DEFAULT 0",
        "streak_clean_sessions": "INTEGER NOT NULL DEFAULT 0",
    })
    _migrate_legacy_accounts(cursor)

    # ===== M1-A：督学战报记录表 =====
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            scene_title TEXT NOT NULL,
            goal_text TEXT NOT NULL DEFAULT '',
            duration_minutes INTEGER NOT NULL DEFAULT 0,
            avg_score INTEGER,
            strikes_used INTEGER NOT NULL DEFAULT 0,
            punishment_type TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_user_time "
        "ON study_sessions(user_id, created_at DESC)"
    )

    # ===== M1-A：AI 剧本缓存表 =====
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scene_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_name TEXT UNIQUE NOT NULL,
            scene_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ===== 行为采样明细表（保留原字段，幂等补加 session_id）=====
    # 新库外键指向统一用户表；历史库的外键保持原样，不重建表以免丢失数据。
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS behavior_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            focus_score INTEGER,
            distraction_count INTEGER,
            head_pose TEXT,
            body_stability TEXT,
            fatigue_level TEXT,
            eye_focus TEXT,
            suggestion TEXT,
            model_prediction TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expire_at TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES users(id)
        )
    ''')
    _ensure_columns(cursor, "behavior_records", {"session_id": "TEXT", "expire_at": "TIMESTAMP"})

    # 30 天留存策略：回填历史记录到期时间并清理超期数据（隐私最小化）
    cursor.execute(
        "UPDATE behavior_records SET expire_at = datetime(created_at, '+30 days') "
        "WHERE expire_at IS NULL AND created_at IS NOT NULL"
    )
    cursor.execute(
        "DELETE FROM behavior_records "
        "WHERE (expire_at IS NOT NULL AND expire_at < CURRENT_TIMESTAMP) "
        "   OR (expire_at IS NULL AND created_at < datetime('now', '-30 days'))"
    )
    if cursor.rowcount and cursor.rowcount > 0:
        logger.info("已清理 %s 条超过 30 天留存期的行为记录", cursor.rowcount)

    # 保留原复合索引：学生最新一条行为记录查询走索引定位
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_behavior_student_time "
        "ON behavior_records(student_id, created_at DESC)"
    )
    logger.info("复合索引 idx_behavior_student_time 已就绪")


# ========== 数据库连接 ==========
# P2-4：get_db 改为上下文管理器，统一连接开箱参数与自动事务收尾
@contextmanager
def get_db():
    """数据库连接上下文管理器

    统一行为（每次取连接即就位，避免各处漏设 PRAGMA）：
      1) journal_mode=WAL：读写并发互不阻塞；
      2) foreign_keys=ON：显式打开外键约束保证引用完整性；
      3) synchronous=NORMAL：WAL 模式下安全与写入性能的平衡点；
      4) 正常退出自动 commit()，抛异常自动 rollback()，finally 必定 close()。
    """
    conn = sqlite3.connect(DB_FILE, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ========== 统一账号 CRUD ==========
def get_user_by_username(username: str):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def get_user_by_id(user_id: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def create_user(username: str, password: str, nickname: str):
    """创建统一账号；用户名冲突时返回 None，由上层转换为友好提示"""
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, nickname, password_hash) VALUES (?, ?, ?)",
                (username, nickname, _hash_password(password))
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_user_profile(user_id: int):
    """个人成长档案：基础信息 + 实时军衔 + 护盾数 + 最近 20 条战报（时间统一 UTC ISO）"""
    user = get_user_by_id(user_id)
    if not user:
        return None
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, user_id, mode, scene_title, goal_text, duration_minutes, avg_score, "
            "strikes_used, punishment_type, status, created_at FROM study_sessions "
            "WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 20",
            (user_id,)
        ).fetchall()
    sessions = []
    for row in rows:
        item = dict(row)
        item["created_at"] = to_iso_utc(item.get("created_at"))
        sessions.append(item)
    return {
        "id": user["id"],
        "username": user["username"],
        "nickname": user["nickname"],
        "total_focus_minutes": user["total_focus_minutes"],
        "rank_title": calc_rank_title(user["total_focus_minutes"]),
        "shield_count": user["shield_count"],
        "streak_clean_sessions": user["streak_clean_sessions"],
        "created_at": to_iso_utc(user["created_at"]),
        "sessions": sessions,
    }


def delete_user_learning_data(user_id: int) -> dict:
    """清空个人行为采样明细与战报；房间缓存由路由层通过 state 独立清理"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 行为明细的历史字段名为 student_id，M1-A 起复用它承载统一账号 id
        cursor.execute("DELETE FROM behavior_records WHERE student_id = ?", (user_id,))
        behavior_deleted = cursor.rowcount
        cursor.execute("DELETE FROM study_sessions WHERE user_id = ?", (user_id,))
        sessions_deleted = cursor.rowcount
    return {
        "behavior_records_deleted": behavior_deleted,
        "study_sessions_deleted": sessions_deleted,
    }


# ========== 行为明细与战报辅助 ==========
def save_behavior_record(user_id: int, data: dict, session_id: str = None):
    """写入单条行为采样；session_id 为可选的单局战报关联键"""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO behavior_records
               (student_id, session_id, focus_score, distraction_count, head_pose, body_stability,
                fatigue_level, eye_focus, suggestion, model_prediction, expire_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '+30 days'))""",
            (user_id, session_id, data.get("focus_score"), data.get("distraction_count"),
             data.get("head_pose"), data.get("body_stability"), data.get("fatigue_level"),
             data.get("eye_focus"), data.get("suggestion"), data.get("model_prediction"))
        )


def normalize_record_times(record: dict) -> dict:
    """把行为记录中的时间字段统一转成带 Z 的 ISO-8601，前端按本地时区正确渲染"""
    if not record:
        return record
    for key in ("created_at", "expire_at"):
        if key in record:
            record[key] = to_iso_utc(record.get(key))
    return record


def get_user_behavior_records(user_id: int, limit: int = 50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM behavior_records WHERE student_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    return [normalize_record_times(dict(row)) for row in rows]


def get_user_latest_record(user_id: int):
    """最新一条行为采样（依赖复合索引 idx_behavior_student_time）"""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM behavior_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        ).fetchone()
    return normalize_record_times(dict(row)) if row else None
