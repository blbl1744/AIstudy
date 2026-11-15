"""
数据访问层：SQLite 连接上下文管理器、建表/迁移/索引与全部 CRUD 函数。

依赖：config（BASE_DIR、日志）、state（时间格式化）。
说明：建号时需要 bcrypt 哈希，哈希实现归属认证层（auth.py），
      为避免 db ↔ auth 的导入期循环，这里以函数内延迟导入的方式单向下依赖 auth。
"""
import os
import secrets
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

def init_db():
    """初始化数据库（建表 / 历史表迁移 / 索引创建 / 留存清理 / 管理员初始化）"""
    with get_db() as conn:
        _init_db_schema(conn)
    logger.info("数据库初始化完成")

def _init_db_schema(conn):
    """init_db 的实际执行体（传入已开启 WAL 与事务的连接，提交与关闭由 get_db 统一负责）"""
    cursor = conn.cursor()

    # ===== 教师表 =====
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ===== 学生表（学生自己注册）=====
    # 先检查旧表结构是否需要迁移
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    table_exists = cursor.fetchone() is not None

    need_migration = False
    if table_exists:
        # 检查是否有 username 列
        cursor.execute("PRAGMA table_info(students)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'username' not in columns or 'password_hash' not in columns:
            need_migration = True
            logger.warning("检测到旧版 students 表结构，正在执行迁移")

    if need_migration:
        # 备份旧数据
        cursor.execute("ALTER TABLE students RENAME TO students_old")
        # 创建新表
        cursor.execute('''
            CREATE TABLE students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                student_no TEXT DEFAULT '',
                class_name TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 迁移旧数据（为旧学生生成默认用户名和密码）
        # P0-9：不再使用可猜测的旧默认弱口令，改为随机不可预测口令的哈希
        cursor.execute('''
            INSERT INTO students (id, username, password_hash, name, student_no, class_name, created_at)
            SELECT id, 
                   'student_' || id, 
                   ?, 
                   COALESCE(name, '学生' || id), 
                   COALESCE(student_no, ''), 
                   COALESCE(class_name, ''), 
                   COALESCE(created_at, CURRENT_TIMESTAMP)
            FROM students_old
        ''', (_hash_password(secrets.token_urlsafe(16)),))
        # 删除旧表
        cursor.execute("DROP TABLE students_old")
        logger.info("students 表迁移完成")
    else:
        # 直接创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                student_no TEXT DEFAULT '',
                class_name TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    # ===== 行为记录表 =====
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
            model_prediction TEXT,  -- 新增：模型预测结果
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expire_at TIMESTAMP,    -- P0-10：数据留存到期时间（默认写入后 30 天）
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')

    # ===== P0-10：行为记录 30 天留存策略 =====
    # 1) 为已存在的旧表幂等补充 expire_at 字段（重复执行不会报错）
    cursor.execute("PRAGMA table_info(behavior_records)")
    _br_columns = [col[1] for col in cursor.fetchall()]
    if "expire_at" not in _br_columns:
        cursor.execute("ALTER TABLE behavior_records ADD COLUMN expire_at TIMESTAMP")
        logger.info("behavior_records 表已补充 expire_at 字段")
    # 2) 回填历史记录的到期时间（写入时间 + 30 天）
    cursor.execute(
        "UPDATE behavior_records SET expire_at = datetime(created_at, '+30 days') "
        "WHERE expire_at IS NULL AND created_at IS NOT NULL"
    )
    # 3) 清理已过期记录（未成年人隐私最小化：超期数据不留存）
    cursor.execute(
        "DELETE FROM behavior_records "
        "WHERE (expire_at IS NOT NULL AND expire_at < CURRENT_TIMESTAMP) "
        "   OR (expire_at IS NULL AND created_at < datetime('now', '-30 days'))"
    )
    if cursor.rowcount and cursor.rowcount > 0:
        logger.info("已清理 %s 条超过 30 天留存期的行为记录", cursor.rowcount)

    # ===== 管理员账号初始化（P0-4：不再内置 teacher 默认弱口令）=====
    # 仅当环境变量 AI_STUDY_ADMIN_PASSWORD 非空、且库中不存在 teacher 账号时才创建
    _admin_password = (os.getenv("AI_STUDY_ADMIN_PASSWORD") or "").strip()
    cursor.execute("SELECT COUNT(*) FROM teachers WHERE username = 'teacher'")
    if cursor.fetchone()[0] == 0:
        if _admin_password:
            cursor.execute(
                "INSERT INTO teachers (username, password_hash) VALUES (?, ?)",
                ("teacher", _hash_password(_admin_password))
            )
            logger.info("已依据环境变量 AI_STUDY_ADMIN_PASSWORD 初始化管理员账号：teacher")
        else:
            logger.info("未配置环境变量 AI_STUDY_ADMIN_PASSWORD，跳过管理员账号初始化")

    # ===== P2-4：复合索引（幂等创建）=====
    # 用途：学生最新一条行为记录的查询（WHERE student_id=? ORDER BY created_at DESC LIMIT 1）
    # 由全表扫描 + 临时排序，降为一次索引定位；教师端批量取最新记录的分组聚合也能借助
    # student_id 有序前缀完成 GROUP BY，避免额外排序。
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_behavior_student_time "
        "ON behavior_records(student_id, created_at DESC)"
    )
    logger.info("复合索引 idx_behavior_student_time 已就绪")

# ========== 数据库操作 ==========
# P2-4：get_db 改为上下文管理器，统一连接开箱参数与自动事务收尾
@contextmanager
def get_db():
    """数据库连接上下文管理器

    统一行为（每次取连接即就位，避免各处漏设 PRAGMA）：
      1) journal_mode=WAL：读写并发互不阻塞，学生持续写行为记录时教师端查询不会长时间被锁；
      2) foreign_keys=ON：SQLite 默认不启用外键约束，这里显式打开以保证引用完整性；
      3) synchronous=NORMAL：WAL 模式下兼顾安全与写入性能的平衡点；
      4) 正常退出自动 commit()，抛异常自动 rollback()，finally 必定 close()。
    使用方式统一为 `with get_db() as conn:`，不再需要散落的手动 close()。
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

def get_teacher_by_username(username: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM teachers WHERE username = ?", (username,))
        return cursor.fetchone()

def get_student_by_username(username: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE username = ?", (username,))
        return cursor.fetchone()

def create_student(username: str, password: str, name: str, student_no: str, class_name: str):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO students (username, password_hash, name, student_no, class_name) VALUES (?, ?, ?, ?, ?)",
                (username, _hash_password(password), name, student_no, class_name)
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        # 用户名唯一约束冲突：交由上层返回友好提示
        return None

def get_all_students():
    """教师获取所有学生"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, name, student_no, class_name, created_at FROM students ORDER BY created_at DESC")
        students = cursor.fetchall()
    result = []
    for s in students:
        item = dict(s)
        item["created_at"] = to_iso_utc(item.get("created_at"))
        result.append(item)
    return result

def get_student_by_id(student_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, name, student_no, class_name, created_at FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
    if not student:
        return None
    item = dict(student)
    item["created_at"] = to_iso_utc(item.get("created_at"))
    return item

def save_behavior_record(student_id: int, data: dict):
    # P0-10：写入时即设定 30 天留存到期时间（SQLite datetime('now') 为 UTC，与 created_at 默认值一致）
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO behavior_records
               (student_id, focus_score, distraction_count, head_pose, body_stability, fatigue_level, eye_focus, suggestion, model_prediction, expire_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '+30 days'))""",
            (student_id, data.get("focus_score"), data.get("distraction_count"),
             data.get("head_pose"), data.get("body_stability"), data.get("fatigue_level"),
             data.get("eye_focus"), data.get("suggestion"), data.get("model_prediction"))
        )

def normalize_record_times(record: dict) -> dict:
    """P1-7：把行为记录中的时间字段统一转成带 Z 的 ISO-8601，前端按本地时区正确渲染"""
    if not record:
        return record
    for key in ("created_at", "expire_at"):
        if key in record:
            record[key] = to_iso_utc(record.get(key))
    return record

def get_student_behavior_records(student_id: int, limit: int = 50):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM behavior_records WHERE student_id = ? ORDER BY created_at DESC LIMIT ?",
            (student_id, limit)
        )
        records = cursor.fetchall()
    return [normalize_record_times(dict(r)) for r in records]

def get_student_latest_record(student_id: int):
    """获取学生最新一条行为记录（依赖复合索引 idx_behavior_student_time，走索引定位无需回扫）"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM behavior_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
            (student_id,)
        )
        record = cursor.fetchone()
    return normalize_record_times(dict(record)) if record else None

def get_latest_records_for_students(student_ids):
    """一次性批量取出多个学生各自的最新一条行为记录（P2-4：消除教师端列表的 N+1 查询）

    原实现是在教师端学生列表的 for 循环里逐个调用 get_student_latest_record()，
    20 个学生就会触发 21 次数据库查询（N+1）。这里改为「一条 GROUP BY 聚合 + JOIN」：
    先用子查询按 student_id 分组取 MAX(id)（id 自增，最大值即该学生最新写入的记录），
    再 JOIN 回主表取回完整字段，无论多少个学生查库次数恒为 1 次。
    聚合子查询只涉及 (student_id, id)，可借助复合索引 idx_behavior_student_time 的
    student_id 有序前缀完成分组，避免全表排序。
    """
    if not student_ids:
        return {}
    id_list = [int(x) for x in student_ids]
    placeholders = ",".join("?" for _ in id_list)
    sql = (
        "SELECT b.* FROM behavior_records b "
        "JOIN ("
        "  SELECT student_id, MAX(id) AS latest_id FROM behavior_records "
        "  WHERE student_id IN (" + placeholders + ") GROUP BY student_id"
        ") m ON m.latest_id = b.id"
    )
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(id_list))
        rows = cursor.fetchall()
    latest_map = {}
    for r in rows:
        item = normalize_record_times(dict(r))
        latest_map[item.get("student_id")] = item
    return latest_map
