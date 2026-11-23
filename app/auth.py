"""
认证授权层：bcrypt 密码哈希、登录失败限速、JWT 签发校验、get_current_user 依赖与请求模型。

M1-A：取消师生角色概念，JWT 载荷统一为 {"sub", "user_id", "nickname"}。
依赖：config（JWT 配置）、db（读取/更新统一账号密码哈希）。
"""
import hashlib
import hmac
import threading
import time
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .config import JWT_ALGORITHM, JWT_EXPIRE_DAYS, JWT_SECRET_KEY, logger
from .db import get_db

# HTTP Bearer 认证器：供 get_current_user 依赖注入使用
security = HTTPBearer()

# ========== JWT 认证工具 ==========
def create_jwt_token(user_id: int, username: str, nickname: str) -> str:
    """创建统一账号 JWT：载荷只含用户名、用户 ID 与昵称，不再携带角色"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub": username,
        "user_id": user_id,
        "nickname": nickname,
        "exp": expire,
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    """验证 JWT 令牌"""
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token已过期")
    except jwt.InvalidTokenError:  # PyJWT 2.x：JWTError 已移除，改用 InvalidTokenError 基类
        raise HTTPException(status_code=401, detail="无效的Token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前登录用户（JWT 载荷）"""
    return verify_jwt_token(credentials.credentials)


# ========== 密码安全（P0-4） ==========
def hash_password(password: str) -> str:
    """生成密码哈希：bcrypt（随机盐，cost=12）"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def is_bcrypt_hash(hashed: str) -> bool:
    """判断哈希是否为 bcrypt 格式（$2a$ / $2b$ / $2y$）"""
    return isinstance(hashed, str) and hashed.startswith("$2")


def verify_password(password: str, hashed: str) -> bool:
    """校验密码：bcrypt 哈希用 bcrypt.checkpw；历史无盐 SHA256 哈希用常量时间比对"""
    if not hashed:
        return False
    if is_bcrypt_hash(hashed):
        try:
            return bcrypt.checkpw((password or "").encode("utf-8"), hashed.encode("utf-8"))
        except (ValueError, TypeError):
            return False
    legacy_hash = hashlib.sha256((password or "").encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy_hash, hashed)


def upgrade_password_hash(user_id: int, new_hash: str) -> None:
    """将统一账号的密码哈希原地升级（仅用于历史 SHA256 → bcrypt 的平滑迁移）"""
    try:
        with get_db() as conn:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
    except Exception:
        # 内部堆栈只进日志，绝不返回给前端
        logger.exception("密码哈希升级失败（user_id=%s）", user_id)


# ========== 登录失败限速（P0-4） ==========
_LOGIN_MAX_FAILS = 5          # 连续失败次数上限
_LOGIN_LOCK_SECONDS = 900     # 锁定时长：15 分钟
_login_fail = {}              # 结构：{"username": {"count": int, "locked_until": float}}
_login_fail_lock = threading.Lock()


def _login_key(username: str) -> str:
    """限速键：用户名统一小写、去空格，避免大小写绕过"""
    return (username or "").strip().lower()


def _raise_if_login_locked(username: str) -> None:
    """锁定期内直接返回 429"""
    key = _login_key(username)
    now = time.time()
    remain = 0
    with _login_fail_lock:
        record = _login_fail.get(key)
        if not record:
            return
        locked_until = record.get("locked_until", 0)
        if locked_until and locked_until > now:
            remain = int(locked_until - now) + 1
        elif locked_until:
            _login_fail.pop(key, None)
            return
        else:
            return
    raise HTTPException(status_code=429, detail=f"登录失败次数过多，请在 {remain} 秒后重试")


def _record_login_failure(username: str) -> None:
    """记录一次登录失败，达到上限则锁定"""
    key = _login_key(username)
    now = time.time()
    with _login_fail_lock:
        record = _login_fail.setdefault(key, {"count": 0, "locked_until": 0.0})
        if record.get("locked_until") and record["locked_until"] <= now:
            record["count"] = 0
            record["locked_until"] = 0.0
        record["count"] += 1
        if record["count"] >= _LOGIN_MAX_FAILS:
            record["locked_until"] = now + _LOGIN_LOCK_SECONDS


def _clear_login_failures(username: str) -> None:
    """登录成功后清零失败计数"""
    with _login_fail_lock:
        _login_fail.pop(_login_key(username), None)


# ========== 数据模型 ==========
# 昵称会回显到个人主页/房间，必须做长度与字符白名单约束（存储型 XSS 防护）
# 仅允许中文、字母、数字、下划线、间隔号、连字符与空白，长度 1~32
NAME_PATTERN = r"^[\u4e00-\u9fa5A-Za-z0-9_·\-\s]{1,32}$"


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=64)
    nickname: str = Field(..., min_length=1, max_length=32, pattern=NAME_PATTERN)


class LoginRequest(BaseModel):
    username: str
    password: str
