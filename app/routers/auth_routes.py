"""
路由层 · 统一认证与个人成长接口（M1-A）。

提供注册、登录、个人档案查询与个人数据清空，不再区分师生角色。
"""
from fastapi import APIRouter, Depends, HTTPException

from ..auth import (
    LoginRequest,
    RegisterRequest,
    _clear_login_failures,
    _raise_if_login_locked,
    _record_login_failure,
    create_jwt_token,
    get_current_user,
    hash_password,
    is_bcrypt_hash,
    upgrade_password_hash,
    verify_password,
)
from ..db import (
    create_user,
    delete_user_learning_data,
    get_user_by_username,
    get_user_profile,
)
from ..state import clear_user_room_cache

router = APIRouter()


def _current_user_id(user: dict) -> int:
    """从 JWT 载荷取统一账号 ID；缺失则视为无效凭证"""
    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="无效的Token")
    return user_id


@router.post("/api/auth/register")
async def register(request: RegisterRequest):
    """统一注册：用户名、密码、昵称（昵称已由模型做 XSS 字符白名单校验）"""
    user_id = create_user(request.username, request.password, request.nickname)
    if user_id is None:
        raise HTTPException(status_code=400, detail="用户名已存在")
    return {"message": "注册成功", "user_id": user_id}


@router.post("/api/auth/login")
async def login(request: LoginRequest):
    """统一登录：失败限速 + bcrypt 校验 + 签发无角色 JWT"""
    _raise_if_login_locked(request.username)

    user = get_user_by_username(request.username)
    if not user or not verify_password(request.password, user["password_hash"]):
        _record_login_failure(request.username)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    _clear_login_failures(request.username)
    # 历史 SHA256 哈希校验通过后，立即原地升级为 bcrypt
    if not is_bcrypt_hash(user["password_hash"]):
        upgrade_password_hash(user["id"], hash_password(request.password))

    token = create_jwt_token(user["id"], user["username"], user["nickname"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user["id"],
        "username": user["username"],
        "nickname": user["nickname"],
    }


@router.get("/api/users/me/profile")
async def get_my_profile(user: dict = Depends(get_current_user)):
    """个人成长档案：账号信息、军衔、护盾数及最近 20 条战报"""
    profile = get_user_profile(_current_user_id(user))
    if not profile:
        raise HTTPException(status_code=404, detail="用户不存在")
    return profile


@router.delete("/api/users/me/data")
async def delete_my_data(user: dict = Depends(get_current_user)):
    """一键清空个人行为记录、战报与活跃房间缓存（账号本身保留）"""
    user_id = _current_user_id(user)
    deleted = delete_user_learning_data(user_id)
    room_cleared = clear_user_room_cache(user_id)
    return {
        "message": "个人数据已清空",
        "behavior_records_deleted": deleted["behavior_records_deleted"],
        "study_sessions_deleted": deleted["study_sessions_deleted"],
        "room_cache_cleared": room_cleared,
    }
