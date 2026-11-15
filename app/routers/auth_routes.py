"""
路由层 · 认证：学生/教师注册、登录、当前用户信息（/api/me）。"""
import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from ..auth import (
    LoginRequest,
    StudentRegisterRequest,
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
from ..db import create_student, get_db, get_student_by_username, get_teacher_by_username

router = APIRouter()


# --- 学生注册 ---
@router.post("/api/student/register")
async def student_register(request: StudentRegisterRequest):
    if len(request.username) < 3 or len(request.password) < 6:
        raise HTTPException(status_code=400, detail="用户名至少3位，密码至少6位")
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="请输入真实姓名")

    student_id = create_student(request.username, request.password, request.name, request.student_no, request.class_name)
    if student_id is None:
        raise HTTPException(status_code=400, detail="用户名已存在")

    return {"message": "注册成功", "student_id": student_id}

# --- 教师注册 ---
@router.post("/api/teacher/register")
async def teacher_register(request: LoginRequest):
    """教师注册"""
    if len(request.username) < 3 or len(request.password) < 6:
        raise HTTPException(status_code=400, detail="用户名至少3位，密码至少6位")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO teachers (username, password_hash) VALUES (?, ?)",
                (request.username, hash_password(request.password))
            )
            teacher_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="用户名已存在")
    return {"message": "注册成功", "teacher_id": teacher_id}

# --- 学生登录 ---
@router.post("/api/student/login")
async def student_login(request: LoginRequest):
    # P0-4：失败次数达到上限则直接拒绝（HTTP 429）
    _raise_if_login_locked("student", request.username)

    student = get_student_by_username(request.username)
    if not student or not verify_password(request.password, student["password_hash"]):
        _record_login_failure("student", request.username)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 登录成功：清零失败计数
    _clear_login_failures("student", request.username)

    # P0-4：历史 SHA256 哈希校验通过后，立即原地升级为 bcrypt
    if not is_bcrypt_hash(student["password_hash"]):
        upgrade_password_hash("student", student["id"], hash_password(request.password))

    token = create_jwt_token(student["id"], "student", request.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": request.username,
        "name": student["name"],
        "role": "student",
        "student_id": student["id"]
    }

# --- 教师登录 ---
@router.post("/api/teacher/login")
async def teacher_login(request: LoginRequest):
    # P0-4：失败次数达到上限则直接拒绝（HTTP 429）
    _raise_if_login_locked("teacher", request.username)

    teacher = get_teacher_by_username(request.username)
    if not teacher or not verify_password(request.password, teacher["password_hash"]):
        _record_login_failure("teacher", request.username)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 登录成功：清零失败计数
    _clear_login_failures("teacher", request.username)

    # P0-4：历史 SHA256 哈希校验通过后，立即原地升级为 bcrypt
    if not is_bcrypt_hash(teacher["password_hash"]):
        upgrade_password_hash("teacher", teacher["id"], hash_password(request.password))

    token = create_jwt_token(teacher["id"], "teacher", request.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": request.username,
        "role": "teacher"
    }

# --- 获取当前用户信息 ---
@router.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    role = user.get("role")
    if role == "student":
        student = get_student_by_username(user["sub"])
        if not student:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"username": student["username"], "name": student["name"], "role": "student", "student_id": student["id"]}
    elif role == "teacher":
        teacher = get_teacher_by_username(user["sub"])
        if not teacher:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"username": teacher["username"], "role": "teacher"}
