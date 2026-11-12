# -*- coding: utf-8 -*-
"""
AI学伴 · 1对1专注度智能监护系统
- 学生注册/登录 → 进入自己的专注度监控页面（摄像头）
- 教师登录 → 查看所有学生列表 → 点击学生查看其上课行为
- 集成训练好的专注度模型进行预测
"""

""" AI学伴 · 1对1专注度智能监护系统 """

import asyncio
import base64
import json
import random
import time
import math
import os
import sys
import pickle
import numpy as np
import hashlib
import sqlite3
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import cv2                         
import mediapipe as mp          
# ========== 先加载配置 ==========
try:
    from config import (
        VOLC_ACCESS_KEY, VOLC_SECRET_KEY, VOLC_TOKEN_URL, VOLC_ASR_ENDPOINT,
        HOST, PORT, HEAD_DOWN_THRESHOLD, BODY_SWAY_LIMIT, EYE_CLOSED_FRAMES,
        DISTRACTION_INTERVAL, WARNING_LEVELS, ASR_CONFIG, TTS_CONFIG,
        POSE_CONFIG, POSE_THRESHOLDS, JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_DAYS
    )
except ImportError:
    VOLC_ACCESS_KEY = "YOUR_ACCESS_KEY"
    VOLC_SECRET_KEY = "YOUR_SECRET_KEY"
    VOLC_TOKEN_URL = "https://open.volcengineapi.com/api/v3/auth/token"
    VOLC_ASR_ENDPOINT = "https://openspeech.bytedance.com/api/v1/auc/recognize"
    HOST = "0.0.0.0"
    PORT = 8001
    HEAD_DOWN_THRESHOLD = 25
    BODY_SWAY_LIMIT = 15
    EYE_CLOSED_FRAMES = 10
    DISTRACTION_INTERVAL = 5
    WARNING_LEVELS = {"mild": 60, "moderate": 40, "severe": 20}
    ASR_CONFIG = {"language": "zh-CN", "sample_rate": 16000, "silence_timeout": 120, "enable_volcengine": True, "enable_browser_asr": True}
    TTS_CONFIG = {"enabled": True, "language": "zh-CN", "rate": 0.9, "pitch": 1.1, "volume": 0.8, "cooldown": 30}
    POSE_CONFIG = {"static_image_mode": True, "model_complexity": 1, "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5}
    POSE_THRESHOLDS = {"head_tilt_minor": 10, "head_tilt_major": 20, "head_drop_minor": 0.6, "head_drop_major": 0.3, "body_sway_minor": 8, "body_sway_major": 15, "shoulder_level_minor": 0.08, "shoulder_level_major": 0.15, "consecutive_low_focus": 3}
    JWT_SECRET_KEY = "your-secret-key-change-in-production"
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_DAYS = 7
# ========== MediaPipe 姿态识别 ==========
app = FastAPI()
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MediaPipe 初始化 ==========
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.3
)

# ========== 新增：Face Mesh 初始化 ==========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)


# ========== 模型加载（全局缓存，只加载一次）==========
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "focus_model.pkl")
_cached_model = None
_cached_scaler = None
_cached_model_loaded = False

def load_trained_model():
    """加载训练好的模型（全局缓存）"""
    global _cached_model, _cached_model_loaded
    if _cached_model_loaded:
        return _cached_model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                _cached_model = pickle.load(f)
            _cached_model_loaded = True
            print(f"✅ 专注度模型已加载: {MODEL_PATH}")
            return _cached_model
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")
    _cached_model_loaded = True  # 标记已尝试过，避免重复尝试
    return None

def predict_focus(features):
    """预测专注度 - 支持标准化"""
    global _cached_model, _cached_scaler
    if _cached_model is None:
        return None, 0.5
    try:
        features = features.reshape(1, -1)
        if _cached_scaler is not None:
            features = _cached_scaler.transform(features)
        proba = _cached_model.predict_proba(features)[0] if hasattr(_cached_model, 'predict_proba') else [0.5, 0.5]
        raw_focused_prob = round(float(proba[1]), 4)
        # 模型概率映射：将原始概率映射到更合理的范围
        # 原始范围 [0, 1] → 映射到 [0.35, 0.98]，使专注时模型分能达到95+
        focused_prob = min(0.98, max(0.15, 0.35 + raw_focused_prob * 0.63))
        distracted_prob = 1.0 - focused_prob
        return {
            "is_focused": focused_prob > 0.5,
            "confidence": focused_prob if focused_prob > 0.5 else distracted_prob,
            "label": "专注" if focused_prob > 0.5 else "分心",
            "prob_focused": focused_prob,
            "prob_distracted": distracted_prob
        }, focused_prob
    except Exception as e:
        print(f"WARNING: model predict error: {e}")
        return None, 0.5

def extract_pose_features(image_path):
    """从图片中提取姿态特征用于模型预测"""
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark

        def get_landmark(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])

        nose = get_landmark(0)
        left_eye = get_landmark(2)
        right_eye = get_landmark(5)
        left_shoulder = get_landmark(11)
        right_shoulder = get_landmark(12)
        left_ear = get_landmark(7)
        right_ear = get_landmark(8)
        left_hip = get_landmark(23)
        right_hip = get_landmark(24)

        features = []
        head_tilt = np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
        features.append(head_tilt)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        head_relative = nose - shoulder_center
        features.extend(head_relative.tolist())
        eye_distance = np.linalg.norm(left_eye - right_eye)
        features.append(eye_distance)
        shoulder_level = abs(left_shoulder[1] - right_shoulder[1])
        features.append(shoulder_level)
        body_angle = np.arctan2(right_shoulder[1] - left_shoulder[1],
                                right_shoulder[0] - left_shoulder[0])
        features.append(body_angle)
        head_shoulder_dist = np.linalg.norm(nose - shoulder_center)
        features.append(head_shoulder_dist)
        eye_center = (left_eye + right_eye) / 2
        eye_shoulder_dist = np.linalg.norm(eye_center - shoulder_center)
        features.append(eye_shoulder_dist)
        hip_center = (left_hip + right_hip) / 2
        features.extend(hip_center.tolist())
        body_lean = shoulder_center[2] - hip_center[2]
        features.append(body_lean)
        return np.array(features)
    except Exception as e:
        print(f"⚠️ 特征提取失败: {e}")
        return None

def extract_pose_features_from_frame(frame):
    """从内存中的视频帧直接提取姿态特征（避免磁盘IO）"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark

        def get_landmark(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])

        nose = get_landmark(0)
        left_eye = get_landmark(2)
        right_eye = get_landmark(5)
        left_shoulder = get_landmark(11)
        right_shoulder = get_landmark(12)
        left_ear = get_landmark(7)
        right_ear = get_landmark(8)
        left_hip = get_landmark(23)
        right_hip = get_landmark(24)

        features = []
        head_tilt = np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
        features.append(head_tilt)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        head_relative = nose - shoulder_center
        features.extend(head_relative.tolist())
        eye_distance = np.linalg.norm(left_eye - right_eye)
        features.append(eye_distance)
        shoulder_level = abs(left_shoulder[1] - right_shoulder[1])
        features.append(shoulder_level)
        body_angle = np.arctan2(right_shoulder[1] - left_shoulder[1],
                                right_shoulder[0] - left_shoulder[0])
        features.append(body_angle)
        head_shoulder_dist = np.linalg.norm(nose - shoulder_center)
        features.append(head_shoulder_dist)
        eye_center = (left_eye + right_eye) / 2
        eye_shoulder_dist = np.linalg.norm(eye_center - shoulder_center)
        features.append(eye_shoulder_dist)
        hip_center = (left_hip + right_hip) / 2
        features.extend(hip_center.tolist())
        body_lean = shoulder_center[2] - hip_center[2]
        features.append(body_lean)
        return np.array(features)
    except Exception as e:
        print(f"⚠️ 内存特征提取失败: {e}")
        return None

# ========== 数据库初始化 ==========
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_study_buddy.db")

def init_db():
    """初始化数据库（含迁移逻辑）"""
    conn = sqlite3.connect(DB_FILE)
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
            print("⚠️ 检测到旧版students表结构，正在迁移...")

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
        ''', (hash_password("123456"),))
        # 删除旧表
        cursor.execute("DROP TABLE students_old")
        print("✅ students表迁移完成")
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
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')

    # ===== 插入默认教师账号 =====
    cursor.execute("SELECT COUNT(*) FROM teachers WHERE username = 'teacher'")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO teachers (username, password_hash) VALUES (?, ?)",
            ("teacher", hash_password("123456"))
        )

    conn.commit()
    conn.close()
    print("✅ 数据库初始化完成（默认教师账号: teacher / 123456）")

# ========== JWT 认证工具 ==========
def create_jwt_token(user_id: int, role: str, username: str) -> str:
    """创建JWT令牌，包含角色信息"""
    import jwt
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub": username,
        "user_id": user_id,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    """验证JWT令牌"""
    import jwt
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token已过期")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="无效的Token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前登录用户"""
    token = credentials.credentials
    return verify_jwt_token(token)

def hash_password(password: str) -> str:
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

# ========== 数据模型 ==========
class StudentRegisterRequest(BaseModel):
    username: str
    password: str
    name: str
    student_no: str = ""
    class_name: str = ""

class LoginRequest(BaseModel):
    username: str
    password: str

# ========== 数据库操作 ==========
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def get_teacher_by_username(username: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM teachers WHERE username = ?", (username,))
    teacher = cursor.fetchone()
    conn.close()
    return teacher

def get_student_by_username(username: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE username = ?", (username,))
    student = cursor.fetchone()
    conn.close()
    return student

def create_student(username: str, password: str, name: str, student_no: str, class_name: str):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (username, password_hash, name, student_no, class_name) VALUES (?, ?, ?, ?, ?)",
            (username, hash_password(password), name, student_no, class_name)
        )
        conn.commit()
        student_id = cursor.lastrowid
        conn.close()
        return student_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def get_all_students():
    """教师获取所有学生"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, student_no, class_name, created_at FROM students ORDER BY created_at DESC")
    students = cursor.fetchall()
    conn.close()
    return [dict(s) for s in students]

def get_student_by_id(student_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, student_no, class_name, created_at FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()
    conn.close()
    return dict(student) if student else None

def save_behavior_record(student_id: int, data: dict):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO behavior_records
           (student_id, focus_score, distraction_count, head_pose, body_stability, fatigue_level, eye_focus, suggestion, model_prediction)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (student_id, data.get("focus_score"), data.get("distraction_count"),
         data.get("head_pose"), data.get("body_stability"), data.get("fatigue_level"),
         data.get("eye_focus"), data.get("suggestion"), data.get("model_prediction"))
    )
    conn.commit()
    conn.close()

def get_student_behavior_records(student_id: int, limit: int = 50):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM behavior_records WHERE student_id = ? ORDER BY created_at DESC LIMIT ?",
        (student_id, limit)
    )
    records = cursor.fetchall()
    conn.close()
    return [dict(r) for r in records]

def get_student_latest_record(student_id: int):
    """获取学生最新一条行为记录"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM behavior_records WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
        (student_id,)
    )
    record = cursor.fetchone()
    conn.close()
    return dict(record) if record else None

# ========== API 路由 ==========

# --- 学生注册 ---
@app.post("/api/student/register")
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
@app.post("/api/teacher/register")
async def teacher_register(request: LoginRequest):
    """教师注册"""
    if len(request.username) < 3 or len(request.password) < 6:
        raise HTTPException(status_code=400, detail="用户名至少3位，密码至少6位")
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO teachers (username, password_hash) VALUES (?, ?)",
            (request.username, hash_password(request.password))
        )
        conn.commit()
        teacher_id = cursor.lastrowid
        conn.close()
        return {"message": "注册成功", "teacher_id": teacher_id}
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="用户名已存在")

# --- 学生登录 ---
@app.post("/api/student/login")
async def student_login(request: LoginRequest):
    student = get_student_by_username(request.username)
    if not student or student["password_hash"] != hash_password(request.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

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
@app.post("/api/teacher/login")
async def teacher_login(request: LoginRequest):
    teacher = get_teacher_by_username(request.username)
    if not teacher or teacher["password_hash"] != hash_password(request.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_jwt_token(teacher["id"], "teacher", request.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": request.username,
        "role": "teacher"
    }

# --- 获取当前用户信息 ---
@app.get("/api/me")
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

# --- 教师获取所有学生列表 ---
@app.get("/api/teacher/students")
async def teacher_list_students(user: dict = Depends(get_current_user)):
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    students = get_all_students()
    # 为每个学生附加最新行为记录和在线状态
    for s in students:
        latest = get_student_latest_record(s["id"])
        s["latest_record"] = latest
        # 检查在线状态（最近10秒有视频帧更新）
        cache = get_student_video_cache(s["id"])
        if cache and (time.time() - cache["timestamp"]) < 10:
            s["online"] = True
            s["live_focus_score"] = cache["focus_score"]
        else:
            s["online"] = False
            s["live_focus_score"] = None
    return {"students": students}

# --- 教师查看某个学生的行为记录 ---
@app.get("/api/teacher/students/{student_id}/records")
async def teacher_view_records(student_id: int, user: dict = Depends(get_current_user)):
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    records = get_student_behavior_records(student_id)
    return {"student": student, "records": records}

# --- 教师获取学生实时视频帧 ---
@app.get("/api/teacher/students/{student_id}/live")
async def teacher_get_student_live(student_id: int, user: dict = Depends(get_current_user)):
    """教师获取学生的实时摄像头画面"""
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问")
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    
    cache = get_student_video_cache(student_id)
    if not cache:
        return {
            "student_id": student_id,
            "online": False,
            "message": "学生未在线或未开启监控"
        }
    
    # 检查是否在最近10秒有更新
    is_online = (time.time() - cache["timestamp"]) < 10
    
    return {
        "student_id": student_id,
        "online": is_online,
        "image": cache["image"],
        "focus_score": cache["focus_score"],
        "timestamp": cache["timestamp"]
    }

# --- 学生保存自己的行为数据 ---
@app.post("/api/student/behavior")
async def student_save_behavior(data: dict, user: dict = Depends(get_current_user)):
    if user.get("role") != "student":
        raise HTTPException(status_code=403, detail="仅学生可访问")
    student_id = user.get("user_id")
    save_behavior_record(student_id, data)
    return {"message": "行为数据已保存"}

# ========== 火山引擎语音识别封装 ==========
import requests

class VolcEngineASR:
    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key
        self.token_url = VOLC_TOKEN_URL
        self.asr_url = VOLC_ASR_ENDPOINT

    def get_token(self):
        try:
            resp = requests.post(self.token_url, data={
                "grant_type": "client_credentials",
                "client_id": self.access_key,
                "client_secret": self.secret_key
            }, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("access_token", "")
        except Exception as e:
            print(f"⚠️ 获取火山引擎Token失败: {e}")
        return ""

    def recognize_audio(self, audio_bytes, sample_rate=16000):
        if not self.access_key or self.access_key == "YOUR_ACCESS_KEY":
            return None
        token = self.get_token()
        if not token:
            return None
        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer;{token}"}
            payload = {
                "app": {"appid": "default", "cluster": "volcengine_streaming_common"},
                "user": {"uid": "ai_study_buddy"},
                "audio": {"format": "wav", "rate": sample_rate, "language": "zh-CN", "bits": 16, "channel": 1, "codec": "raw"},
                "request": {"reqid": f"req_{int(time.time())}", "sequence": 1, "nbest": 1, "text": ""},
                "audio_data": base64.b64encode(audio_bytes).decode('utf-8')
            }
            resp = requests.post(self.asr_url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                return result.get("result", {}).get("text", "")
        except Exception as e:
            print(f"⚠️ 语音识别失败: {e}")
        return None

asr_engine = VolcEngineASR(VOLC_ACCESS_KEY, VOLC_SECRET_KEY)

# ========== MediaPipe 姿态分析器（集成训练模型） ==========
class MediaPipePoseAnalyzer:
    def __init__(self):
        self.consecutive_low_focus = 0
        self.total_distractions = 0
        self.last_analysis_time = time.time()
        self.prev_landmarks = None
        self.head_tilt_history = []
        self.shoulder_history = []
        self.model_loaded = False
        self.model_n_features = 0

        self.focus_score_history = []
        self.last_valid_score = 80
        # 动作分类历史（用于平滑）
        self.action_history = []
        # 闭眼/疲劳持续时间计数（避免眨眼误检）
        self.eyes_closed_frames = 0
        self.eyes_closed_threshold = 60  # 约2秒（假设30fps）
        self.confirmed_eyes_closed = False

        self.check_model_status()

    def check_model_status(self):
        """检查模型是否已训练"""
        try:
            model = load_trained_model()
            self.model_loaded = model is not None
            if self.model_loaded:
                self.model_n_features = getattr(model, 'n_features_in_', 12)
                print(f"✅ 专注度模型已加载 (特征维度={self.model_n_features})")
            else:
                print("⚠️ 专注度预测模型未训练，将使用规则-based分析")
        except Exception as e:
            print(f"⚠️ 模型检查失败: {e}")
            self.model_loaded = False

    # MediaPipe 上半身关键点索引（用于特征提取）
    UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def _extract_model_features(self, landmarks):
        """从已检测的 landmarks 中提取上半身关键点特征（51维）"""
        try:
            features = []
            for idx in self.UPPER_BODY_INDICES:
                lm = landmarks[idx]
                features.extend([lm.x, lm.y, lm.z])
            return np.array(features)
        except Exception as e:
            print(f"⚠️ 特征提取失败: {e}")
            return None

    def analyze(self, image_base64, scene="selfstudy", student_id=None):
        try:
            img_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return self._fallback_analysis(scene, image_base64)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)
            if not results.pose_landmarks:
                return self._fallback_analysis(scene, image_base64)
            landmarks = results.pose_landmarks.landmark

            # 绘制骨架
            annotated = self._draw_skeleton(frame, landmarks)
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')

            # 提取关键点
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            # 计算传统姿态特征
            head_tilt = self._calc_head_tilt(nose, left_shoulder, right_shoulder)
            head_drop = self._calc_head_drop(nose, left_shoulder, right_shoulder)
            head_pose_status = self._evaluate_head_pose(head_tilt, head_drop)
            shoulder_level = abs(left_shoulder.y - right_shoulder.y)
            body_sway = self._calc_body_sway(left_shoulder, right_shoulder, left_hip, right_hip)
            body_stability_status = self._evaluate_body_stability(shoulder_level, body_sway)
            # ===== 眼神/疲劳：用 Face Mesh 虹膜 + EAR（疲劳优先）=====
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0].landmark
                ear_value = self._calc_ear(face_lm)
                fatigue_status = self._evaluate_fatigue(ear_value)
                # 闭眼持续时间检测：避免眨眼误检
                is_eyes_closed_now = (fatigue_status == "明显疲劳")
                if is_eyes_closed_now:
                    self.eyes_closed_frames += 1
                else:
                    self.eyes_closed_frames = max(0, self.eyes_closed_frames - 2)
                # 只有持续闭眼超过阈值才确认闭眼
                if self.eyes_closed_frames >= self.eyes_closed_threshold:
                    self.confirmed_eyes_closed = True
                elif self.eyes_closed_frames == 0:
                    self.confirmed_eyes_closed = False
                # 第1优先级：确认闭眼→闭眼/走神，否则用疲劳状态
                if self.confirmed_eyes_closed:
                    eye_focus_status = "闭眼/走神"
                elif fatigue_status == "明显疲劳":
                    eye_focus_status = "视线模糊"  # 疲劳但未持续闭眼
                elif fatigue_status == "轻度疲劳":
                    eye_focus_status = "视线模糊"
                else:
                    # 第2优先级：眼睛睁开时，用虹膜偏移判断视线
                    iris_offset = self._calc_iris_offset(face_lm)
                    if iris_offset > 0.25:
                        eye_focus_status = "频繁偏离"
                    elif iris_offset > 0.12:
                        eye_focus_status = "偶尔偏离"
                    else:
                        eye_focus_status = "注视屏幕"
            else:
                # Face Mesh 没检测到人脸（可能完全闭眼 / 眼镜反光），给保守默认值
                fatigue_status = "轻度疲劳"
                eye_focus_status = "视线模糊"

            # 姿态稳定性（辅助指标）
            pose_stability = self._calculate_pose_stability(head_tilt, shoulder_level)

            # ========== 使用训练模型进行预测 ==========
            model_prediction = None
            if self.model_loaded:
                try:
                    features = self._extract_model_features(landmarks)
                    if features is not None:
                        model_result, focused_prob = predict_focus(features)
                        model_prediction = model_result
                        if model_result is not None:
                            model_score = int(focused_prob * 100)
                            rule_score = self._calculate_focus_score(head_pose_status, body_stability_status, fatigue_status, eye_focus_status, scene, pose_stability)
                            # 智能融合：当模型分与规则分差距过大时，以规则分为主
                            # 模型分持续<30说明模型未正确训练，应信任规则分
                            score_diff = abs(model_score - rule_score)
                            if rule_score > 90:
                                # 规则分很高时，大幅降低模型影响，确保最终分95+
                                focus_score = int(0.08 * model_score + 0.92 * rule_score)
                            elif model_score < 30 and rule_score > 70:
                                # 模型明显异常，几乎完全信任规则
                                focus_score = int(0.05 * model_score + 0.95 * rule_score)
                            elif score_diff > 40:
                                # 分歧较大，提高规则权重
                                focus_score = int(0.15 * model_score + 0.85 * rule_score)
                            else:
                                # 正常融合
                                focus_score = int(0.25 * model_score + 0.75 * rule_score)
                            print(f"📊 模型分={model_score} 规则分={rule_score} 最终={focus_score} 姿态={head_pose_status} 眼神={eye_focus_status} 疲劳={fatigue_status}")
                        else:
                            focus_score = self._calculate_focus_score(head_pose_status, body_stability_status, fatigue_status, eye_focus_status, scene, pose_stability)
                    else:
                        focus_score = self._calculate_focus_score(head_pose_status, body_stability_status, fatigue_status, eye_focus_status, scene, pose_stability)
                except Exception as e:
                    print(f"⚠️ 模型预测失败: {e}")
                    focus_score = self._calculate_focus_score(head_pose_status, body_stability_status, fatigue_status, eye_focus_status, scene, pose_stability)
            else:
                # 未训练模型，使用传统规则
                rule_score = self._calculate_focus_score(head_pose_status, body_stability_status, fatigue_status, eye_focus_status, scene, pose_stability)
                focus_score = rule_score
                print(f"📊 规则分={rule_score} (模型失效已忽略) 最终={focus_score} 姿态={head_pose_status} 眼神={eye_focus_status} 疲劳={fatigue_status}")

            # ========== 时序平滑 ==========
            focus_score = self._smooth_focus_score(focus_score)

            # ========== 保存有效分数（供fallback使用）==========
            self.last_valid_score = focus_score

            is_distracted = focus_score < 60
            if is_distracted:
                self.consecutive_low_focus += 1
                if self.consecutive_low_focus >= POSE_THRESHOLDS.get("consecutive_low_focus", 3):
                    self.total_distractions += 1
                    self.consecutive_low_focus = 0
                    # 自动发送语音提醒到学生端
                    try:
                        reminder_msg = self._generate_suggestion(focus_score, head_pose_status, fatigue_status, scene)
                        student_notifications[student_id] = {
                            "student_id": student_id,
                            "message": reminder_msg,
                            "timestamp": time.time(),
                            "type": "voice"
                        }
                    except Exception as e:
                        print(f"⚠️ 发送提醒失败: {e}")
            else:
                self.consecutive_low_focus = max(0, self.consecutive_low_focus - 1)

            suggestion = self._generate_suggestion(focus_score, head_pose_status, fatigue_status, scene)

            # 动作小分类检测
            raw_action = self._classify_action(landmarks, head_pose_status, eye_focus_status, fatigue_status, body_stability_status)
            action_label = self._smooth_action(raw_action)
            action_cn = {"focused": "专注", "sleep": "睡觉", "playphone": "玩手机", "distracted": "分心"}.get(action_label, action_label)
            if action_label != "focused":
                print(f"🏷️ 动作分类: {action_cn} ({action_label})")

            # 构建模型预测信息
            model_info = ""
            if model_prediction:
                model_info = f"模型预测: {model_prediction.get('label', '未知')} (专注概率: {model_prediction.get('prob_focused', 0.5):.0%})"

            return {
                "focus_score": focus_score,
                "action_label": action_label,
                "distraction_count": self.total_distractions,
                "head_pose": head_pose_status,
                "body_stability": body_stability_status,
                "fatigue_level": fatigue_status,
                "eye_focus": eye_focus_status,
                "suggestion": suggestion,
                "pose_detected": True,
                "annotated_image": annotated_b64,
                "model_prediction": model_info,
                "model_enabled": self.model_loaded,
                "pose_stability": round(pose_stability, 2)
            }
        except Exception as e:
            print(f"⚠️ MediaPipe分析异常: {e}")
            return self._fallback_analysis(scene, image_base64)

    def _draw_skeleton(self, frame, landmarks):
        """绘制人体骨架"""
        annotated = frame.copy()
        h, w, _ = annotated.shape
        SKELETON_CONNECTIONS = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
            (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
            (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
            (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EAR),
            (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
            (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
            (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
            (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_EAR),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_EAR),
        ]
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            sp, ep = landmarks[start_idx], landmarks[end_idx]
            cv2.line(annotated, (int(sp.x*w), int(sp.y*h)), (int(ep.x*w), int(ep.y*h)), (34, 211, 238), 2, cv2.LINE_AA)
        KEY_POINTS = [
            mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
            mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        for kp_idx in KEY_POINTS:
            kp = landmarks[kp_idx]
            cx, cy = int(kp.x*w), int(kp.y*h)
            cv2.circle(annotated, (cx, cy), 8, (99, 102, 241), -1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 4, (22, 211, 238), -1, cv2.LINE_AA)
        return annotated

    def _calc_head_tilt(self, n, ls, rs):
        """计算头部水平偏移：鼻子相对肩膀中心的水平偏差"""
        shoulder_center_x = (ls.x + rs.x) / 2
        return abs(n.x - shoulder_center_x) * 100

    def _calc_head_drop(self, n, ls, rs):
        sw = abs(ls.x - rs.x)
        return ((ls.y + rs.y) / 2 - n.y) / sw if sw > 0.01 else 0

    def _evaluate_head_pose(self, tilt, drop):
        if tilt > 20: return "明显偏移"
        elif tilt > 10: return "轻微偏移"
        elif drop < 0.15: return "明显低头"
        elif drop < 0.35: return "轻微低头"
        else: return "端正"

    def _calc_body_sway(self, ls, rs, lh, rh):
        return abs((ls.x + rs.x) / 2 - (lh.x + rh.x) / 2) * 100

    def _evaluate_body_stability(self, sl, sway):
        th = POSE_THRESHOLDS
        if sl > th["shoulder_level_major"] or sway > th["body_sway_major"]: return "明显晃动"
        elif sl > th["shoulder_level_minor"] or sway > th["body_sway_minor"]: return "轻微晃动"
        else: return "稳定"

    def _evaluate_eye_focus(self, n, ls, rs):
        """（保留兼容：旧方法，新版 analyze 不再调用）"""
        shoulder_center_x = (ls.x + rs.x) / 2
        offset = abs(n.x - shoulder_center_x) * 100
        if offset > 18: return "频繁偏离"
        elif offset > 9: return "偶尔偏离"
        else: return "注视屏幕"

    def _calc_ear(self, face_landmarks):
        """基于 Face Mesh 关键点计算 EAR（眼纵横比），闭眼时 EAR 接近 0"""
        if face_landmarks is None:
            return 0.3
        import math
        def _ear(p1, p2, p3, p4, p5, p6):
            v1 = math.hypot(p2.x - p6.x, p2.y - p6.y)
            v2 = math.hypot(p3.x - p5.x, p3.y - p5.y)
            h = math.hypot(p1.x - p4.x, p1.y - p4.y)
            if h < 1e-6: return 0.3
            return (v1 + v2) / (2.0 * h)
        left = _ear(
            face_landmarks[33], face_landmarks[160], face_landmarks[158],
            face_landmarks[133], face_landmarks[153], face_landmarks[144])
        right = _ear(
            face_landmarks[362], face_landmarks[385], face_landmarks[387],
            face_landmarks[263], face_landmarks[373], face_landmarks[380])
        return (left + right) / 2.0

    def _evaluate_fatigue(self, ear):
        """基于 EAR 判断疲劳等级：< 0.22 明显疲劳，< 0.30 轻度疲劳，否则清醒"""
        if ear is None or ear <= 0: return "未知"
        if ear < 0.22: return "明显疲劳"
        if ear < 0.30: return "轻度疲劳"
        return "精力充沛"

    def _calc_iris_offset(self, face_landmarks):
        """虹膜中心相对眼角水平中心的偏移比例（0 正对屏幕）。返回左右眼平均值 0-1"""
        if face_landmarks is None:
            return 0.5
        def _one(inner_idx, outer_idx, iris_idx):
            inner = face_landmarks[inner_idx]
            outer = face_landmarks[outer_idx]
            iris = face_landmarks[iris_idx]
            eye_w = abs(outer.x - inner.x)
            if eye_w < 1e-6: return 0.0
            return abs(iris.x - (inner.x + outer.x) / 2) / eye_w
        left = _one(33, 133, 468)
        right = _one(362, 263, 473)
        return (left + right) / 2.0

    # ========== 姿态稳定性评估 ==========
    def _calculate_pose_stability(self, head_tilt, shoulder_level):
        """计算姿态稳定性"""
        self.head_tilt_history.append(head_tilt)
        self.shoulder_history.append(shoulder_level)

        if len(self.head_tilt_history) > 10:
            self.head_tilt_history.pop(0)
        if len(self.shoulder_history) > 10:
            self.shoulder_history.pop(0)

        if len(self.head_tilt_history) < 3:
            return 1.0

        tilt_variance = np.var(self.head_tilt_history)
        shoulder_variance = np.var(self.shoulder_history)
        stability = max(0, 1 - (tilt_variance / 100 + shoulder_variance * 10))
        return stability

    # ========== 时序平滑 ==========
    def _smooth_focus_score(self, raw_score):
        """对专注度分数进行指数平滑，减少短期波动"""
        self.focus_score_history.append(raw_score)
        if len(self.focus_score_history) > 15:
            self.focus_score_history.pop(0)
        if len(self.focus_score_history) < 3:
            return raw_score
        # 使用更强的平滑权重（0.9），让历史分数影响更大
        weights = [0.9 ** (len(self.focus_score_history) - 1 - i) for i in range(len(self.focus_score_history))]
        weights = [w / sum(weights) for w in weights]
        smoothed = sum(s * w for s, w in zip(self.focus_score_history, weights))
        # 增加惯性：如果新分数与历史均值差距<10，更倾向于保持当前趋势
        if len(self.focus_score_history) >= 5:
            recent_avg = sum(self.focus_score_history[-5:]) / 5
            if abs(smoothed - recent_avg) < 8:
                smoothed = 0.7 * smoothed + 0.3 * recent_avg
        return round(smoothed)

    # ========== 动作小分类检测 ==========
    def _classify_action(self, landmarks, head_pose_status, eye_focus_status, fatigue_status, body_stability_status, ear_value=0.3):
        """
        检测具体动作类型（仅在 distracted 状态下分类）
        分类：focused / sleep / playphone / distracted
        基于姿态关键点规则判断：
        - sleep: 头部严重下垂 + 眼睛闭合(EAR低) + 身体不动
        - playphone: 双手腕在胸前/腹部区域 + 肘部弯曲 + 头部略微低头
        - distracted: 其他分心行为
        """
        if head_pose_status == "端正" and eye_focus_status == "注视屏幕" and fatigue_status == "精力充沛":
            return "focused"

        # 获取关键点
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        torso_mid_y = (shoulder_center_y + hip_center_y) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2

        # ===== 睡觉检测 =====
        # 头部严重下垂（鼻子y坐标远低于肩膀）+ 眼睛闭合 + 身体稳定
        head_drop_ratio = nose.y - shoulder_center_y
        is_eyes_closed = fatigue_status == "明显疲劳" or eye_focus_status == "闭眼/走神"
        is_body_still = body_stability_status == "稳定"

        if head_drop_ratio > 0.15 and is_eyes_closed and is_body_still:
            return "sleep"

        # 备选：头部略微下垂 + 眼睛闭合 + 长期不动也判定为睡觉
        if head_drop_ratio > 0.08 and is_eyes_closed and is_body_still and fatigue_status == "明显疲劳":
            return "sleep"

        # ===== 玩手机检测 =====
        # 手腕在躯干前方区域（允许在肩膀上方，因为举手机时手会抬高）
        # 范围：肩膀上方0.15到臀部下方0.05
        left_wrist_in_torso = (left_wrist.y > shoulder_center_y - 0.15 and
                               left_wrist.y < hip_center_y + 0.05)
        right_wrist_in_torso = (right_wrist.y > shoulder_center_y - 0.15 and
                                right_wrist.y < hip_center_y + 0.05)

        # 手腕靠近身体中心线（水平距离小于肩宽的80%，更宽松）
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        left_wrist_near_center = abs(left_wrist.x - shoulder_center_x) < shoulder_width * 0.8
        right_wrist_near_center = abs(right_wrist.x - shoulder_center_x) < shoulder_width * 0.8

        # 肘部弯曲检测：手臂收起时，手腕与肘部的y坐标差距不大（无论上下）
        # 或者手腕在肘部上方（举着手机）
        left_elbow_bent = abs(left_wrist.y - left_elbow.y) < 0.15 or left_wrist.y < left_elbow.y
        right_elbow_bent = abs(right_wrist.y - right_elbow.y) < 0.15 or right_wrist.y < right_elbow.y

        # 综合判断：至少一只手在胸前区域 + 肘部弯曲
        # 玩手机的典型姿势：手在胸前/脸前，手臂弯曲收起
        left_hand_valid = left_wrist_in_torso and left_wrist_near_center
        right_hand_valid = right_wrist_in_torso and right_wrist_near_center
        any_hand_in_torso = left_hand_valid or right_hand_valid
        any_elbow_bent = left_elbow_bent or right_elbow_bent

        # 调试输出（帮助诊断）
        debug_info = f"[DEBUG] hand_L={left_hand_valid} hand_R={right_hand_valid} elbow_L={left_elbow_bent} elbow_R={right_elbow_bent} lw_y={left_wrist.y:.3f} le_y={left_elbow.y:.3f}"
        if any_hand_in_torso and any_elbow_bent:
            print(f"🏷️ {debug_info} -> playphone")
            return "playphone"
        elif any_hand_in_torso or any_elbow_bent:
            print(f"🏷️ {debug_info} -> distracted (partial match)")

        # ===== 默认：普通分心 =====
        return "distracted"

    def _smooth_action(self, action):
        """对动作分类进行时序平滑，避免频繁跳变"""
        self.action_history.append(action)
        if len(self.action_history) > 8:
            self.action_history.pop(0)
        if len(self.action_history) < 3:
            return action
        # 取最近5帧的众数
        recent = self.action_history[-5:]
        from collections import Counter
        most_common = Counter(recent).most_common(1)[0]
        return most_common[0]

    # ========== 评分算法 ==========
    def _calculate_focus_score(self, hp, bs, fat, ef, scene, stability=1.0):
        s = 100
        if hp == "明显偏移": s -= 20
        elif hp == "轻微偏移": s -= 10
        elif hp == "明显低头": s -= 8
        elif hp == "轻微低头": s -= 4
        if bs == "明显晃动": s -= 12
        elif bs == "轻微晃动": s -= 5
        if fat == "明显疲劳": s -= 15
        elif fat == "轻度疲劳": s -= 8
        # 眼神状态：闭眼/走神最严重，视线模糊次之，偶尔偏离（正常眨眼）扣分很少
        if ef == "闭眼/走神": s -= 30
        elif ef == "视线模糊": s -= 18
        elif ef == "频繁偏离": s -= 12
        elif ef == "偶尔偏离": s -= 3
        if stability > 0.85: s += 4
        elif stability < 0.2: s -= 3
        if scene == "onetoone": s = min(s + 3, 100)
        elif scene == "online": s = max(s - 3, 10)
        return max(10, min(100, s))

    def _generate_suggestion(self, fs, hp, fat, scene):
        if fs >= 80: return random.choice(["✅ 专注度极佳，保持当前学习节奏！", "✅ 学习状态非常好，继续保持！", "✅ 你的专注力令人印象深刻！"])
        elif fs >= 60:
            if "偏移" in hp: return "💡 注意调整坐姿，保持头部端正有助于集中注意力"
            if "低头" in hp: return "💡 检测到轻微低头，适当调整坐姿，保持平视"
            if fat == "轻度疲劳": return "💡 检测到轻微疲劳，建议调整呼吸，做几次深呼吸"
            return "💡 状态良好，建议适时调整坐姿，预防疲劳"
        elif fs >= 40:
            if fat == "明显疲劳": return "⚠️ 检测到疲劳信号，建议闭眼休息30秒再继续"
            if "偏移" in hp: return "⚠️ 头部偏移较大，请调整坐姿，保持正视屏幕"
            if "低头" in hp: return "⚠️ 低头时间较长，建议抬抬头活动颈椎"
            return "⚠️ 专注度下降，建议起身活动2分钟，喝口水"
        else: return random.choice(["🚨 严重分心！建议暂停学习，休息5分钟后再继续", "🚨 注意力严重分散！请先放松一下，调整好状态再学习", "🚨 检测到多项异常指标，强烈建议休息调整"])

    def _fallback_analysis(self, scene, image_base64=None):
        """无人脸时：不返回虚假专注度，携带原始画面以便前端显示摄像头"""
        return {
            "focus_score": None,
            "distraction_count": self.total_distractions,
            "head_pose": "未检测到", "body_stability": "未检测到",
            "fatigue_level": "未知", "eye_focus": "未知",
            "suggestion": "未检测到人体，请确保摄像头能拍摄到上半身",
            "pose_detected": False,
            "annotated_image": image_base64,
            "model_prediction": "", "model_enabled": self.model_loaded
        }


analyzer = MediaPipePoseAnalyzer()

# ========== 学生实时视频帧缓存 ==========
student_video_cache = {}
student_notifications = {}  # 存储学生语音提醒
online_students = {}  # 存储在线学生信息

def update_student_video_cache(student_id: int, annotated_image: str, focus_score: int,
                                 action_label="unknown", head_pose="", eye_focus="", fatigue_level="",
                                 name=""):
    """更新学生视频帧缓存"""
    student_video_cache[student_id] = {
        "image": annotated_image,
        "timestamp": time.time(),
        "focus_score": focus_score,
        "action_label": action_label,
        "head_pose": head_pose,
        "eye_focus": eye_focus,
        "fatigue_level": fatigue_level,
        "name": name
    }

def get_student_video_cache(student_id: int):
    """获取学生视频帧缓存"""
    return student_video_cache.get(student_id)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 客户端已连接")
    current_student_id = None
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if "image" in msg:
                scene = msg.get("scene", "selfstudy")
                student_id = msg.get("student_id")
                current_student_id = student_id
                result = analyzer.analyze(msg["image"], scene, student_id)
                
                # 缓存学生视频帧（供教师实时查看）
                if student_id and result.get("annotated_image"):
                    # 查询学生姓名
                    student_name = "学生" + str(student_id)
                    try:
                        conn = sqlite3.connect("users.db")
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM users WHERE id=?", (student_id,))
                        row = cursor.fetchone()
                        if row and row[0]:
                            student_name = row[0]
                        conn.close()
                    except:
                        pass
                    
                    update_student_video_cache(
                        student_id,
                        result["annotated_image"],
                        result.get("focus_score", 0),
                        action_label=result.get("action_label", "unknown"),
                        head_pose=result.get("head_pose", ""),
                        eye_focus=result.get("eye_focus", ""),
                        fatigue_level=result.get("fatigue_level", ""),
                        name=student_name
                    )
                    result["student_id"] = student_id
                    # 更新在线学生列表
                    online_students[student_id] = {
                        "name": student_name,
                        "focus_score": result.get("focus_score", 0),
                        "status": result.get("action_label", "unknown"),
                        "last_seen": time.time()
                    }
                
                await websocket.send_json(result)
            elif msg.get("action") == "encouragement":
                print(f"💪 收到鼓励: {msg.get('message', '')}")
            elif msg.get("action") == "asr_result":
                print(f"🎙️ 语音识别: {msg.get('text', '')}")
    except Exception as e:
        print("❌ 连接断开：", e)
    finally:
        # 断开时从在线列表移除
        if current_student_id and current_student_id in online_students:
            del online_students[current_student_id]


# ========== HTML 页面 ==========
HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI学伴 · 专注度智能监护系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1040 25%, #0d2b45 50%, #0a1e3d 75%, #0f0c29 100%);
            min-height: 100vh;
            color: #e2e8f0;
            padding: 20px;
            overflow-x: hidden;
        }
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(236,72,153,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 40% 80%, rgba(34,211,238,0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 90% 80%, rgba(168,85,247,0.06) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
            position: relative;
            z-index: 1;
        }
        .header {
            background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(236,72,153,0.1) 50%, rgba(34,211,238,0.1) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 24px;
            padding: 20px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        .header-left h1 {
            font-size: 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(135deg, #c084fc, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header-left h1 .icon { font-size: 2rem; -webkit-text-fill-color: initial; }
        .scene-badge {
            background: linear-gradient(135deg, #6366f1, #ec4899);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            box-shadow: 0 2px 10px rgba(99,102,241,0.3);
        }
        .connection-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #22c55e; animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
            50% { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
        }
        .grid-main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card {
            backdrop-filter: blur(10px); border-radius: 20px; padding: 24px;
            transition: all 0.3s ease; position: relative; overflow: hidden;
        }
        .card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0;
            height: 3px; border-radius: 20px 20px 0 0;
        }
        .card:nth-child(1) { background: linear-gradient(145deg, rgba(34,211,238,0.08) 0%, rgba(99,102,241,0.06) 100%); border: 1px solid rgba(34,211,238,0.15); }
        .card:nth-child(1)::before { background: linear-gradient(90deg, #22d3ee, #6366f1); }
        .card:nth-child(2) { background: linear-gradient(145deg, rgba(168,85,247,0.08) 0%, rgba(236,72,153,0.06) 100%); border: 1px solid rgba(168,85,247,0.15); }
        .card:nth-child(2)::before { background: linear-gradient(90deg, #a855f7, #ec4899); }
        .card:nth-child(3) { background: linear-gradient(145deg, rgba(34,197,94,0.08) 0%, rgba(16,185,129,0.06) 100%); border: 1px solid rgba(34,197,94,0.15); }
        .card:nth-child(3)::before { background: linear-gradient(90deg, #22c55e, #10b981); }
        .card:nth-child(4) { background: linear-gradient(145deg, rgba(251,146,60,0.08) 0%, rgba(245,158,11,0.06) 100%); border: 1px solid rgba(251,146,60,0.15); }
        .card:nth-child(4)::before { background: linear-gradient(90deg, #fb923c, #f59e0b); }
        .card.full-width {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(236,72,153,0.04) 50%, rgba(34,211,238,0.04) 100%);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card.full-width::before { background: linear-gradient(90deg, #6366f1, #ec4899, #22d3ee); }
        .card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.15); }
        .card-title {
            font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;
            color: #c4b5fd; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
        }
        .focus-ring-container { display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .focus-ring {
            position: relative; width: 180px; height: 180px; border-radius: 50%;
            background: conic-gradient(#22d3ee 0%, #6366f1 var(--focus), #1e293b var(--focus));
            display: flex; align-items: center; justify-content: center; margin-bottom: 12px;
            box-shadow: 0 0 30px rgba(99,102,241,0.2), inset 0 0 20px rgba(0,0,0,0.3);
            transition: all 0.5s ease;
        }
        .focus-ring::before {
            content: ''; position: absolute; width: 140px; height: 140px; border-radius: 50%;
            background: radial-gradient(circle, #131b2e 0%, #0a0e27 100%);
        }
        .focus-value {
            position: relative; z-index: 1; font-size: 2.5rem; font-weight: 700;
            background: linear-gradient(135deg, #22d3ee, #6366f1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px; }
        .metric-item {
            text-align: center; padding: 12px; background: rgba(255,255,255,0.04);
            border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); transition: all 0.3s;
        }
        .metric-item:hover { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.12); }
        .metric-value { font-size: 1.5rem; font-weight: 700; }
        .metric-label { font-size: 0.7rem; color: #94a3b8; margin-top: 4px; }
        .color-good { color: #34d399; } .color-warning { color: #fbbf24; } .color-danger { color: #f87171; }
        .status-bar {
            display: flex; align-items: center; gap: 8px; padding: 10px 16px;
            border-radius: 12px; margin-bottom: 12px; font-size: 0.85rem; font-weight: 500; transition: all 0.3s;
        }
        .status-bar.focused { background: linear-gradient(135deg, rgba(34,211,238,0.15), rgba(34,197,94,0.15)); color: #22d3ee; border: 1px solid rgba(34,211,238,0.2); }
        .status-bar.distracted { background: linear-gradient(135deg, rgba(251,191,36,0.15), rgba(245,158,11,0.15)); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
        .status-bar.fatigued { background: linear-gradient(135deg, rgba(248,113,113,0.15), rgba(239,68,68,0.15)); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }
        .timeline { display: flex; gap: 4px; height: 40px; align-items: flex-end; }
        .timeline-bar { flex: 1; border-radius: 4px 4px 0 0; transition: height 0.3s, background 0.3s; min-height: 3px; }
        .suggestion-box {
            background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.08));
            border-left: 3px solid #a855f7; border-radius: 0 12px 12px 0; padding: 14px 18px; margin-top: 12px;
        }
        .btn {
            background: linear-gradient(135deg, #6366f1, #a855f7); border: none; color: white;
            padding: 10px 22px; border-radius: 25px; font-weight: 500; cursor: pointer;
            transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px;
            font-size: 0.85rem; box-shadow: 0 2px 10px rgba(99,102,241,0.2);
        }
        .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(99,102,241,0.4); }
        .btn-outline { background: transparent; border: 1px solid rgba(255,255,255,0.2); box-shadow: none; }
        .btn-outline:hover { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.4); box-shadow: none; }
        .btn-voice { background: linear-gradient(135deg, #ec4899, #f43f5e); box-shadow: 0 2px 10px rgba(236,72,153,0.3); }
        .btn-voice:hover { box-shadow: 0 4px 20px rgba(236,72,153,0.5); }
        .btn-voice.active { animation: voicePulse 1.5s infinite; }
        @keyframes voicePulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(236,72,153,0.4); } 50% { box-shadow: 0 0 0 12px rgba(236,72,153,0); } }
        .btn-small { padding: 6px 14px; font-size: 0.75rem; }
        .btn-success { background: linear-gradient(135deg, #22c55e, #16a34a); box-shadow: 0 2px 10px rgba(34,197,94,0.3); }
        .btn-warning { background: linear-gradient(135deg, #f59e0b, #d97706); box-shadow: 0 2px 10px rgba(245,158,11,0.3); }
        .camera-preview {
            position: fixed; bottom: 24px; right: 24px; width: 200px; border-radius: 16px;
            overflow: hidden; border: 2px solid rgba(99,102,241,0.4); z-index: 1000;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .camera-preview canvas { width: 100%; display: block; border-radius: 14px; }
        .cam-label {
            position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.6);
            color: #22d3ee; font-size: 10px; padding: 3px 10px; border-radius: 20px; backdrop-filter: blur(4px);
        }
        .pomodoro { text-align: center; padding: 16px; }
        .pomodoro-time {
            font-size: 2rem; font-weight: 700; font-variant-numeric: tabular-nums; font-family: monospace;
            background: linear-gradient(135deg, #34d399, #22d3ee);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .pomodoro-settings { margin-top: 12px; padding: 12px; background: rgba(255,255,255,0.04); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); }
        .pomodoro-settings label { display: block; font-size: 0.75rem; color: #94a3b8; margin-bottom: 8px; }
        .time-presets { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 12px; }
        .time-input-group { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 8px; }
        .time-input {
            width: 60px; padding: 8px 12px; background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; color: #e2e8f0;
            font-size: 1rem; text-align: center; font-family: monospace;
        }
        .time-input:focus { outline: none; border-color: #6366f1; box-shadow: 0 0 0 2px rgba(99,102,241,0.2); }
        .time-unit { font-size: 0.8rem; color: #94a3b8; }
        .voice-status {
            display: flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 12px;
            margin-top: 12px; font-size: 0.8rem; background: rgba(236,72,153,0.08); border: 1px solid rgba(236,72,153,0.15);
        }
        .voice-wave { display: flex; align-items: center; gap: 2px; height: 16px; }
        .voice-wave span { width: 3px; background: #ec4899; border-radius: 2px; animation: wave 0.8s ease-in-out infinite; }
        .voice-wave span:nth-child(1) { height: 6px; animation-delay: 0s; }
        .voice-wave span:nth-child(2) { height: 12px; animation-delay: 0.1s; }
        .voice-wave span:nth-child(3) { height: 8px; animation-delay: 0.2s; }
        .voice-wave span:nth-child(4) { height: 14px; animation-delay: 0.3s; }
        .voice-wave span:nth-child(5) { height: 6px; animation-delay: 0.4s; }
        @keyframes wave { 0%, 100% { transform: scaleY(0.5); } 50% { transform: scaleY(1.2); } }
        .speech-toast {
            position: fixed; top: 20px; left: 50%; transform: translateX(-50%) translateY(-100px);
            background: linear-gradient(135deg, rgba(99,102,241,0.9), rgba(168,85,247,0.9));
            color: white; padding: 12px 24px; border-radius: 30px; font-size: 0.9rem; font-weight: 500;
            z-index: 2000; transition: transform 0.4s ease; box-shadow: 0 8px 30px rgba(99,102,241,0.4); backdrop-filter: blur(10px);
        }
        .speech-toast.show { transform: translateX(-50%) translateY(0); }

        /* ===== 登录/注册页面 ===== */
        .auth-container {
            max-width: 440px; margin: 60px auto; padding: 40px;
            background: linear-gradient(145deg, rgba(99,102,241,0.1) 0%, rgba(236,72,153,0.08) 100%);
            border: 1px solid rgba(255,255,255,0.1); border-radius: 24px;
            backdrop-filter: blur(20px); box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .auth-title {
            text-align: center; font-size: 1.8rem; margin-bottom: 8px;
            background: linear-gradient(135deg, #c084fc, #60a5fa, #34d399);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .auth-subtitle { text-align: center; margin-bottom: 24px; color: #94a3b8; font-size: 0.9rem; }
        .role-tabs {
            display: flex; gap: 0; margin-bottom: 24px; border-radius: 12px; overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .role-tab {
            flex: 1; padding: 12px; text-align: center; cursor: pointer; font-size: 0.9rem;
            background: rgba(255,255,255,0.03); color: #94a3b8; transition: all 0.3s; border: none;
        }
        .role-tab.active {
            background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(168,85,247,0.2));
            color: #e2e8f0; font-weight: 600;
        }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; margin-bottom: 6px; font-size: 0.85rem; color: #94a3b8; }
        .form-input {
            width: 100%; padding: 12px 16px; background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; color: #e2e8f0;
            font-size: 1rem; transition: all 0.3s;
        }
        .form-input:focus { outline: none; border-color: #6366f1; background: rgba(255,255,255,0.08); box-shadow: 0 0 0 3px rgba(99,102,241,0.2); }
        .form-input::placeholder { color: #64748b; }
        .auth-btn { width: 100%; padding: 14px; font-size: 1rem; margin-top: 8px; }
        .auth-switch { text-align: center; margin-top: 16px; font-size: 0.85rem; color: #94a3b8; }
        .auth-switch a { color: #6366f1; cursor: pointer; text-decoration: none; }
        .auth-switch a:hover { text-decoration: underline; }
        .error-msg { color: #f87171; font-size: 0.85rem; margin-top: 8px; text-align: center; min-height: 20px; }
        .form-row { display: flex; gap: 12px; }
        .form-row .form-group { flex: 1; }

        /* ===== 学生列表 ===== */
        .student-list { display: grid; gap: 12px; }
        .student-item {
            display: flex; align-items: center; justify-content: space-between;
            padding: 16px 20px; background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
            transition: all 0.3s; cursor: pointer;
        }
        .student-item:hover { background: rgba(255,255,255,0.08); border-color: rgba(99,102,241,0.3); transform: translateX(4px); }
        .student-info { display: flex; align-items: center; gap: 16px; }
        .student-avatar {
            width: 48px; height: 48px; border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem; flex-shrink: 0;
        }
        .student-details h3 { font-size: 1.1rem; margin-bottom: 4px; }
        .student-details p { font-size: 0.8rem; color: #94a3b8; }
        .student-status {
            display: flex; align-items: center; gap: 12px;
        }
        .status-badge {
            padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 500;
        }
        .status-badge.online { background: rgba(34,197,94,0.15); color: #34d399; border: 1px solid rgba(34,197,94,0.2); }
        .status-badge.offline { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.2); }
        .view-btn {
            padding: 8px 16px; border-radius: 20px; font-size: 0.8rem; cursor: pointer;
            background: linear-gradient(135deg, #6366f1, #a855f7); border: none; color: white;
            transition: all 0.2s;
        }
        .view-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(99,102,241,0.4); }

        /* ===== 教师查看学生详情 ===== */
        .detail-header {
            display: flex; align-items: center; gap: 20px; margin-bottom: 20px;
        }
        .detail-avatar {
            width: 64px; height: 64px; border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            display: flex; align-items: center; justify-content: center; font-size: 2rem;
        }
        .detail-info h2 { font-size: 1.4rem; margin-bottom: 4px; }
        .detail-info p { color: #94a3b8; font-size: 0.85rem; }
        .record-list { max-height: 400px; overflow-y: auto; }
        .record-item {
            padding: 12px 16px; background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05); border-radius: 12px;
            margin-bottom: 8px; font-size: 0.85rem;
        }
        .record-item .record-time { color: #64748b; font-size: 0.75rem; margin-bottom: 6px; }
        .record-item .record-score { font-size: 1.2rem; font-weight: 700; }
        .record-metrics { display: flex; gap: 16px; margin-top: 6px; flex-wrap: wrap; }
        .record-metric { font-size: 0.8rem; color: #94a3b8; }
        .back-btn {
            display: inline-flex; align-items: center; gap: 6px; color: #94a3b8;
            cursor: pointer; margin-bottom: 16px; transition: color 0.2s; font-size: 0.9rem;
        }
        .back-btn:hover { color: #e2e8f0; }

        .page { display: none; }
        .page.active { display: block; }

        .empty-state { text-align: center; color: #64748b; padding: 60px 20px; }
        .empty-state .empty-icon { font-size: 3rem; margin-bottom: 16px; }

        /* ===== 模型状态指示器 ===== */
        .model-status {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        }
        .model-status.active { background: rgba(34,197,94,0.15); color: #34d399; border-color: rgba(34,197,94,0.3); }
        .model-status.inactive { background: rgba(100,116,139,0.15); color: #94a3b8; border-color: rgba(100,116,139,0.3); }
        .model-dot { width: 6px; height: 6px; border-radius: 50%; }
        .model-dot.active { background: #34d399; animation: pulse 2s infinite; }
        .model-dot.inactive { background: #94a3b8; }

        @media (max-width: 768px) {
            .grid-main { grid-template-columns: 1fr; }
            .camera-preview { width: 140px; bottom: 12px; right: 12px; }
            .auth-container { margin: 30px 16px; padding: 30px 20px; }
        }
    </style>
</head>
<body>
    <div class="speech-toast" id="speechToast">🔊 正在播报鼓励...</div>

    <!-- ===== 登录页面 ===== -->
    <div id="loginPage" class="page active">
        <div class="auth-container">
            <h1 class="auth-title">🎓 AI学伴</h1>
            <p class="auth-subtitle">专注度智能监护系统</p>

            <div class="role-tabs">
                <button class="role-tab active" onclick="switchLoginRole('student')" id="tabStudent">👨‍🎓 学生登录</button>
                <button class="role-tab" onclick="switchLoginRole('teacher')" id="tabTeacher">👨‍🏫 教师登录</button>
            </div>

            <form id="loginForm">
                <div class="form-group">
                    <label>用户名</label>
                    <input type="text" class="form-input" id="loginUsername" placeholder="请输入用户名" required>
                </div>
                <div class="form-group">
                    <label>密码</label>
                    <input type="password" class="form-input" id="loginPassword" placeholder="请输入密码" required>
                </div>
                <div class="error-msg" id="loginError"></div>
                <button type="submit" class="btn auth-btn">登录</button>
            </form>
            <div class="auth-switch" id="studentRegisterSwitch">
                还没有账号？<a onclick="showRegister()">学生注册</a>
            </div>
            <div class="auth-switch" id="teacherRegisterSwitch" style="display: none;">
                还没有账号？<a onclick="showTeacherRegister()">教师注册</a>
            </div>
        </div>
    </div>

    <!-- ===== 学生注册页面 ===== -->
    <div id="registerPage" class="page">
        <div class="auth-container">
            <h1 class="auth-title">🎓 AI学伴</h1>
            <p class="auth-subtitle">👨‍🎓 学生注册</p>
            <form id="registerForm">
                <div class="form-group">
                    <label>用户名（登录用）</label>
                    <input type="text" class="form-input" id="regUsername" placeholder="至少3位字符" required>
                </div>
                <div class="form-group">
                    <label>真实姓名</label>
                    <input type="text" class="form-input" id="regName" placeholder="请输入真实姓名" required>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>学号（选填）</label>
                        <input type="text" class="form-input" id="regStudentNo" placeholder="学号">
                    </div>
                    <div class="form-group">
                        <label>班级（选填）</label>
                        <input type="text" class="form-input" id="regClass" placeholder="班级">
                    </div>
                </div>
                <div class="form-group">
                    <label>密码</label>
                    <input type="password" class="form-input" id="regPassword" placeholder="至少6位字符" required>
                </div>
                <div class="form-group">
                    <label>确认密码</label>
                    <input type="password" class="form-input" id="regPassword2" placeholder="再次输入密码" required>
                </div>
                <div class="error-msg" id="registerError"></div>
                <button type="submit" class="btn auth-btn">注册</button>
            </form>
            <div class="auth-switch">
                已有账号？<a onclick="showLogin()">返回登录</a>
            </div>
        </div>
    </div>

    <!-- ===== 教师注册页面 ===== -->
    <div id="teacherRegisterPage" class="page">
        <div class="auth-container">
            <h1 class="auth-title">🎓 AI学伴</h1>
            <p class="auth-subtitle">👨‍🏫 教师注册</p>
            <form id="teacherRegisterForm">
                <div class="form-group">
                    <label>用户名</label>
                    <input type="text" class="form-input" id="teacherRegUsername" placeholder="至少3位字符" required>
                </div>
                <div class="form-group">
                    <label>密码</label>
                    <input type="password" class="form-input" id="teacherRegPassword" placeholder="至少6位字符" required>
                </div>
                <div class="form-group">
                    <label>确认密码</label>
                    <input type="password" class="form-input" id="teacherRegPassword2" placeholder="再次输入密码" required>
                </div>
                <div class="error-msg" id="teacherRegisterError"></div>
                <button type="submit" class="btn auth-btn">注册</button>
            </form>
            <div class="auth-switch">
                已有账号？<a onclick="showLogin()">返回登录</a>
            </div>
        </div>
    </div>

    <!-- ===== 教师端：学生列表 ===== -->
    <div id="teacherStudentListPage" class="page">
        <div class="container">
            <div class="header">
                <div class="header-left">
                    <h1><span class="icon">🎓</span> AI学伴 · 教师端</h1>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <span id="teacherName" style="color: #94a3b8; font-size: 0.9rem;"></span>
                    <button class="btn btn-outline btn-small" onclick="logout()">退出登录</button>
                </div>
            </div>
            <div class="card full-width">
                <div class="card-title">👥 已注册学生列表</div>
                <div id="teacherStudentList" class="student-list">
                    <div class="empty-state">
                        <div class="empty-icon">📋</div>
                        <p>暂无注册学生</p>
                        <p style="font-size: 0.8rem; margin-top: 8px;">学生需要先注册账号后才会出现在此列表中</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- ===== 教师端：查看学生行为详情 ===== -->
    <div id="teacherStudentDetailPage" class="page">
        <div class="container">
            <div class="header">
                <div class="header-left">
                    <h1><span class="icon">🎓</span> AI学伴 · 学生行为</h1>
                </div>
                <button class="btn btn-outline btn-small" onclick="backToTeacherList()">← 返回列表</button>
            </div>
            <div class="card full-width">
                <div class="detail-header">
                    <div class="detail-avatar" id="detailAvatar">👤</div>
                    <div class="detail-info">
                        <h2 id="detailName">--</h2>
                        <p id="detailMeta">--</p>
                    </div>
                </div>
            </div>
            <!-- 实时视频监控区域 -->
            <div class="card full-width">
                <div class="card-title">📹 实时监控画面</div>
                <div id="liveVideoContainer" style="text-align: center;">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 16px;">
                        <span id="studentOnlineStatus" class="status-badge offline">⚫ 离线</span>
                        <span id="liveFocusScore" style="font-size: 1.2rem; font-weight: 700; color: #64748b;">专注度: --</span>
                        <button class="btn btn-small" id="refreshLiveBtn" onclick="refreshStudentLive()">🔄 刷新画面</button>
                        <label style="display: flex; align-items: center; gap: 6px; font-size: 0.85rem; color: #94a3b8;">
                            <input type="checkbox" id="autoRefreshLive" checked onchange="toggleAutoRefresh()">
                            自动刷新
                        </label>
                    </div>
                    <div style="position: relative; display: inline-block; border-radius: 16px; overflow: hidden; border: 2px solid rgba(99,102,241,0.3); background: #0a0e27;">
                        <img id="studentLiveImage" src="" alt="等待画面..." style="width: 480px; height: 360px; object-fit: contain; display: block;">
                        <div style="position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.6); color: #22d3ee; font-size: 10px; padding: 3px 10px; border-radius: 20px;">
                            📹 实时画面
                        </div>
                    </div>
                </div>
            </div>
            <div class="card full-width">
                <div class="card-title">📊 行为记录</div>
                <div id="detailRecords" class="record-list">
                    <div class="empty-state">
                        <div class="empty-icon">📭</div>
                        <p>暂无行为记录</p>
                        <p style="font-size: 0.8rem; margin-top: 8px;">该学生尚未开始上课监控</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- ===== 学生端：监控页面 ===== -->
    <div id="studentMonitorPage" class="page">
        <div class="container">
            <div class="header">
                <div class="header-left">
                    <h1><span class="icon">🎓</span> AI学伴 · 智能专注监护</h1>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <span class="scene-badge" id="sceneLabel">📚 自习模式</span>
                    <span style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem;">
                        <span class="connection-dot" id="connectionDot"></span>
                        <span id="connectionText">已连接</span>
                    </span>
                    <span id="studentGreeting" style="color: #94a3b8; font-size: 0.85rem;"></span>
                    <button class="btn btn-outline btn-small" onclick="logout()">退出</button>
                </div>
            </div>

            <!-- 场景切换 -->
            <div class="card full-width">
                <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                    <span style="font-size: 0.85rem; color: #94a3b8;">当前场景：</span>
                    <button class="btn btn-outline" onclick="switchScene('onetoone')" id="btnOneToOne">👨‍🏫 1对1课堂</button>
                    <div style="margin-left: auto; display: flex; gap: 8px;">
                        <button class="btn btn-voice" id="voiceBtn" onclick="toggleVoiceBroadcast()">🔊 语音播报</button>
                        <button class="btn btn-voice" id="asrBtn" onclick="toggleASR()">🎙️ 语音识别</button>
                    </div>
                </div>
            </div>

            <!-- 主内容区 -->
            <div class="grid-main">
                <div class="card">
                    <div class="card-title">🎯 实时专注度</div>
                    <div class="focus-ring-container">
                        <div class="focus-ring" id="focusRing">
                            <span class="focus-value" id="focusValue">--%</span>
                        </div>
                    </div>
                    <div class="status-bar focused" id="statusBar">
                        <span>🟢</span> <span id="statusText">等待分析...</span>
                    </div>
                    <div id="actionLabel" style="font-size: 0.8rem; color: #fbbf24; text-align: center; margin-top: 6px; font-weight: 600; min-height: 20px;"></div>
                    <div id="modelPrediction" style="font-size: 0.75rem; color: #94a3b8; text-align: center; margin-top: 4px;"></div>
                    <div class="metrics-row">
                        <div class="metric-item">
                            <div class="metric-value color-good" id="focusScore">--</div>
                            <div class="metric-label">专注评分</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value color-warning" id="distractCount">0</div>
                            <div class="metric-label">分心次数</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value" id="studyDuration" style="color: #22d3ee;">00:00</div>
                            <div class="metric-label">学习时长</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">📊 学习状态分析</div>
                    <div style="margin-bottom: 16px;">
                        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 8px;">近10分钟专注趋势</div>
                        <div class="timeline" id="trendTimeline"></div>
                    </div>
                    <div id="stateDetails" style="font-size: 0.85rem;">
                        <div style="margin-bottom: 8px;"><span style="color: #94a3b8;">头部姿态：</span><span id="headPose">--</span></div>
                        <div style="margin-bottom: 8px;"><span style="color: #94a3b8;">身体稳定性：</span><span id="bodyStability">--</span></div>
                        <div style="margin-bottom: 8px;"><span style="color: #94a3b8;">疲劳度：</span><span id="fatigueLevel">--</span></div>
                        <div style="margin-bottom: 8px;"><span style="color: #94a3b8;">眼神专注：</span><span id="eyeFocus">--</span></div>
                    </div>
                    <div class="suggestion-box" id="suggestionBox">
                        <span style="color: #94a3b8; font-size: 0.75rem;">💡 AI建议</span><br>
                        <span id="aiSuggestion" style="font-size: 0.85rem;">系统初始化中...</span>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">⏱️ 番茄钟 · 专注计时</div>
                    <div class="pomodoro">
                        <div class="pomodoro-time" id="pomodoroTimer">25:00</div>
                        <div style="margin-top: 8px; font-size: 0.8rem; color: #94a3b8;" id="pomodoroPhase">准备开始</div>
                        <div style="margin-top: 12px; display: flex; gap: 8px; justify-content: center;">
                            <button class="btn" onclick="togglePomodoro()" id="pomodoroBtn">▶ 开始</button>
                            <button class="btn btn-outline" onclick="resetPomodoro()">↺ 重置</button>
                        </div>
                    </div>
                    <div class="pomodoro-settings">
                        <label>⏰ 设置专注时长</label>
                        <div class="time-presets">
                            <button class="btn btn-small btn-outline" onclick="setPomodoroTime(15)">15分钟</button>
                            <button class="btn btn-small" onclick="setPomodoroTime(25)">25分钟</button>
                            <button class="btn btn-small btn-outline" onclick="setPomodoroTime(45)">45分钟</button>
                            <button class="btn btn-small btn-outline" onclick="setPomodoroTime(60)">60分钟</button>
                        </div>
                        <div class="time-input-group">
                            <span class="time-unit">自定义:</span>
                            <input type="number" class="time-input" id="customMinutes" min="1" max="180" value="25" onchange="updateCustomTime()">
                            <span class="time-unit">分钟</span>
                            <button class="btn btn-small" onclick="applyCustomTime()">应用</button>
                        </div>
                    </div>
                    <div class="voice-status" id="voiceStatus" style="display: none;">
                        <div class="voice-wave"><span></span><span></span><span></span><span></span><span></span></div>
                        <span id="asrText" style="color: #f9a8d4;">正在聆听...</span>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">🔔 智能提醒与干预</div>
                    <div id="reminderHistory" style="max-height: 180px; overflow-y: auto; font-size: 0.8rem;">
                        <div style="color: #64748b; padding: 8px 0;">暂无提醒记录</div>
                    </div>
                    <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                        <button class="btn" onclick="sendEncouragement()">💪 发送鼓励</button>
                        <button class="btn btn-outline" onclick="exportReport()">📄 导出报告</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="camera-preview" id="cameraPreview">
            <div class="cam-label">📹 MediaPipe 骨架</div>
            <canvas id="poseCanvas" width="640" height="480"></canvas>
            <video id="video" autoplay muted playsinline style="display:none;"></video>
        </div>
    </div>


    <script>
        // ========== 全局状态 ==========
        let currentToken = localStorage.getItem('token');
        let currentUser = JSON.parse(localStorage.getItem('user') || 'null');
        let loginRole = 'student';
        let modelEnabled = false;

        // ========== 页面切换 ==========
        function showPage(pageId) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');
        }

        function showLogin() { showPage('loginPage'); }
        function showRegister() { showPage('registerPage'); }
        function showTeacherRegister() { showPage('teacherRegisterPage'); }










        // ========== 登录角色切换 ==========
        function switchLoginRole(role) {
            loginRole = role;
            document.getElementById('tabStudent').className = role === 'student' ? 'role-tab active' : 'role-tab';
            document.getElementById('tabTeacher').className = role === 'teacher' ? 'role-tab active' : 'role-tab';
            document.getElementById('studentRegisterSwitch').style.display = role === 'student' ? 'block' : 'none';
            document.getElementById('teacherRegisterSwitch').style.display = role === 'teacher' ? 'block' : 'none';
            document.getElementById('loginError').textContent = '';
        }

        // ========== API请求 ==========
        async function apiRequest(url, options = {}) {
            const headers = { 'Content-Type': 'application/json', ...options.headers };
            if (currentToken) headers['Authorization'] = `Bearer ${currentToken}`;
            const response = await fetch(url, { ...options, headers });
            if (response.status === 401) { logout(); return null; }
            return response;
        }

        // ========== 登录 ==========
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            const endpoint = loginRole === 'student' ? '/api/student/login' : '/api/teacher/login';
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await response.json();
                if (response.ok) {
                    currentToken = data.access_token;
                    currentUser = data;
                    localStorage.setItem('token', currentToken);
                    localStorage.setItem('user', JSON.stringify(data));

                    if (data.role === 'student') {
                        showStudentMonitor();
                    } else {
                        showTeacherStudentList();
                    }
                } else {
                    document.getElementById('loginError').textContent = data.detail || '登录失败';
                }
            } catch (error) {
                document.getElementById('loginError').textContent = '网络错误';
            }
        });

        // ========== 学生注册 ==========
        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('regUsername').value;
            const name = document.getElementById('regName').value;
            const student_no = document.getElementById('regStudentNo').value;
            const class_name = document.getElementById('regClass').value;
            const password = document.getElementById('regPassword').value;
            const password2 = document.getElementById('regPassword2').value;

            if (password !== password2) {
                document.getElementById('registerError').textContent = '两次密码不一致';
                return;
            }

            try {
                const response = await fetch('/api/student/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password, name, student_no, class_name })
                });
                const data = await response.json();
                if (response.ok) {
                    alert('注册成功，请登录');
                    showLogin();
                } else {
                    document.getElementById('registerError').textContent = data.detail || '注册失败';
                }
            } catch (error) {
                document.getElementById('registerError').textContent = '网络错误';
            }
        });

        // ========== 教师注册 ==========
        document.getElementById('teacherRegisterForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('teacherRegUsername').value;
            const password = document.getElementById('teacherRegPassword').value;
            const password2 = document.getElementById('teacherRegPassword2').value;

            if (password !== password2) {
                document.getElementById('teacherRegisterError').textContent = '两次密码不一致';
                return;
            }

            try {
                const response = await fetch('/api/teacher/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await response.json();
                if (response.ok) {
                    alert('注册成功，请登录');
                    showLogin();
                } else {
                    document.getElementById('teacherRegisterError').textContent = data.detail || '注册失败';
                }
            } catch (error) {
                document.getElementById('teacherRegisterError').textContent = '网络错误';
            }
        });

        function logout() {
            currentToken = null;
            currentUser = null;
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            stopCamera();
            showLogin();
        }

        // ========== 教师端：学生列表 ==========
        async function showTeacherStudentList() {
            showPage('teacherStudentListPage');
            document.getElementById('teacherName').textContent = `👨‍🏫 ${currentUser.username}`;
            await loadTeacherStudents();
        }

        async function loadTeacherStudents() {
            const response = await apiRequest('/api/teacher/students');
            if (!response) return;
            const data = await response.json();
            const container = document.getElementById('teacherStudentList');

            if (data.students.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">📋</div>
                        <p>暂无注册学生</p>
                        <p style="font-size: 0.8rem; margin-top: 8px;">学生需要先注册账号后才会出现在此列表中</p>
                    </div>`;
                return;
            }

            container.innerHTML = data.students.map(s => {
                const latest = s.latest_record;
                const scoreText = latest ? latest.focus_score + '%' : '暂无数据';
                const scoreColor = latest ? (latest.focus_score >= 70 ? '#34d399' : latest.focus_score >= 40 ? '#fbbf24' : '#f87171') : '#64748b';
                const timeText = latest ? new Date(latest.created_at).toLocaleString('zh-CN') : '';
                const onlineStatus = s.online ? 
                    '<span class="status-badge online">🟢 在线</span>' : 
                    '<span class="status-badge offline">⚫ 离线</span>';
                const liveScore = s.online && s.live_focus_score !== null ? 
                    `<span style="font-size: 0.8rem; color: ${s.live_focus_score >= 70 ? '#34d399' : s.live_focus_score >= 40 ? '#fbbf24' : '#f87171'};">实时: ${s.live_focus_score}%</span>` : '';
                return `
                <div class="student-item">
                    <div class="student-info">
                        <div class="student-avatar">${s.name.charAt(0)}</div>
                        <div class="student-details">
                            <h3>${s.name}</h3>
                            <p>${s.student_no ? '学号: ' + s.student_no : ''} ${s.class_name ? '班级: ' + s.class_name : ''}</p>
                        </div>
                    </div>
                    <div class="student-status">
                        <div style="text-align: right;">
                            ${onlineStatus} ${liveScore}
                            <div style="font-size: 1.2rem; font-weight: 700; color: ${scoreColor}; margin-top: 4px;">${scoreText}</div>
                            ${timeText ? '<div style="font-size: 0.7rem; color: #64748b;">' + timeText + '</div>' : ''}
                        </div>
                        <button class="view-btn" onclick="viewStudentDetail(${s.id})">查看详情</button>
                    </div>
                </div>`;
            }).join('');
        }

        // ========== 教师端：查看学生行为详情 ==========
        let currentViewingStudentId = null;
        let liveRefreshInterval = null;

        async function viewStudentDetail(studentId) {
            showPage('teacherStudentDetailPage');
            currentViewingStudentId = studentId;

            const response = await apiRequest(`/api/teacher/students/${studentId}/records`);
            if (!response) return;
            const data = await response.json();

            const student = data.student;
            document.getElementById('detailAvatar').textContent = student.name.charAt(0);
            document.getElementById('detailName').textContent = student.name;
            document.getElementById('detailMeta').textContent =
                `${student.student_no ? '学号: ' + student.student_no : ''} ${student.class_name ? '班级: ' + student.class_name : ''} | 注册时间: ${new Date(student.created_at).toLocaleString('zh-CN')}`;

            refreshStudentLive();
            if (document.getElementById('autoRefreshLive').checked) {
                startAutoRefresh();
            }

            const container = document.getElementById('detailRecords');
            if (data.records.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">📭</div>
                        <p>暂无行为记录</p>
                        <p style="font-size: 0.8rem; margin-top: 8px;">该学生尚未开始上课监控</p>
                    </div>`;
                return;
            }

            container.innerHTML = data.records.map(r => {
                const score = r.focus_score || 0;
                const scoreColor = score >= 70 ? '#34d399' : score >= 40 ? '#fbbf24' : '#f87171';
                return `
                <div class="record-item">
                    <div class="record-time">📅 ${new Date(r.created_at).toLocaleString('zh-CN')}</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div class="record-score" style="color: ${scoreColor};">专注度: ${score}%</div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">分心次数: ${r.distraction_count || 0}</div>
                    </div>
                    <div class="record-metrics">
                        <span class="record-metric">头部: ${r.head_pose || '--'}</span>
                        <span class="record-metric">身体: ${r.body_stability || '--'}</span>
                        <span class="record-metric">疲劳: ${r.fatigue_level || '--'}</span>
                        <span class="record-metric">眼神: ${r.eye_focus || '--'}</span>
                    </div>
                    ${r.suggestion ? '<div style="margin-top: 6px; font-size: 0.8rem; color: #c4b5fd;">' + r.suggestion + '</div>' : ''}
                </div>`;
            }).join('');
        }

        async function refreshStudentLive() {
            if (!currentViewingStudentId) return;
            try {
                const response = await apiRequest(`/api/teacher/students/${currentViewingStudentId}/live`);
                if (!response) return;
                const data = await response.json();

                const statusEl = document.getElementById('studentOnlineStatus');
                const scoreEl = document.getElementById('liveFocusScore');
                const imgEl = document.getElementById('studentLiveImage');

                if (data.online) {
                    statusEl.className = 'status-badge online';
                    statusEl.innerHTML = '🟢 在线';
                    if (data.image) {
                        imgEl.src = 'data:image/jpeg;base64,' + data.image;
                    }
                    const score = data.focus_score || 0;
                    const scoreColor = score >= 70 ? '#34d399' : score >= 40 ? '#fbbf24' : '#f87171';
                    scoreEl.innerHTML = `专注度: <span style="color: ${scoreColor};">${score}%</span>`;
                } else {
                    statusEl.className = 'status-badge offline';
                    statusEl.innerHTML = '⚫ 离线';
                    scoreEl.innerHTML = '专注度: --';
                    imgEl.src = '';
                    imgEl.alt = data.message || '学生未在线';
                }
            } catch (e) {
                console.error('刷新实时画面失败:', e);
            }
        }

        function startAutoRefresh() {
            if (liveRefreshInterval) clearInterval(liveRefreshInterval);
            liveRefreshInterval = setInterval(refreshStudentLive, 2000);
        }

        function stopAutoRefresh() {
            if (liveRefreshInterval) {
                clearInterval(liveRefreshInterval);
                liveRefreshInterval = null;
            }
        }

        function toggleAutoRefresh() {
            if (document.getElementById('autoRefreshLive').checked) {
                startAutoRefresh();
            } else {
                stopAutoRefresh();
            }
        }

        function backToTeacherList() {
            stopAutoRefresh();
            currentViewingStudentId = null;
            showTeacherStudentList();
        }

        // ========== 学生端：监控页面 ==========
        const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
        let currentScene = 'selfstudy';
        let ws = null;
        let videoElement = null;
        let mediaStream = null;
        let frameInterval = null;
        let studyStartTime = null;
        let distractCount = 0;
        let pomodoroInterval = null;
        let pomodoroSeconds = 25 * 60;
        let pomodoroRunning = false;
        let pomodoroTotalSeconds = 25 * 60;
        let trendData = Array(10).fill(0);
        let reminderHistory = [];
        let voiceBroadcastEnabled = true;
        let speechSynth = window.speechSynthesis || window.webkitSpeechSynthesis;
        let preferredVoice = null;
        let asrEnabled = false;
        let recognition = null;
        let lastFocusLevel = 'good';
        let encourageCooldown = 0;
        let behaviorSaveInterval = null;

        function showStudentMonitor() {
            showPage('studentMonitorPage');
            document.getElementById('studentGreeting').textContent = `👨‍🎓 ${currentUser.name || currentUser.username}`;
            initMonitor();

        }



        function initMonitor() {
            initCamera();
            initWebSocket();
            frameInterval = setInterval(sendFrame, 500);
            renderTrend();
            initASR();
            loadPreferredVoice();
            studyStartTime = Date.now();
            updateStudyDuration();
            behaviorSaveInterval = setInterval(saveBehaviorToBackend, 60000);
            // 轮询获取 Coze 发来的提醒通知
            notificationInterval = setInterval(checkNotifications, 3000);
        }

        let notificationInterval = null;

        async function checkNotifications() {
            if (!currentUser || !currentUser.student_id) return;
            try {
                const resp = await fetch(`/api/student/notifications?student_id=${currentUser.student_id}`);
                const data = await resp.json();
                if (data.success && data.notification) {
                    const msg = data.notification.message;
                    addReminder('📢 ' + msg, 'encourage');
                    speak(msg);
                    showSpeechToast(msg);
                }
            } catch(e) { console.error('获取提醒失败:', e); }
        }

        function stopCamera() {
            if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
            if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
            if (pomodoroInterval) { clearInterval(pomodoroInterval); pomodoroInterval = null; }
            if (behaviorSaveInterval) { clearInterval(behaviorSaveInterval); behaviorSaveInterval = null; }
            if (notificationInterval) { clearInterval(notificationInterval); notificationInterval = null; }
            if (recognition) { try { recognition.stop(); } catch(e) {} asrEnabled = false; }
            if (ws) { ws.close(); ws = null; }
        }

        let lastBehaviorData = null;

        async function saveBehaviorToBackend() {
            if (!lastBehaviorData || !currentUser) return;
            try {
                await apiRequest('/api/student/behavior', {
                    method: 'POST',
                    body: JSON.stringify(lastBehaviorData)
                });
            } catch(e) { console.error('保存行为数据失败:', e); }
        }

        async function initCamera() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                    audio: true
                });
                videoElement = document.getElementById('video');
                videoElement.srcObject = mediaStream;
                videoElement.play();
            } catch(e) {
                console.error('❌ 摄像头/麦克风错误:', e);
                document.getElementById('connectionText').innerText = '设备未授权';
            }
        }

        function initWebSocket() {
            ws = new WebSocket(WS_URL);
            ws.onopen = () => {
                document.getElementById('connectionDot').style.background = '#22c55e';
                document.getElementById('connectionText').innerText = '已连接';
            };
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                lastBehaviorData = data;
                updateUI(data);
            };
            ws.onclose = () => {
                document.getElementById('connectionDot').style.background = '#ef4444';
                document.getElementById('connectionText').innerText = '重连中...';
                setTimeout(initWebSocket, 3000);
            };
            ws.onerror = () => {
                document.getElementById('connectionDot').style.background = '#ef4444';
                document.getElementById('connectionText').innerText = '连接失败';
            };
        }

        async function sendFrame() {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            if (!videoElement || videoElement.readyState < 2) return;
            const canvas = document.createElement('canvas');
            canvas.width = videoElement.videoWidth || 640;
            canvas.height = videoElement.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoElement, 0, 0);
            const base64 = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
            ws.send(JSON.stringify({ image: base64, scene: currentScene, student_id: currentUser?.student_id }));
        }

        // ========== 语音播报 ==========
        function loadPreferredVoice() {
            if (!speechSynth) return;
            const voices = speechSynth.getVoices();
            if (voices.length === 0) { speechSynth.onvoiceschanged = selectBestVoice; }
            else { selectBestVoice(); }
        }
        function selectBestVoice() {
            const voices = speechSynth.getVoices();
            const names = ['Google 普通话（中国大陆）','Microsoft Xiaoxiao','Microsoft Yaoyao','Ting-Ting','Lili'];
            for (const n of names) { const f = voices.find(v => v.name.includes(n) && v.lang.includes('zh')); if (f) { preferredVoice = f; return; } }
            const zh = voices.find(v => v.lang && v.lang.includes('zh'));
            if (zh) preferredVoice = zh;
        }
        function speak(text) {
            if (!voiceBroadcastEnabled || !speechSynth) return;
            speechSynth.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.lang = 'zh-CN'; u.rate = 1.0; u.pitch = 1.0; u.volume = 0.9;
            if (preferredVoice) u.voice = preferredVoice;
            showSpeechToast(text);
            u.onend = () => hideSpeechToast(); u.onerror = () => hideSpeechToast();
            speechSynth.speak(u);
        }
        function showSpeechToast(t) { const el = document.getElementById('speechToast'); el.textContent = '🔊 ' + t; el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 3000); }
        function hideSpeechToast() { document.getElementById('speechToast').classList.remove('show'); }
        function toggleVoiceBroadcast() {
            voiceBroadcastEnabled = !voiceBroadcastEnabled;
            const btn = document.getElementById('voiceBtn');
            if (voiceBroadcastEnabled) { btn.innerHTML = '🔊 语音播报'; btn.classList.add('active'); speak('语音播报已开启'); }
            else { btn.innerHTML = '🔇 语音关闭'; btn.classList.remove('active'); speechSynth.cancel(); }
        }

        // ========== 语音识别 ==========
        function initASR() {
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SR) return;
            recognition = new SR(); recognition.lang = 'zh-CN'; recognition.continuous = true; recognition.interimResults = true;
            recognition.onresult = (e) => {
                let final = '', interim = '';
                for (let i = e.resultIndex; i < e.results.length; i++) { if (e.results[i].isFinal) final += e.results[i][0].transcript; else interim += e.results[i][0].transcript; }
                document.getElementById('asrText').textContent = final || interim || '正在聆听...';
                if (final && ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'asr_result', text: final, scene: currentScene }));
            };
            recognition.onend = () => { if (asrEnabled) try { recognition.start(); } catch(e) {} };
        }
        function toggleASR() {
            if (!recognition) return;
            asrEnabled = !asrEnabled;
            const btn = document.getElementById('asrBtn'), st = document.getElementById('voiceStatus');
            if (asrEnabled) { recognition.start(); btn.innerHTML = '🎙️ 识别中'; btn.classList.add('active'); st.style.display = 'flex'; }
            else { recognition.stop(); btn.innerHTML = '🎙️ 语音识别'; btn.classList.remove('active'); st.style.display = 'none'; }
        }

        // ========== UI更新 ==========
        function updateUI(data) {
            const focus = data.focus_score || 0;
            document.getElementById('focusValue').innerText = focus + '%';
            document.getElementById('focusScore').innerText = focus;
            document.getElementById('focusRing').style.setProperty('--focus', focus + '%');
            if (data.annotated_image) {
                const canvas = document.getElementById('poseCanvas'), ctx = canvas.getContext('2d'), img = new Image();
                img.onload = () => { canvas.width = img.width; canvas.height = img.height; ctx.drawImage(img, 0, 0); };
                img.src = 'data:image/jpeg;base64,' + data.annotated_image;
            }
            
            // 显示模型预测信息
            if (data.model_prediction) {
                document.getElementById('modelPrediction').textContent = data.model_prediction;
            }
            
            let sb = document.getElementById('statusBar'), st = document.getElementById('statusText'), fl = 'good';
            let actionLabel = document.getElementById('actionLabel');
            const actionMap = { 'focused': '', 'sleep': '😴 检测到睡觉', 'playphone': '📱 检测到玩手机', 'distracted': '⚠️ 检测到分心' };
            if (focus >= 70) { sb.className = 'status-bar focused'; st.innerText = '高度专注中'; fl = 'good'; }
            else if (focus >= 40) { sb.className = 'status-bar distracted'; st.innerText = '轻度分心'; fl = 'warning'; }
            else { sb.className = 'status-bar fatigued'; st.innerText = '严重分心/疲劳'; fl = 'danger'; }
            actionLabel.innerText = actionMap[data.action_label] || '';
            const now = Date.now();
            if (now - encourageCooldown > 30000) {
                if (fl === 'good' && lastFocusLevel !== 'good') { speak('太棒了！你重新找回了专注状态，继续保持！'); encourageCooldown = now; }
                else if (fl === 'danger' && lastFocusLevel !== 'danger') { speak('注意，你的专注度很低，建议休息一下再继续学习'); encourageCooldown = now; }
            }
            lastFocusLevel = fl;
            distractCount = data.distraction_count || 0;
            document.getElementById('distractCount').innerText = distractCount;
            document.getElementById('headPose').innerText = data.head_pose || '正常';
            document.getElementById('bodyStability').innerText = data.body_stability || '稳定';
            document.getElementById('fatigueLevel').innerText = data.fatigue_level || '正常';
            document.getElementById('eyeFocus').innerText = data.eye_focus || '注视屏幕';
            document.getElementById('aiSuggestion').innerText = data.suggestion || '系统分析中...';
            if (data.focus_score !== undefined) { trendData.push(data.focus_score); if (trendData.length > 10) trendData.shift(); renderTrend(); }
        }
        function renderTrend() {
            const c = document.getElementById('trendTimeline'), mh = 40;
            c.innerHTML = trendData.map(v => {
                let co; if (v >= 70) co = 'linear-gradient(to top, #22d3ee, #6366f1)'; else if (v >= 40) co = 'linear-gradient(to top, #fbbf24, #f59e0b)'; else co = 'linear-gradient(to top, #f87171, #ef4444)';
                return `<div class="timeline-bar" style="height:${(v/100)*mh}px; background:${co};" title="${v}%"></div>`;
            }).join('');
        }
        function updateStudyDuration() {
            setInterval(() => {
                if (studyStartTime) {
                    let e = Math.floor((Date.now() - studyStartTime) / 1000), m = Math.floor(e / 60), s = e % 60;
                    document.getElementById('studyDuration').innerText = String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
                }
            }, 1000);
        }

        // ========== 番茄钟 ==========
        function setPomodoroTime(minutes) {
            if (pomodoroRunning) { clearInterval(pomodoroInterval); pomodoroRunning = false; document.getElementById('pomodoroBtn').innerHTML = '▶ 开始'; }
            pomodoroTotalSeconds = minutes * 60; pomodoroSeconds = pomodoroTotalSeconds; updatePomodoroDisplay();
            document.querySelectorAll('.time-presets .btn').forEach(b => b.className = 'btn btn-small btn-outline');
            event.target.className = 'btn btn-small';
            document.getElementById('pomodoroPhase').innerText = `已设置为 ${minutes} 分钟`;
            document.getElementById('customMinutes').value = minutes;
        }
        function updateCustomTime() { let m = parseInt(document.getElementById('customMinutes').value); if (m < 1) m = 1; if (m > 180) m = 180; document.getElementById('customMinutes').value = m; }
        function applyCustomTime() { let m = parseInt(document.getElementById('customMinutes').value); if (isNaN(m) || m < 1) m = 25; if (m > 180) m = 180; setPomodoroTime(m); document.querySelectorAll('.time-presets .btn').forEach(b => b.className = 'btn btn-small btn-outline'); }
        function updatePomodoroDisplay() { let m = Math.floor(pomodoroSeconds / 60), s = pomodoroSeconds % 60; document.getElementById('pomodoroTimer').innerText = String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0'); }
        function togglePomodoro() {
            if (pomodoroRunning) { clearInterval(pomodoroInterval); pomodoroRunning = false; document.getElementById('pomodoroBtn').innerHTML = '▶ 继续'; document.getElementById('pomodoroPhase').innerText = '已暂停'; }
            else {
                pomodoroRunning = true; document.getElementById('pomodoroBtn').innerHTML = '⏸ 暂停'; document.getElementById('pomodoroPhase').innerText = '专注中...';
                pomodoroInterval = setInterval(() => {
                    if (pomodoroSeconds > 0) { pomodoroSeconds--; updatePomodoroDisplay(); }
                    else { clearInterval(pomodoroInterval); pomodoroRunning = false; document.getElementById('pomodoroPhase').innerText = '🍅 时间到！休息一下吧'; document.getElementById('pomodoroBtn').innerHTML = '▶ 开始'; addReminder('🍅 番茄钟结束，建议休息5分钟', 'system'); speak('番茄钟时间到了！休息一下吧'); }
                }, 1000);
            }
        }
        function resetPomodoro() { clearInterval(pomodoroInterval); pomodoroRunning = false; pomodoroSeconds = pomodoroTotalSeconds; updatePomodoroDisplay(); document.getElementById('pomodoroBtn').innerHTML = '▶ 开始'; document.getElementById('pomodoroPhase').innerText = '准备开始'; }

        // ========== 提醒系统 ==========
        function addReminder(text, type) {
            reminderHistory.push({ text, type, time: Date.now() });
            const c = document.getElementById('reminderHistory');
            const colors = { 'auto': '#fbbf24', 'system': '#a855f7', 'encourage': '#34d399', 'info': '#22d3ee' };
            c.innerHTML = reminderHistory.slice(-5).reverse().map(r => `<div style="padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05); color:${colors[r.type]||'#94a3b8'};">[${new Date(r.time).toLocaleTimeString()}] ${r.text}</div>`).join('');
        }
        function sendEncouragement() {
            const msgs = ['💪 你做得很好，继续保持！','🌟 专注的你最棒！','🎯 目标就在前方，加油！','📚 每一分钟的努力都有意义','✨ 你已经很棒了，再坚持一下','🔥 你的坚持终将美好！'];
            let msg = msgs[Math.floor(Math.random() * msgs.length)];
            addReminder(msg, 'encourage'); speak(msg.replace(/[💪🌟🎯📚✨🔥]\s*/g, ''));
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'encouragement', message: msg }));
        }
  function exportReport() {
            let report = `【AI学伴 · 专注度学习报告】\n学生: ${currentUser?.name || '--'}\n导出时间: ${new Date().toLocaleString()}\n场景: ${currentScene}\n专注评分: ${document.getElementById('focusScore').innerText}\n分心次数: ${distractCount}\n学习时长: ${document.getElementById('studyDuration').innerText}\n状态: ${document.getElementById('statusText').innerText}\n建议: ${document.getElementById('aiSuggestion').innerText}\n--- 报告结束 ---`;
            let blob = new Blob([report], {type: 'text/plain'}), a = document.createElement('a');
            a.href = URL.createObjectURL(blob); a.download = `focus_report_${(currentUser?.name||'unknown')}_${new Date().toISOString().slice(0,10)}.txt`; a.click();
        }

        // ========== 初始化 ==========
        if (currentToken && currentUser) {
            if (currentUser.role === 'student') showStudentMonitor();
            else if (currentUser.role === 'teacher') showTeacherStudentList();
        } else { showLogin(); }
    </script>
</body>
</html>
'''


# ========== Coze API 路由（供 Coze Bot 调用） ==========

@app.get("/api/coze/students")
async def coze_get_students():
    """Coze: 获取所有学生列表"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, name, student_no, class_name FROM students")
        students = []
        for row in cursor.fetchall():
            students.append({
                "id": row[0],
                "username": row[1],
                "name": row[2] or "",
                "student_no": row[3] or "",
                "class_name": row[4] or ""
            })
        conn.close()
        return {"success": True, "students": students, "count": len(students)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/coze/student/online")
async def coze_get_online_students():
    try:
        online = []
        for student_id, cache in student_video_cache.items():
            is_online = (time.time() - cache.get("timestamp", 0)) < 10
            if is_online:
                online.append({
                    "id": student_id,
                    "name": cache.get("name", f"学生{student_id}"),
                    "focus_score": cache.get("focus_score", 0),
                    "action_label": cache.get("action_label", "unknown"),
                    "action_label_cn": {"focused": "专注", "sleep": "睡觉", "playphone": "玩手机", "distracted": "分心"}.get(cache.get("action_label", "unknown"), "未知")
                })
        
        return {
            "success": True,
            "online_students": online,  # 保留原名
            "students": online,         # ✅ 新增别名，兼容 Coze
            "count": len(online)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/coze/student/live")
async def coze_student_live(
    student_id: int = Query(..., description="学生ID")
):
    """Coze Bot 获取学生实时状态"""
    student = get_student_by_id(student_id)    
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    
    cache = get_student_video_cache(student_id)
    if not cache:
        return {
            "student_id": student_id,
            "student_name": student["name"],
            "online": False,
            "message": "学生未在线或未开启监控"
        }
    
    is_online = (time.time() - cache["timestamp"]) < 10
    
    return {
        "student_id": student_id,
        "student_name": student["name"],
        "online": is_online,
        "focus_score": cache["focus_score"],
        "image_base64": cache.get("image", ""),  # ✅ 新增: 返回截图
        "timestamp": cache["timestamp"]
    }


@app.get("/api/coze/student/records")
async def coze_student_records(
    student_id: int = Query(..., description="学生ID"),
    limit: int = Query(10, description="返回记录数量", ge=1, le=100)
):
    """Coze Bot 获取学生行为记录"""
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    
    records = get_student_behavior_records(student_id, limit=limit)
    
   # 简化记录，只返回关键字段
    simplified_records = []
    for r in records:
        simplified_records.append({
            "focus_score": r.get("focus_score"),
            "distraction_count": r.get("distraction_count"),
            "head_pose": r.get("head_pose"),
            "body_stability": r.get("body_stability"),
            "fatigue_level": r.get("fatigue_level"),
            "eye_focus": r.get("eye_focus"),
            "suggestion": r.get("suggestion"),
            "created_at": r.get("created_at")
        })
    
    return {
        "student_id": student_id,
        "student_name": student["name"],
        "total_records": len(simplified_records),
        "records": simplified_records
    }


# 发送语音提醒
@app.post("/api/coze/student/notify")
async def coze_notify_student(
    student_id: int = Query(..., description="学生ID"),
    message: str = Query(..., description="提醒内容")
):
    """Coze: 向学生发送语音提醒"""
    try:
        # 将提醒存入数据库或内存，学生端轮询获取
        notification = {
            "student_id": student_id,
            "message": message,
            "timestamp": time.time(),
            "type": "voice"
        }
        # 存储到全局变量或数据库
        student_notifications[student_id] = notification
        return {"success": True, "message": "提醒已发送"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 调试接口：查看所有缓存
@app.get("/api/debug/cache")
async def debug_cache():
    """调试接口：查看所有学生视频缓存"""
    result = {}
    for sid, cache in student_video_cache.items():
        result[sid] = {
            "has_image": cache.get("image") is not None,
            "image_length": len(cache.get("image", "")),
            "timestamp": cache.get("timestamp"),
            "name": cache.get("name"),
            "focus_score": cache.get("focus_score")
        }
    return {
        "total_students": len(student_video_cache),
        "caches": result
    }

# 获取学生最新截图 (返回 JSON，包含 image_url 图片链接，供 Coze 使用)
@app.get("/api/coze/student/snapshot")
async def coze_get_snapshot(
    student_id: int = Query(..., description="学生ID"),
    base_url: str = Query("https://foucs-analyze.cyou", description="API 基础地址")
):
    """Coze: 获取学生摄像头最新截图，返回 JSON（轻量，不含 base64）"""
    try:
        print(f"🔍 查询截图: student_id={student_id}")
        cache = get_student_video_cache(student_id)
        
        if cache:
            cache_age = time.time() - cache.get("timestamp", 0)
            print(f"📦 缓存状态: has_image={cache.get('image') is not None}, age={cache_age:.1f}秒")
        else:
            print(f"📦 缓存状态: cache=None")
        
        if not cache or not cache.get("image"):
            print(f"❌ 无截图: student_id={student_id}")
            return {
                "success": False,
                "student_id": student_id,
                "image_url": "",
                "timestamp": 0,
                "message": "暂无截图，学生可能未开启摄像头"
            }

        print(f"✅ 返回图片链接: student_id={student_id}")
        
        # 生成图片 URL（指向二进制图片接口）
        image_url = f"{base_url}/api/coze/student/snapshot/image?student_id={student_id}"
        
        return {
            "success": True,
            "student_id": student_id,
            "image_url": image_url,
            "timestamp": cache.get("timestamp", 0),
            "message": "获取成功"
        }
    except Exception as e:
        print(f"❌ 截图异常: {e}")
        return {
            "success": False,
            "student_id": student_id,
            "image_url": "",
            "timestamp": 0,
            "focus_score": cache.get("focus_score", 0),
            "message": str(e)
        }


# 获取学生截图图片 (返回二进制 JPEG 图片)
@app.get("/api/coze/student/snapshot/image")
async def coze_get_snapshot_image(
    student_id: int = Query(..., description="学生ID")
):
    """返回学生摄像头截图的二进制 JPEG 图片"""
    try:
        cache = get_student_video_cache(student_id)
        if not cache or not cache.get("image"):
            # 返回 1x1 透明 PNG 占位图
            import io
            placeholder = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
            return StreamingResponse(io.BytesIO(placeholder), media_type="image/png")

        import io
        from urllib.parse import quote
        image_bytes = base64.b64decode(cache["image"])
        student_name = cache.get("name", "")
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Student-ID": str(student_id),
                "X-Student-Name": quote(student_name, safe=''),
                "X-Focus-Score": str(cache.get("focus_score", 0)),
            }
        )
    except Exception as e:
        import io
        placeholder = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        return StreamingResponse(io.BytesIO(placeholder), media_type="image/png")


# 获取学生最新截图
@app.get("/api/coze/student/snapshot/json")
async def coze_get_snapshot_json(
    student_id: int = Query(..., description="学生ID")
):
    """Coze: 获取学生摄像头最新截图 (JSON 格式，包含 base64)"""
    try:
        cache = get_student_video_cache(student_id)
        if not cache or not cache.get("image"):
            return {"success": False, "message": "暂无截图，学生可能未开启摄像头"}

        return {
            "success": True,
            "student_id": student_id,
            "student_name": cache.get("name", ""),
            "image_base64": cache["image"],
            "focus_score": cache.get("focus_score"),
            "timestamp": cache["timestamp"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Coze 插件：获取学生实时动作分类
@app.get("/api/coze/student/action")
async def coze_student_action(
    student_id: int = Query(..., description="学生ID")
):
    """
    Coze 插件接口：获取学生当前动作分类
    返回：focused / sleep / playphone / distracted
    """
    try:
        student = get_student_by_id(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="学生不存在")

        cache = get_student_video_cache(student_id)
        if not cache:
            return {
                "success": True,
                "student_id": student_id,
                "student_name": student["name"],
                "online": False,
                "action_label": "unknown",
                "action_label_cn": "未知",
                "focus_score": 0,
                "message": "学生未在线"
            }

        is_online = (time.time() - cache["timestamp"]) < 10
        action_label = cache.get("action_label", "unknown")

        # 动作标签中文映射
        action_map = {
            "focused": "专注",
            "sleep": "睡觉",
            "playphone": "玩手机",
            "distracted": "分心",
            "unknown": "未知"
        }

        return {
            "success": True,
            "student_id": student_id,
            "student_name": student["name"],
            "online": is_online,
            "action_label": action_label,
            "action_label_cn": action_map.get(action_label, "未知"),
            "focus_score": cache.get("focus_score", 0),
            "head_pose": cache.get("head_pose", ""),
            "eye_focus": cache.get("eye_focus", ""),
            "fatigue_level": cache.get("fatigue_level", ""),
            "timestamp": cache["timestamp"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
async def root():
    return HTMLResponse(HTML)
@app.get("/api/student/notifications")
async def get_student_notifications(student_id: int):
    """学生端：获取自己的提醒"""
    try:
        notification = student_notifications.get(student_id)
        if notification:
            # 获取后删除，避免重复提醒
            del student_notifications[student_id]
            return {"success": True, "notification": notification}
        return {"success": True, "notification": None}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # 初始化数据库
    init_db()
    print("=" * 50)
    print("🎓 AI学伴 · 专注度智能监护系统")
    print("📍 访问地址: http://0.0.0.0:8001")  # 改成 0.0.0.0，方便你确认
    print("👤 学生: 注册后登录 → 进入专注度监控")
    print("👨‍🏫 教师: teacher / 123456 → 查看学生列表")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001)
