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
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import cv2                         
import mediapipe as mp          
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


# ========== 前端页面（已抽离至 templates/index.html） ==========


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


@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))
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
