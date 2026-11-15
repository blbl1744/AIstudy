"""
P3-3 核心守护用例（test_core）

覆盖 6 组防线：特征契约、规则分区间、WebSocket 握手鉴权、越权矩阵、
数据库迁移幂等、时间格式与健康检查。

约束：本文件只读业务代码，不修改任何 app/ 下的实现；所有断言均以「重构前后
行为零漂移」为目标，任何一条失败都意味着重构破坏了既有契约。
"""
import base64
import copy
from datetime import datetime, timezone

import pytest
from starlette.websockets import WebSocketDisconnect

from app import state as state_module
from app.analyzer import MediaPipePoseAnalyzer
from app.db import get_db, init_db
from app.model_loader import FEATURE_DIM, assert_model_contract, extract_model_features

# 合法 Origin（与 conftest 中 AI_STUDY_CORS_ORIGINS 保持一致）
VALID_ORIGIN = "http://localhost:8001"
INVALID_ORIGIN = "http://evil.example.com"

# 学生密码（长度满足注册最小 6 位）
STUDENT_PASSWORD = "stu-pass-123456"
TEACHER_PASSWORD = "tea-pass-123456"


# ========== 测试内部工具 ==========
def _register_and_login_student(client, username: str, name: str = "测试学生") -> dict:
    """注册并登录学生，返回 {token, student_id, headers}"""
    resp = client.post("/api/student/register", json={
        "username": username,
        "password": STUDENT_PASSWORD,
        "name": name,
        "student_no": "20230001",
        "class_name": "高一1班",
    })
    assert resp.status_code == 200, resp.text
    student_id = resp.json()["student_id"]

    login = client.post("/api/student/login", json={"username": username, "password": STUDENT_PASSWORD})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {"token": token, "student_id": student_id, "headers": {"Authorization": f"Bearer {token}"}}


def _register_and_login_teacher(client, username: str) -> dict:
    """注册并登录教师，返回 {token, headers}"""
    resp = client.post("/api/teacher/register", json={"username": username, "password": TEACHER_PASSWORD})
    assert resp.status_code == 200, resp.text

    login = client.post("/api/teacher/login", json={"username": username, "password": TEACHER_PASSWORD})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {"token": token, "headers": {"Authorization": f"Bearer {token}"}}


def _rule_chain(landmarks):
    """按 analyze() 中姿态分支的真实顺序调用规则引擎，返回 (分数, 姿态状态, 身体状态)

    与 analyzer.analyze 的取值路径逐行对应：先由关键点算倾斜/下沉/肩高差/摆动，
    再走 _evaluate_* 分级，最后交给 _calculate_focus_score 出分。
    """
    engine = MediaPipePoseAnalyzer()
    nose, left_shoulder, right_shoulder = landmarks[0], landmarks[11], landmarks[12]
    left_hip, right_hip = landmarks[23], landmarks[24]

    head_tilt = engine._calc_head_tilt(nose, left_shoulder, right_shoulder)
    head_drop = engine._calc_head_drop(nose, left_shoulder, right_shoulder)
    head_pose_status = engine._evaluate_head_pose(head_tilt, head_drop)

    shoulder_level = abs(left_shoulder.y - right_shoulder.y)
    body_sway = engine._calc_body_sway(left_shoulder, right_shoulder, left_hip, right_hip)
    body_status = engine._evaluate_body_stability(shoulder_level, body_sway)

    stability = engine._calculate_pose_stability(head_tilt, shoulder_level)
    score = engine._calculate_focus_score(
        head_pose_status, body_status, "未检测", "未检测", "selfstudy", stability
    )
    return score, head_pose_status, body_status


def _close_code(client, url: str, headers=None):
    """建立 WebSocket 连接并返回服务端关闭码；若握手成功则视为用例失败"""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(url, headers=headers or {}):
            pass
    return exc_info.value.code


# ========== 1. 特征契约 ==========
def test_feature_contract(fake_landmarks):
    """13 维特征契约：输出形状严格为 (13,)，维度不符的模型必须抛 RuntimeError"""
    features = extract_model_features(fake_landmarks)
    assert features is not None, "33 点完整关键点不应提取失败"
    assert features.shape == (FEATURE_DIM,), f"特征形状应为 ({FEATURE_DIM},)，实际 {features.shape}"
    assert FEATURE_DIM == 13, "契约维度必须恒为 13，任何变更都需要同步升级特征版本"

    class _WrongDimModel:
        """模拟训练期使用旧维度（51 维）的模型"""
        n_features_in_ = 51

    with pytest.raises(RuntimeError):
        assert_model_contract(_WrongDimModel())

    class _GoodDimModel:
        n_features_in_ = 13

    assert assert_model_contract(_GoodDimModel()) is True

    # 关键点缺失（不足 33 点）时必须安全返回 None，而不是抛异常打断推理
    assert extract_model_features(fake_landmarks[:20]) is None
    assert extract_model_features(None) is None


# ========== 2. 规则分区间 ==========
def test_rule_score_range(fake_landmarks):
    """端正坐姿高分、低头+晃动明显掉分、无人时 focus_score 为 None"""
    good_score, good_head, good_body = _rule_chain(fake_landmarks)
    assert good_head == "端正", f"端正坐姿应判定为「端正」，实际 {good_head}"
    assert good_body == "稳定", f"端正坐姿身体应判定为「稳定」，实际 {good_body}"
    assert good_score >= 85, f"端正坐姿规则分应 >= 85，实际 {good_score}"

    # 明显低头（鼻尖接近肩线）
    head_down = copy.deepcopy(fake_landmarks)
    head_down[0].y = 0.58
    # 明显晃动（髋部相对肩部横向偏移）
    head_down[23].x = 0.20
    head_down[24].x = 0.32

    bad_score, bad_head, bad_body = _rule_chain(head_down)
    assert bad_head == "明显低头", f"应判定「明显低头」，实际 {bad_head}"
    assert bad_body == "明显晃动", f"应判定「明显晃动」，实际 {bad_body}"
    assert bad_score <= good_score - 15, f"低头+晃动应明显掉分：{good_score} → {bad_score}"

    # 未检测到人体/关键点：返回 focus_score=None 且不抛异常
    invalid_image = base64.b64encode(b"not-a-real-image").decode("utf-8")
    result = MediaPipePoseAnalyzer().analyze(invalid_image, scene="selfstudy", student_id=1)
    assert isinstance(result, dict)
    assert result.get("focus_score") is None, "未检测到人体时 focus_score 必须为 None"
    assert result.get("pose_detected") is False


# ========== 3. WebSocket 握手鉴权 ==========
def test_ws_auth(client, monkeypatch):
    """无 token → 4401；非法 Origin → 4403；身份只认令牌，消息体伪造的 student_id 无效"""
    student_a = _register_and_login_student(client, "stu_a", "学生甲")
    student_b = _register_and_login_student(client, "stu_b", "学生乙")

    # 3.1 缺少令牌
    assert _close_code(client, "/ws") == 4401
    # 3.2 伪造/非法令牌
    assert _close_code(client, "/ws?token=invalid.token.value") == 4401
    # 3.3 非法 Origin（令牌合法也不能放行）
    assert _close_code(
        client, f"/ws?token={student_a['token']}", headers={"origin": INVALID_ORIGIN}
    ) == 4403

    # 3.4 合法 token + 合法 Origin：连接成功，且身份只取自令牌
    from app.analyzer import analyzer as analyzer_instance

    def _fake_analyze(image_base64, scene="selfstudy", student_id=None):
        """替身推理：直接返回带标注帧的成功结果，跳过真实视觉推理"""
        return {
            "focus_score": 88,
            "action_label": "学习",
            "annotated_image": "ZmFrZS1mcmFtZQ==",
            "head_pose": "端正",
            "eye_focus": "注视屏幕",
            "fatigue_level": "精力充沛",
        }

    monkeypatch.setattr(analyzer_instance, "analyze", _fake_analyze)

    with client.websocket_connect(
        f"/ws?token={student_a['token']}", headers={"origin": VALID_ORIGIN}
    ) as websocket:
        # 消息体中伪造 student_id=B，服务端必须忽略
        websocket.send_json({"image": "ZmFrZQ==", "scene": "selfstudy", "student_id": student_b["student_id"]})
        payload = websocket.receive_json()
        assert payload.get("focus_score") == 88

        # 连接仍在时校验缓存归属：只记录学生 A，学生 B 未被污染
        # （online_students 在连接断开时会主动清理，因此断言必须放在会话内）
        assert state_module.student_video_cache.get(student_a["student_id"]) is not None
        assert state_module.student_video_cache.get(student_b["student_id"]) is None
        assert state_module.online_students.get(student_a["student_id"]) is not None
        assert state_module.online_students.get(student_b["student_id"]) is None

    # 断开后：实时画面缓存保留（供教师端 10 秒内拉取），在线状态被清理
    assert state_module.student_video_cache.get(student_a["student_id"]) is not None
    assert state_module.online_students.get(student_a["student_id"]) is None
    assert state_module.student_video_cache.get(student_b["student_id"]) is None


# ========== 4. 越权矩阵 ==========
def test_authz_matrix(client):
    """角色隔离与输入侧 XSS 拦截：越权一律 403/404，非法姓名 422"""
    student = _register_and_login_student(client, "stu_z", "学生乙")
    teacher = _register_and_login_teacher(client, "tea_z")
    student_id = student["student_id"]

    # 4.1 学生访问教师接口 → 403
    assert client.get("/api/teacher/students", headers=student["headers"]).status_code == 403
    assert client.get(
        f"/api/teacher/students/{student_id}/records", headers=student["headers"]
    ).status_code == 403
    assert client.get(
        f"/api/teacher/students/{student_id}/snapshot.jpg", headers=student["headers"]
    ).status_code == 403
    # 4.2 默认未开启 AI_STUDY_DEBUG_API 时，调试接口根本不注册 → 404
    assert client.get("/api/debug/cache", headers=student["headers"]).status_code == 404

    # 4.3 教师访问学生专属接口 → 403
    assert client.get("/api/student/notifications", headers=teacher["headers"]).status_code == 403
    assert client.delete("/api/student/my-data", headers=teacher["headers"]).status_code == 403

    # 4.4 未携带令牌 → 401/403（HTTPBearer 拦截）
    assert client.get("/api/teacher/students").status_code in (401, 403)

    # 4.5 XSS 载荷姓名在输入侧被拦截（422），不得落库
    for bad_name in ("<script>alert(1)</script>", "<img src=x onerror=alert(1)>"):
        resp = client.post("/api/student/register", json={
            "username": "xss_user_1",
            "password": STUDENT_PASSWORD,
            "name": bad_name,
        })
        assert resp.status_code == 422, f"含 XSS 载荷的姓名应被 422 拦截，实际 {resp.status_code}"


# ========== 5. 数据库迁移幂等 ==========
def test_db_migration_idempotent(client):
    """init_db() 连续执行两次不报错；expire_at 列与复合索引必须存在"""
    # lifespan 已执行过一次 init_db，这里再连做两次验证幂等
    init_db()
    init_db()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(behavior_records)")
        columns = [row["name"] for row in cursor.fetchall()]
        assert "expire_at" in columns, "behavior_records 必须包含 expire_at 列（30 天留存到期时间）"

        cursor.execute("PRAGMA index_list(behavior_records)")
        indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_behavior_student_time" in indexes, "缺少复合索引 idx_behavior_student_time"

        cursor.execute("PRAGMA index_info(idx_behavior_student_time)")
        indexed_columns = [row["name"] for row in cursor.fetchall()]
        # 复合索引首列必须是 student_id，才能支撑「按学生取最新记录」的查询
        assert indexed_columns[0] == "student_id", f"复合索引首列应为 student_id，实际 {indexed_columns}"


# ========== 6. 时间格式与健康检查 ==========
def test_time_iso_utc_and_healthz(client):
    """行为记录 created_at 必须是带 Z 的 UTC ISO-8601；/healthz 返回 ok"""
    student = _register_and_login_student(client, "stu_t", "学生丙")
    teacher = _register_and_login_teacher(client, "tea_t")

    saved = client.post("/api/student/behavior", json={"focus_score": 90}, headers=student["headers"])
    assert saved.status_code == 200, saved.text

    records_resp = client.get(
        f"/api/teacher/students/{student['student_id']}/records", headers=teacher["headers"]
    )
    assert records_resp.status_code == 200, records_resp.text
    records = records_resp.json()["records"]
    assert records, "写入后应能查到行为记录"

    created_at = records[0]["created_at"]
    assert isinstance(created_at, str) and created_at.endswith("Z"), f"created_at 应以 Z 结尾：{created_at}"
    parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(None)

    # 30 天留存到期时间同样应是 UTC ISO-8601
    assert records[0]["expire_at"].endswith("Z")

    health = client.get("/healthz")
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["status"] == "ok"
    # 实现中数据库探活字段名为 database（与任务描述的 db 同义），此处按真实契约断言
    assert body["database"] == "ok"
    assert body["model"]["feature_dim"] == 13


# ========== 7. 教师督学提醒闭环（feat/extensions）==========
def test_teacher_notify_flow(client):
    """教师端发送提醒 → 学生端取走并播报的完整闭环

    断言三件事：
    1. 越权：学生调用发送接口必须 403（发送能力只属于教师）；
    2. 目标校验：向不存在的学生发送必须 404，不得静默丢进缓存；
    3. 端到端：教师发送 200 → 学生轮询拿到同一条 message/level/from_teacher → 再次轮询为空
       （说明消费即销毁，同一条提醒不会被重复播报）。
    """
    student = _register_and_login_student(client, "stu_notify", "学生丁")
    teacher = _register_and_login_teacher(client, "tea_notify")
    student_id = student["student_id"]
    notify_url = f"/api/teacher/students/{student_id}/notify"

    # 7.1 学生调用发送接口 → 403
    resp = client.post(notify_url, json={"message": "请保持坐姿端正"}, headers=student["headers"])
    assert resp.status_code == 403, f"学生无权发送提醒，实际 {resp.status_code}"

    # 7.2 教师向不存在的学生发送 → 404（用极大 id 规避自增冲突）
    resp = client.post(
        "/api/teacher/students/99999999/notify",
        json={"message": "请保持坐姿端正"},
        headers=teacher["headers"],
    )
    assert resp.status_code == 404, f"目标学生不存在应 404，实际 {resp.status_code}"

    # 7.3 教师向合法学生发送 → 200
    message = "注意集中注意力，放下无关物品"
    resp = client.post(notify_url, json={"message": message, "level": "info"}, headers=teacher["headers"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "提醒已发送"

    # 7.4 学生轮询：第一条就是刚发的那条提醒（字段完整、时间带 Z）
    first = client.get("/api/student/notifications", headers=student["headers"])
    assert first.status_code == 200, first.text
    notification = first.json()["notification"]
    assert notification is not None, "教师刚发送的提醒必须能被学生取到"
    assert notification["message"] == message
    assert notification["level"] == "info"
    assert notification["from_teacher"], "必须带上发送者标识，学生端才能渲染「来自 X 老师」"
    assert notification["created_at"].endswith("Z"), f"created_at 应为 UTC ISO-8601：{notification['created_at']}"

    # 7.5 再次轮询 → None，确认「读取即销毁」，同一条提醒不会重复播报
    second = client.get("/api/student/notifications", headers=student["headers"])
    assert second.status_code == 200, second.text
    assert second.json()["notification"] is None, "提醒只能被消费一次，重复轮询必须为空"

    # 7.6 入参边界：纯空白内容与超长内容一律 422，且不得污染缓存
    for bad_payload in ({"message": "   "}, {"message": "超" * 101}, {"message": ""}):
        resp = client.post(notify_url, json=bad_payload, headers=teacher["headers"])
        assert resp.status_code == 422, f"非法提醒内容应 422，实际 {resp.status_code}：{bad_payload}"
        assert state_module.student_notifications.get(student_id) is None

    # 7.7 级别非法值同样被结构校验拦截（Literal 白名单）
    resp = client.post(notify_url, json={"message": "请坐好", "level": "critical"}, headers=teacher["headers"])
    assert resp.status_code == 422, f"level 只允许 info/warning，实际 {resp.status_code}"

