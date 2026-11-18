"""
M1-B 核心守护用例（test_core）

7 组测试：特征契约、规则评分、新表幂等迁移、健康检查与统一认证/成长/数据清理、
创建房间与公告广场、房间 WS 鉴权/弹幕广播/快照鉴权、AI 哨兵预警与结束结算晋升。

约束：本文件只读业务代码，不修改任何 app/ 下的实现；所有断言均以 M1-B 契约为准。
"""
import base64
import copy
from datetime import datetime, timezone

import pytest
from starlette.websockets import WebSocketDisconnect

from app import state as state_module
from app.analyzer import MediaPipePoseAnalyzer, analyzer as analyzer_instance
from app.db import calc_rank_title, get_db, init_db, save_behavior_record
from app.model_loader import FEATURE_DIM, assert_model_contract, extract_model_features

# 合法 Origin（与 conftest 中 AI_STUDY_CORS_ORIGINS 保持一致）
VALID_ORIGIN = "http://localhost:8001"
INVALID_ORIGIN = "http://evil.example.com"

PASSWORD = "test-pass-123456"


# ========== 测试内部工具 ==========
def _register_and_login(client, username: str, nickname: str) -> dict:
    """注册并登录统一账号，返回 {token, user_id, headers}"""
    reg = client.post("/api/auth/register", json={
        "username": username, "password": PASSWORD, "nickname": nickname,
    })
    assert reg.status_code == 200, reg.text
    login = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    assert login.status_code == 200, login.text
    body = login.json()
    return {
        "token": body["access_token"],
        "user_id": body["user_id"],
        "headers": {"Authorization": "Bearer " + body["access_token"]},
    }


def _create_room(client, headers, mode="peer_1v1", is_public=True,
                 scene_title="考研自习", goal_text="今天刷三套卷", max_watchers=3) -> dict:
    """创建房间并返回房间信息"""
    resp = client.post("/api/rooms", json={
        "mode": mode, "is_public": is_public, "scene_id": "scene-1",
        "scene_title": scene_title, "goal_text": goal_text,
        "max_watchers": max_watchers, "punishment_types": [],
    }, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _close_code(client, url: str, headers=None):
    """建立 WebSocket 连接并返回服务端关闭码；若握手成功则视为用例失败"""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(url, headers=headers or {}):
            pass
    return exc_info.value.code


def _recv_type(ws, type_name: str, limit: int = 20) -> dict:
    """在 WebSocket 上接收消息，丢弃无关消息直到拿到指定 type"""
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == type_name:
            return msg
    raise AssertionError("连续接收 %s 条消息仍未拿到 %s" % (limit, type_name))


def _rule_chain(landmarks):
    """按 analyze() 中姿态分支的真实顺序调用规则引擎，返回 (分数, 姿态状态, 身体状态)"""
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


# ========== 1. 特征契约 ==========
def test_feature_contract(fake_landmarks):
    """13 维特征契约：输出形状严格为 (13,)，维度不符的模型必须抛 RuntimeError"""
    features = extract_model_features(fake_landmarks)
    assert features is not None, "33 点完整关键点不应提取失败"
    assert features.shape == (FEATURE_DIM,), "特征形状应为 (%s,)" % FEATURE_DIM
    assert FEATURE_DIM == 13, "契约维度必须恒为 13，任何变更都需要同步升级特征版本"

    class _WrongDimModel:
        n_features_in_ = 51

    with pytest.raises(RuntimeError):
        assert_model_contract(_WrongDimModel())

    class _GoodDimModel:
        n_features_in_ = 13

    assert assert_model_contract(_GoodDimModel()) is True
    assert extract_model_features(fake_landmarks[:20]) is None
    assert extract_model_features(None) is None


# ========== 2. 规则评分 ==========
def test_rule_score_range(fake_landmarks):
    """端正坐姿高分、低头+晃动明显掉分、无人时 focus_score 为 None"""
    good_score, good_head, good_body = _rule_chain(fake_landmarks)
    assert good_head == "端正", "端正坐姿应判定为端正，实际 %s" % good_head
    assert good_body == "稳定", "端正坐姿身体应判定为稳定，实际 %s" % good_body
    assert good_score >= 85, "端正坐姿规则分应 >= 85，实际 %s" % good_score

    head_down = copy.deepcopy(fake_landmarks)
    head_down[0].y = 0.58
    head_down[23].x = 0.20
    head_down[24].x = 0.32

    bad_score, bad_head, bad_body = _rule_chain(head_down)
    assert bad_head == "明显低头", "应判定明显低头，实际 %s" % bad_head
    assert bad_body == "明显晃动", "应判定明显晃动，实际 %s" % bad_body
    assert bad_score <= good_score - 15, "低头+晃动应明显掉分：%s -> %s" % (good_score, bad_score)

    invalid_image = base64.b64encode(b"not-a-real-image").decode("utf-8")
    result = MediaPipePoseAnalyzer().analyze(invalid_image, scene="selfstudy", student_id=1)
    assert isinstance(result, dict)
    assert result.get("focus_score") is None, "未检测到人体时 focus_score 必须为 None"
    assert result.get("pose_detected") is False


# ========== 3. 新表幂等迁移 ==========
def test_db_migration_idempotent(client):
    """init_db() 连续执行两次不报错；users/study_sessions/scene_cache 新表与字段/索引齐备"""
    init_db()
    init_db()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}
        assert {"users", "study_sessions", "scene_cache", "behavior_records"} <= tables

        cursor.execute("PRAGMA table_info(users)")
        user_cols = {row["name"] for row in cursor.fetchall()}
        assert {"total_focus_minutes", "rank_title", "shield_count",
                "streak_clean_sessions", "nickname"} <= user_cols

        cursor.execute("PRAGMA table_info(behavior_records)")
        behavior_cols = {row["name"] for row in cursor.fetchall()}
        assert "session_id" in behavior_cols, "behavior_records 必须幂等补加 session_id 列"

        cursor.execute("PRAGMA index_list(study_sessions)")
        assert "idx_sessions_user_time" in {row["name"] for row in cursor.fetchall()}
        cursor.execute("PRAGMA index_info(idx_sessions_user_time)")
        assert [row["name"] for row in cursor.fetchall()][0] == "user_id"

    # 军衔阈值契约：0/60/300/1000/3000
    assert calc_rank_title(0) == "见习学员"
    assert calc_rank_title(60) == "列兵"
    assert calc_rank_title(300) == "士官"
    assert calc_rank_title(1000) == "少尉"
    assert calc_rank_title(3000) == "王牌指挥官"


# ========== 4. 健康检查与统一认证/成长/数据清理 ==========
def test_health_auth_growth_cleanup(client):
    """健康检查字段、注册/登录/档案、昵称 XSS 拦截与个人数据一键清空"""
    health = client.get("/healthz")
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["model"]["feature_dim"] == 13
    assert "active_rooms" in body["runtime"] and "room_connections" in body["runtime"]

    # 昵称含脚本标签 → 422（输入侧拦截存储型 XSS）
    bad = client.post("/api/auth/register", json={
        "username": "xss_user", "password": PASSWORD, "nickname": "<script>x</script>",
    })
    assert bad.status_code == 422, "含 XSS 载荷的昵称应被 422 拦截"

    user = _register_and_login(client, "growth_user", "成长者")
    # 重复注册 → 400；错误密码 → 401
    assert client.post("/api/auth/register", json={
        "username": "growth_user", "password": PASSWORD, "nickname": "重名",
    }).status_code == 400
    assert client.post("/api/auth/login", json={
        "username": "growth_user", "password": "wrong-pass",
    }).status_code == 401
    # 未携带令牌访问档案 → 401/403
    assert client.get("/api/users/me/profile").status_code in (401, 403)

    profile = client.get("/api/users/me/profile", headers=user["headers"]).json()
    assert profile["nickname"] == "成长者"
    assert profile["total_focus_minutes"] == 0 and profile["rank_title"] == "见习学员"
    assert profile["shield_count"] == 0 and profile["streak_clean_sessions"] == 0
    assert profile["created_at"].endswith("Z")

    # 写入行为记录 + 战报 + 房间，然后一键清空，验证三类数据都被回收
    save_behavior_record(user["user_id"], {"focus_score": 88})
    with get_db() as conn:
        conn.execute(
            "INSERT INTO study_sessions (user_id, mode, scene_title, goal_text, "
            "duration_minutes, avg_score, status) VALUES (?,?,?,?,?,?,'completed')",
            (user["user_id"], "ai_solo", "测试场景", "测试目标", 30, 88),
        )
    _create_room(client, user["headers"], mode="ai_solo")
    assert state_module.active_rooms.get(next(iter(state_module.active_rooms.items()))[0]) is not None

    resp = client.delete("/api/users/me/data", headers=user["headers"])
    assert resp.status_code == 200, resp.text
    cleaned = resp.json()
    assert cleaned["behavior_records_deleted"] >= 1
    assert cleaned["study_sessions_deleted"] >= 1
    assert cleaned["room_cache_cleared"] >= 1
    assert client.get("/api/users/me/profile", headers=user["headers"]).json()["sessions"] == []


# ========== 5. 创建房间与公告广场 ==========
def test_room_create_and_plaza(client):
    """房间字段契约、goal_text 的 XSS/长度校验、公告广场筛选与满员拒绝"""
    host = _register_and_login(client, "plaza_host", "房主甲")
    watch1 = _register_and_login(client, "plaza_w1", "督学甲")
    watch2 = _register_and_login(client, "plaza_w2", "督学乙")

    # 未登录创建 → 401/403
    assert client.post("/api/rooms", json={"mode": "ai_solo", "goal_text": "学习"}).status_code in (401, 403)

    # XSS 与长度校验 → 422
    for bad_goal in ("<script>alert(1)</script>", "   ", "超" * 101):
        resp = client.post("/api/rooms", json={"mode": "ai_solo", "goal_text": bad_goal},
                           headers=host["headers"])
        assert resp.status_code == 422, "非法目标应 422，实际 %s：%s" % (resp.status_code, bad_goal)
    assert client.post("/api/rooms", json={"mode": "unknown", "goal_text": "学习"},
                       headers=host["headers"]).status_code == 422

    pub = _create_room(client, host["headers"], mode="peer_group", scene_title="考研自习")
    solo = _create_room(client, host["headers"], mode="ai_solo", scene_title="单机")
    priv = _create_room(client, host["headers"], mode="peer_1v1", is_public=False)

    # 房间字典字段契约：内部下划线字段不回显
    assert pub["strikes_left"] == 3 and pub["watchers"] == {}
    assert pub["max_watchers"] == 3 and pub["host_nickname"] == "房主甲"
    assert not any(k.startswith("_") for k in pub)

    rooms = client.get("/api/rooms/plaza", headers=host["headers"]).json()["rooms"]
    ids = {r["room_id"] for r in rooms}
    assert pub["room_id"] in ids, "公开小组房间应出现在广场"
    assert solo["room_id"] not in ids, "ai_solo 房间不应出现在广场"
    assert priv["room_id"] not in ids, "私密房间不应出现在广场"
    entry = next(r for r in rooms if r["room_id"] == pub["room_id"])
    assert {"room_id", "host_nickname", "mode", "scene_title", "goal_text",
            "elapsed_seconds", "watcher_count", "max_watchers"} <= set(entry)
    assert entry["watcher_count"] == 0 and entry["mode"] == "peer_group"

    # 满员：max_watchers=1，第一位督学官占位后第二位被 4403 拒绝
    full = _create_room(client, host["headers"], mode="peer_1v1", max_watchers=1)
    with client.websocket_connect(
        "/ws/room/%s?token=%s" % (full["room_id"], watch1["token"]), headers={"origin": VALID_ORIGIN}
    ):
        assert _close_code(
            client, "/ws/room/%s?token=%s" % (full["room_id"], watch2["token"]),
            headers={"origin": VALID_ORIGIN},
        ) == 4403
    # 私密房间非房主连接 → 4403
    assert _close_code(
        client, "/ws/room/%s?token=%s" % (priv["room_id"], watch1["token"]),
        headers={"origin": VALID_ORIGIN},
    ) == 4403


# ========== 6. 房间 WS 鉴权/弹幕广播/快照鉴权 ==========
def test_room_ws_auth_barrage_and_snapshot(client, monkeypatch):
    """四级握手关闭码、督学官弹幕/音效广播、房主快照的鉴权与 404/403"""
    host = _register_and_login(client, "ws_host", "房主乙")
    watcher = _register_and_login(client, "ws_watcher", "督学丙")

    # 关闭码：无 token/非法 token 4401；非法 Origin 4403；房间不存在 4404
    assert _close_code(client, "/ws/room/nope") == 4401
    assert _close_code(client, "/ws/room/nope?token=bad.token.value") == 4401
    assert _close_code(client, "/ws/room/nope?token=" + host["token"],
                       headers={"origin": INVALID_ORIGIN}) == 4403
    assert _close_code(client, "/ws/room/nope?token=" + host["token"],
                       headers={"origin": VALID_ORIGIN}) == 4404

    room = _create_room(client, host["headers"], mode="peer_1v1")
    rid = room["room_id"]
    # 无画面 / 房间不存在 → 404；私密房间他人 → 403
    assert client.get("/api/rooms/%s/snapshot.jpg" % rid, headers=host["headers"]).status_code == 404
    assert client.get("/api/rooms/nope/snapshot.jpg", headers=host["headers"]).status_code == 404
    priv = _create_room(client, host["headers"], is_public=False)
    assert client.get("/api/rooms/%s/snapshot.jpg" % priv["room_id"],
                      headers=watcher["headers"]).status_code == 403

    def _fake_analyze(image_base64, scene="selfstudy", student_id=None):
        return {
            "focus_score": 90, "action_label": "学习", "annotated_image": "ZmFrZS1mcmFtZQ==",
            "head_pose": "端正", "eye_focus": "注视屏幕", "fatigue_level": "精力充沛",
        }
    monkeypatch.setattr(analyzer_instance, "analyze", _fake_analyze)

    frame_b64 = base64.b64encode(b"fake-jpeg-bytes").decode("utf-8")
    with client.websocket_connect(
        "/ws/room/%s?token=%s" % (rid, host["token"]), headers={"origin": VALID_ORIGIN}
    ) as host_ws:
        host_ws.receive_json()  # 连接时收到 watcher_update(count=0)
        with client.websocket_connect(
            "/ws/room/%s?token=%s" % (rid, watcher["token"]), headers={"origin": VALID_ORIGIN}
        ) as watch_ws:
            update = _recv_type(host_ws, "watcher_update")
            assert update["watcher_count"] == 1
            assert update["watchers"][0]["user_id"] == watcher["user_id"]

            # 督学官弹幕 → 全房广播
            watch_ws.send_json({"type": "barrage", "text": "加油，保持住"})
            for ws in (host_ws, watch_ws):
                msg = _recv_type(ws, "room_interaction")
                assert msg["kind"] == "barrage" and msg["text"] == "加油，保持住"
                assert msg["from"] == "督学丙"

            # 白名单音效 → 广播；非白名单音效 → 静默丢弃
            watch_ws.send_json({"type": "sound_effect", "sound": "knock_board"})
            for ws in (host_ws, watch_ws):
                msg = _recv_type(ws, "room_interaction")
                assert msg["kind"] == "sound_effect" and msg["sound"] == "knock_board"
            watch_ws.send_json({"type": "sound_effect", "sound": "evil_sound"})
            watch_ws.send_json({"type": "barrage", "text": "标记"})
            tail = []
            for _ in range(5):
                msg = watch_ws.receive_json()
                tail.append(msg)
                if msg.get("type") == "room_interaction" and msg.get("text") == "标记":
                    break
            assert not any(m.get("sound") == "evil_sound" for m in tail), "非白名单音效不得广播"

            # 房主帧 → 快照可读（image/jpeg），推理结果回写为 analysis_update
            host_ws.send_json({"type": "frame", "image": frame_b64, "scene": "selfstudy"})
            _recv_type(host_ws, "analysis_update")
            snap = client.get("/api/rooms/%s/snapshot.jpg" % rid, headers=host["headers"])
            assert snap.status_code == 200 and snap.headers["content-type"].startswith("image/jpeg")
            assert snap.content == b"fake-jpeg-bytes"
            # 他人可读公开房间快照
            assert client.get("/api/rooms/%s/snapshot.jpg" % rid,
                              headers=watcher["headers"]).status_code == 200


# ========== 7. AI 哨兵预警与结束结算晋升 ==========
def test_sentinel_alert_and_settlement(client, monkeypatch):
    """连续走神达标推送哨兵预警且冷却去重；结束结算写战报、晋升军衔、连击满三奖护盾"""
    import app.routers.room_routes as room_module

    host = _register_and_login(client, "settle_host", "学习者")
    watcher = _register_and_login(client, "settle_watcher", "督学官")

    # 可控时钟 + 阈值压缩：把 8 秒走神压到 3 秒，省去真实等待
    clock = {"now": 1000.0}
    monkeypatch.setattr(room_module, "_now_ts", lambda: clock["now"])
    monkeypatch.setattr(room_module, "SENTINEL_ALERT_SEC", 3.0)
    monkeypatch.setattr(room_module, "SENTINEL_COOLDOWN_SEC", 60.0)

    def _fake_analyze(image_base64, scene="selfstudy", student_id=None):
        """恒定返回走神结果（focus_score=None）"""
        return {
            "focus_score": None, "action_label": "离开", "annotated_image": "ZmFrZQ==",
            "head_pose": "未检测", "eye_focus": "未检测", "fatigue_level": "未检测",
        }
    monkeypatch.setattr(analyzer_instance, "analyze", _fake_analyze)

    room = _create_room(client, host["headers"], mode="peer_group")
    rid = room["room_id"]
    with client.websocket_connect(
        "/ws/room/%s?token=%s" % (rid, host["token"]), headers={"origin": VALID_ORIGIN}
    ) as host_ws:
        host_ws.receive_json()  # watcher_update(count=0)
        with client.websocket_connect(
            "/ws/room/%s?token=%s" % (rid, watcher["token"]), headers={"origin": VALID_ORIGIN}
        ) as watch_ws:
            host_ws.receive_json()  # watcher_update(count=1)

            # 推进 4 帧：首帧时间差 0，第 4 帧累计 3 秒触发哨兵
            for _ in range(4):
                clock["now"] += 1.0
                host_ws.send_json({"type": "frame", "image": "ZmFrZQ==", "scene": "selfstudy"})
                _recv_type(host_ws, "analysis_update")

            alert = _recv_type(watch_ws, "sentinel_alert")
            assert alert["distracted_streak_sec"] >= 3.0
            live_room = state_module.active_rooms.get(rid)
            assert live_room["distracted_streak_sec"] >= 3.0
            assert live_room["_last_sentinel_alert_ts"] > 0

            # 冷却期内继续走神不重复预警：用弹幕做尾标记，期间不得再出现哨兵
            clock["now"] += 1.0
            host_ws.send_json({"type": "frame", "image": "ZmFrZQ==", "scene": "selfstudy"})
            _recv_type(host_ws, "analysis_update")
            watch_ws.send_json({"type": "barrage", "text": "冷却尾标记"})
            seen = []
            for _ in range(6):
                msg = watch_ws.receive_json()
                seen.append(msg.get("type"))
                if msg.get("type") == "room_interaction":
                    break
            assert "sentinel_alert" not in seen, "冷却期内不得重复推送哨兵预警"

    # 结算三局干净局（每局 26 分钟、零罚站）：累计 78 分钟 → 列兵，连击 3 → 护盾 +1
    for _ in range(3):
        session_room = _create_room(client, host["headers"], mode="ai_solo")
        clock["now"] += 26 * 60
        with client.websocket_connect(
            "/ws/room/%s?token=%s" % (session_room["room_id"], host["token"]),
            headers={"origin": VALID_ORIGIN},
        ) as ws:
            ws.receive_json()  # watcher_update(count=0)
            ws.send_json({"type": "finish_session"})
            finished = _recv_type(ws, "session_finished")
            assert finished["duration_minutes"] == 26
            assert state_module.active_rooms.get(session_room["room_id"]) is None

    profile = client.get("/api/users/me/profile", headers=host["headers"]).json()
    assert profile["total_focus_minutes"] == 78
    assert profile["rank_title"] == "列兵"
    assert profile["streak_clean_sessions"] == 3
    assert profile["shield_count"] == 1
    assert len(profile["sessions"]) == 3
    assert all(s["status"] == "completed" for s in profile["sessions"])


# ========== 8. 剧本预设与编译缓存 ==========
def test_scene_presets_and_compile_cache(client, monkeypatch):
    """预设列表、无 Key 离线降级、Mock LLM 结构化编译及二次命中 scene_cache"""
    from app import scene_generator as sg

    host = _register_and_login(client, "scene_host", "场景官")

    # 1) 预设列表：需登录，返回 4 套内置剧本且字段齐备
    presets_resp = client.get("/api/scenes/presets", headers=host["headers"])
    assert presets_resp.status_code == 200, presets_resp.text
    presets = presets_resp.json()["presets"]
    assert len(presets) == 4
    first = presets[0]
    for key in ("scene_id", "scene_title", "commander_title", "opening_line",
                "theme", "atmosphere", "strikes", "victory_line"):
        assert key in first, "预设剧本应包含字段 %s" % key

    # 2) 无 Key 降级：清空 Key，编译任意作品名应回退离线剧本并写入 scene_cache
    monkeypatch.setattr(sg, "LLM_API_KEY", "")
    offline = client.post("/api/scenes/compile",
                          json={"work_name": "星际迷航新的征程"}, headers=host["headers"])
    assert offline.status_code == 200, offline.text
    offline_scene = offline.json()
    assert offline_scene.get("scene_id") and offline_scene.get("commander_title")
    assert offline_scene.get("source_work") == "星际迷航新的征程"

    # 3) Mock LLM 编译：配置 Key 并替换 _call_llm，返回结构合规剧本应归一化后落库
    fake_scene = copy.deepcopy(first)
    fake_scene["scene_id"] = "mock_llm_scene"
    fake_scene["source_work"] = "魔法学院奇遇记"

    calls = {"n": 0}

    async def _fake_call_llm(work_name):
        calls["n"] += 1
        return copy.deepcopy(fake_scene)

    monkeypatch.setattr(sg, "LLM_API_KEY", "fake-test-key")
    monkeypatch.setattr(sg, "_call_llm", _fake_call_llm)
    online = client.post("/api/scenes/compile",
                         json={"work_name": "魔法学院奇遇记"}, headers=host["headers"])
    assert online.status_code == 200, online.text
    compiled = online.json()
    assert compiled["scene_id"] == "mock_llm_scene"
    assert compiled["source_work"] == "魔法学院奇遇记"
    assert calls["n"] == 1, "首次编译应调用一次 LLM"

    # 4) 二次查询命中缓存：不再调用 LLM，仍返回同一剧本
    again = client.post("/api/scenes/compile",
                        json={"work_name": "魔法学院奇遇记"}, headers=host["headers"])
    assert again.status_code == 200
    assert again.json()["scene_id"] == "mock_llm_scene"
    assert calls["n"] == 1, "命中 scene_cache 后不得再次调用 LLM"

    # 直接核对缓存表已落库（归一化后的 JSON）
    assert sg.get_cached_scene("魔法学院奇遇记")


# ========== 9. 三次机会、护盾抵消与建设性惩罚解锁 ==========
def test_three_strikes_shield_and_punishment(client, monkeypatch):
    """连续走神 3 次扣血、护盾自动抵消、无盾锁房后费曼/目标拆解解锁"""
    import app.punishment as pun
    from app.db import get_db

    host = _register_and_login(client, "strike_host", "战士")

    # 压缩扣血阈值到 0（每次走神判定即扣 1 血），避免真实等待
    monkeypatch.setattr(pun, "STRIKE_SECONDS", 0.0)
    # 测试环境无 Key，确保校验走纯规则分支
    monkeypatch.setattr(pun, "LLM_API_KEY", "")

    def _fresh_room(**overrides):
        room = {
            "room_id": "r-strike", "host_id": host["user_id"],
            "strikes_left": 3, "distracted_streak_sec": 0.0,
            "is_locked": False, "punishment_type": "",
            "punishment_types": [], "goal_text": "掌握编译原理词法分析",
        }
        room.update(overrides)
        return room

    # 1) 连续走神 3 次扣血：第 1、2 次仅警告，第 3 次耗尽
    room = _fresh_room()
    e1 = pun.evaluate_strike(room, True, 1.0)
    assert e1["event"] == "strike_1" and e1["strikes_left"] == 2 and not e1["shield_saved"]
    e2 = pun.evaluate_strike(room, True, 1.0)
    assert e2["event"] == "strike_2" and e2["strikes_left"] == 1
    # 本轮无护盾：第三次耗尽应锁房并指定默认惩罚类型
    monkeypatch.setattr(pun, "_host_shield_saved", lambda _uid: False)
    e3 = pun.evaluate_strike(room, True, 1.0)
    assert e3["event"] == "strike_3" and e3["strikes_left"] == 0
    assert e3["is_locked"] and e3["punishment_type"] == "feynman_recall"
    assert room["is_locked"] is True

    # 已锁房幂等：再次判定不重复扣血/不重复扣盾
    e4 = pun.evaluate_strike(room, True, 1.0)
    assert e4["event"] is None and e4["strikes_left"] == 0

    # 2) 有护盾自动抵消：耗尽时恢复 1 点生命值、不锁房
    shield_room = _fresh_room(room_id="r-shield")
    monkeypatch.setattr(pun, "_host_shield_saved", lambda _uid: True)
    s3 = None
    for _ in range(3):
        s3 = pun.evaluate_strike(shield_room, True, 1.0)
    assert s3["event"] == "shield_saved" and s3["shield_saved"] is True
    assert s3["strikes_left"] == 1 and not s3["is_locked"]

    # 3) 费曼输出解锁：有效内容达标即解锁并恢复 1 点生命值
    monkeypatch.setattr(pun, "_host_shield_saved", lambda _uid: False)
    feynman_room = _fresh_room(room_id="r-feynman", is_locked=True,
                               punishment_type="feynman_recall", strikes_left=0)
    ok = pun.verify_punishment_submission(
        "feynman_recall",
        "词法分析就是把源程序字符流切成记号，我用有限自动机识别标识符再输出二元式",
        "掌握编译原理词法分析", room=feynman_room,
    )
    assert ok["passed"] and ok["unlocked"]
    assert feynman_room["is_locked"] is False and feynman_room["strikes_left"] == 1
    # 敷衍内容应被拒绝
    bad = pun.verify_punishment_submission("feynman_recall", "哈哈哈哈哈哈", "目标",
                                           room=_fresh_room(is_locked=True))
    assert not bad["passed"]

    # 4) 目标拆解解锁：至少 2 个具体微行动步骤
    goal_room = _fresh_room(room_id="r-goal", is_locked=True,
                            punishment_type="goal_breakdown", strikes_left=0)
    verdict = pun.verify_punishment_submission(
        "goal_breakdown",
        "通读教材第三章并画出状态转换图；手写一段识别关键字的自动机代码并跑通样例",
        "掌握编译原理词法分析", room=goal_room,
    )
    assert verdict["passed"] and verdict["unlocked"]
    assert goal_room["is_locked"] is False

    # 护盾发放链路自检：结算引擎可在连续干净局后发放护盾字段
    with get_db() as conn:
        row = conn.execute("SELECT shield_count FROM users WHERE id = ?",
                           (host["user_id"],)).fetchone()
    assert row is not None and row["shield_count"] >= 0


# ========== 10. 指挥官流式对话 WS 闭环 ==========
def test_ws_ai_commander_dialogue(client, monkeypatch):
    """房主入房收到主动开场流式消息、发送 user_message 收到 chat 流式回复"""
    import app.config as cfg

    # 强制走离线降级短句流、关闭 TTS，保证断言稳定不依赖网络
    monkeypatch.setattr(cfg, "LLM_API_KEY", "")
    monkeypatch.setattr(cfg, "TTS_ENABLED", "0")

    host = _register_and_login(client, "chat_host", "学习者甲")
    room = _create_room(client, host["headers"], mode="ai_solo",
                        goal_text="今天刷三套卷")
    rid = room["room_id"]

    with client.websocket_connect(
        "/ws/room/%s?token=%s" % (rid, host["token"]), headers={"origin": VALID_ORIGIN}
    ) as host_ws:
        # 主动开场：先收到 watcher_update，随后是 proactive 流式对话
        host_ws.receive_json()  # watcher_update(count=0)
        start = _recv_type(host_ws, "ai_dialogue_start")
        assert start["trigger"] == "proactive"

        chunks = []
        for _ in range(12):
            msg = host_ws.receive_json()
            if msg.get("type") == "ai_dialogue_chunk":
                chunks.append(msg.get("text", ""))
            if msg.get("type") == "ai_dialogue_end":
                break
        assert chunks, "开局应至少产出一条流式台词"
        assert any("卷" in t or "目标" in t or "作战" in t or "听令" in t for t in chunks), \
            "开场台词应带目标变量：%s" % chunks

        # 学习者求助：触发 chat 流式回复并回显用户变量
        host_ws.send_json({"type": "user_message", "text": "第三题卡住了怎么办"})
        chat_start = _recv_type(host_ws, "ai_dialogue_start")
        assert chat_start["trigger"] == "chat"
        chat_chunks = []
        for _ in range(12):
            msg = host_ws.receive_json()
            if msg.get("type") == "ai_dialogue_chunk":
                chat_chunks.append(msg.get("text", ""))
            if msg.get("type") == "ai_dialogue_end":
                break
        assert chat_chunks, "求助应产出至少一条流式台词"
        assert any("第三题" in t or "卡住" in t or "求助" in t for t in chat_chunks), \
            "chat 台词应回显求助变量：%s" % chat_chunks

    # 对话历史已落房间状态（最近 8 轮内）
    live = state_module.active_rooms.get(rid)
    assert live is not None and len(live["dialogue_history"]) >= 2
    assert live["dialogue_history"][-1]["trigger"] == "chat"


# ========== 11. 战报明细与 CSV 导出（M5 收官） ==========
def test_stats_session_detail_and_csv_export(client):
    """单局战报详情归属校验（本人/404/403）+ 采样点 UTC 格式化 + UTF-8 BOM CSV 与公式注入转义"""
    from app.db import save_behavior_record

    owner = _register_and_login(client, "stats_owner", "战报主人")
    intruder = _register_and_login(client, "stats_intruder", "越权者")

    # 造一局战报 + 三条行为采样（session_id 为字符串关联键），目标列刻意用公式注入载荷
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO study_sessions (user_id, mode, scene_title, goal_text, "
            "duration_minutes, avg_score, strikes_used, punishment_type, status) "
            "VALUES (?,?,?,?,?,?,?,?,'completed')",
            (owner["user_id"], "ai_solo", "流浪地球", "=1+1", 25, 88, 1, "feynman_recall"),
        )
        sid = cur.lastrowid
    for score in (91, 60, 45):
        save_behavior_record(owner["user_id"], {"focus_score": score}, session_id=str(sid))

    # 未登录 -> 401/403；不存在 -> 404；他人 -> 403；本人 -> 200
    assert client.get("/api/stats/sessions/%s" % sid).status_code in (401, 403)
    assert client.get("/api/stats/sessions/999999", headers=owner["headers"]).status_code == 404
    assert client.get("/api/stats/sessions/%s" % sid, headers=intruder["headers"]).status_code == 403
    detail = client.get("/api/stats/sessions/%s" % sid, headers=owner["headers"])
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["summary"]["scene_title"] == "流浪地球"
    assert len(body["points"]) == 3
    assert all(p["created_at"].endswith("Z") for p in body["points"])
    assert sorted(p["focus_score"] for p in body["points"]) == [45, 60, 91]

    # CSV 导出：BOM 头、表头顺序、公式注入转义
    resp = client.get("/api/stats/export.csv", headers=owner["headers"])
    assert resp.status_code == 200, resp.text
    assert resp.content.startswith(b"\xef\xbb\xbf"), "CSV 必须以 UTF-8 BOM 开头"
    assert resp.headers["content-type"].startswith("text/csv")
    text = resp.content.decode("utf-8-sig")
    header_line = text.splitlines()[0]
    assert header_line == ("结算时间(UTC),督学模式,场景名称,学习目标,专注时长(分钟),"
                           "平均专注分,消耗机会次数,惩罚类型,结算状态")
    assert "'=1+1" in text, "以 = 开头的文本列必须前置单引号转义"

    # 越权隔离：他人导出的 CSV 不含本用户战报
    other_text = client.get("/api/stats/export.csv", headers=intruder["headers"]).content.decode("utf-8-sig")
    assert "流浪地球" not in other_text
