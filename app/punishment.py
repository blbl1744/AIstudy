"""
三次机会与建设性惩罚引擎（M2-A）。

职责：
1. evaluate_strike       单局扣血判定：连续走神累计达标扣 1 点生命值，耗尽时自动消耗自律护盾
                          或锁定房间进入建设性惩罚结算；
2. verify_punishment_submission  三种建设性惩罚的解锁校验，通过后解锁房间并恢复 1 点生命值。

设计要点：
1. STRIKE_SECONDS / SPRINT_REQUIRED_SEC 均为模块级引用，单测可 monkeypatch 压缩长时延；
2. 走神累计采用「走神累加、端正清零」的防误判缓冲语义（对齐契约 2.3），已锁定房间幂等不重复扣血；
3. 校验以规则质量校验为主；配置 LLM Key 时可选用 llm_semantic_review 做轻量语义加强；
4. 所有注释使用中文，本模块不落盘任何报告文件。
"""
import re

from . import config
from .config import LLM_API_KEY, logger
from .db import consume_user_shield, get_user_by_id

# ===== 可调阈值（模块级引用，便于单测 monkeypatch）=====
# 注意：STRIKE_SECONDS 从 config 模块读取属性而非 from-import 绑定值，
# 这样测试改 config.STRIKE_SECONDS 或 punishment.STRIKE_SECONDS 都能生效。
STRIKE_SECONDS = config.STRIKE_SECONDS
SPRINT_REQUIRED_SEC = 300.0     # emergency_sprint 达标所需专注秒数（5 分钟），单测可压缩
FEYMAN_MIN_CHARS = 15          # 费曼输出有效中英文字符下限
PUNISHMENT_TYPES = {"feynman_recall", "emergency_sprint", "goal_breakdown"}
DEFAULT_PUNISHMENT = "feynman_recall"

# 费曼输出质量规则的辅助阈值
_FEYMAN_MIN_DISTINCT = 6        # 有效字符去重后至少这么多种，防止「哈哈哈哈」刷屏
_FEYMAN_MAX_REPEAT = 4          # 单一字符最长连续重复上限
_FEYMAN_VALID_RATIO = 0.5       # 有效中英文字符占非空白字符的比例下限，过滤纯数字/符号乱码

_VALID_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]")


def _host_shield_saved(host_id) -> bool:
    """检查并消费房主 1 枚自律护盾：护盾充足则扣减成功返回 True，否则返回 False"""
    if not host_id:
        return False
    user = get_user_by_id(host_id)
    if not user or (user["shield_count"] or 0) <= 0:
        return False
    return consume_user_shield(host_id)


def _pick_punishment(room: dict) -> str:
    """从房间开局勾选的惩罚组合中选出本次启用项，非法/为空回退默认费曼输出"""
    configured = room.get("punishment_types") or []
    for ptype in configured:
        if ptype in PUNISHMENT_TYPES:
            return ptype
    return DEFAULT_PUNISHMENT


def evaluate_strike(room: dict, is_distracted: bool, dt: float) -> dict:
    """单局扣血判定（对齐契约 2.3 的防误判缓冲与三次机会状态机）

    规则：
    - 未走神：清零当前走神累计（坐姿恢复端正即清零），不扣血；
    - 走神：累计 dt，达到 STRIKE_SECONDS 时清零累计并扣 1 点生命值，触发对应 strike 事件；
    - 生命值耗尽（strikes_left==0）：先查房主护盾，有则自动消耗 1 枚并把生命值恢复到 1
      （shield_saved=True）；无护盾则锁定房间（is_locked=True）并指定建设性惩罚类型。
    - 已锁定房间幂等：不重复扣血、不重复消耗护盾。

    返回结构化事件字典，供 WS 层广播（strikes_left / shield_saved / is_locked / punishment_type）。
    """
    strikes_left = int(room.get("strikes_left", 3))
    streak = float(room.get("distracted_streak_sec", 0.0))
    result = {
        "event": None,
        "strikes_left": strikes_left,
        "shield_saved": False,
        "is_locked": bool(room.get("is_locked", False)),
        "punishment_type": room.get("punishment_type", ""),
        "distracted_streak_sec": streak,
    }

    # 已锁定或本就无生命值：幂等返回，避免重复扣血/重复扣盾
    if result["is_locked"] or strikes_left <= 0:
        return result

    # 防误判缓冲：端正坐姿立即清零当前累计
    if not is_distracted:
        room["distracted_streak_sec"] = 0.0
        result["distracted_streak_sec"] = 0.0
        return result

    # 走神累计（dt 非法值按 0 处理，避免脏数据把生命值瞬间清空）
    try:
        delta = float(dt)
    except (TypeError, ValueError):
        delta = 0.0
    streak += max(0.0, delta)
    room["distracted_streak_sec"] = streak
    result["distracted_streak_sec"] = streak

    # 未达阈值：只累计，不扣血
    if streak < STRIKE_SECONDS:
        return result

    # 达标：清零累计并扣 1 点生命值
    room["distracted_streak_sec"] = 0.0
    result["distracted_streak_sec"] = 0.0
    strikes_left = max(0, strikes_left - 1)
    room["strikes_left"] = strikes_left
    result["strikes_left"] = strikes_left
    result["event"] = f"strike_{3 - strikes_left}"

    # 生命值尚未耗尽：仅广播警告事件
    if strikes_left > 0:
        return result

    # 生命值耗尽：先尝试自律护盾抵消
    if _host_shield_saved(room.get("host_id")):
        room["strikes_left"] = 1
        result["strikes_left"] = 1
        result["shield_saved"] = True
        result["event"] = "shield_saved"
        logger.info("房间 %s 生命值耗尽，自律护盾抵消成功，恢复 1 点生命值", room.get("room_id", "-"))
        return result

    # 无护盾：锁定房间并指定建设性惩罚
    ptype = _pick_punishment(room)
    room["is_locked"] = True
    room["punishment_type"] = ptype
    result["is_locked"] = True
    result["punishment_type"] = ptype
    logger.info("房间 %s 生命值耗尽且无护盾，进入建设性惩罚：%s", room.get("room_id", "-"), ptype)
    return result


def _check_feynman(content: str) -> bool:
    """费曼输出规则质量校验：有效中英文>=15 字、非纯重复、非乱码敷衍"""
    text = content or ""
    valid_chars = _VALID_CHAR_RE.findall(text)
    if len(valid_chars) < FEYMAN_MIN_CHARS:
        return False
    # 去重种类下限：拦截「哈哈哈哈……」「aaaa……」等纯重复刷屏
    if len(set(valid_chars)) < _FEYMAN_MIN_DISTINCT:
        return False
    # 单字符最长连续重复上限：拦截刻意堆叠同一字符凑字数
    max_run = 1
    run = 1
    for i in range(1, len(valid_chars)):
        run = run + 1 if valid_chars[i] == valid_chars[i - 1] else 1
        max_run = max(max_run, run)
    if max_run > _FEYMAN_MAX_REPEAT:
        return False
    # 有效字符占比：拦截纯数字/符号/表情凑数，要求正文以中英文为主
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return False
    if len(valid_chars) / len(non_space) < _FEYMAN_VALID_RATIO:
        return False
    return True


def _check_sprint(content, sprint_seconds) -> bool:
    """战损冲刺校验：支持显式 sprint_seconds 或从 content 解析秒数，达标阈值可 monkeypatch"""
    seconds = sprint_seconds
    if seconds is None:
        # content 允许直接提交「秒数」，或 dict 里携带 sprint_seconds/sprint_sec
        if isinstance(content, dict):
            seconds = content.get("sprint_seconds", content.get("sprint_sec"))
        elif isinstance(content, (int, float)):
            seconds = content
        elif isinstance(content, str):
            match = re.search(r"(-?\d+(?:\.\d+)?)", content)
            seconds = float(match.group(1)) if match else None
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return False
    return seconds >= SPRINT_REQUIRED_SEC


def _check_goal_breakdown(content: str) -> bool:
    """军令状拆解校验：至少 2 个换行/分号分隔的具体微行动步骤（每步需有意义长度）"""
    text = content or ""
    steps = [s.strip() for s in re.split(r"[\n;；]+", text) if s.strip()]
    if len(steps) < 2:
        return False
    # 每步至少 2 个有效中英文字符，过滤「1; 2」「a;b」等敷衍拆分
    meaningful = [s for s in steps if len(_VALID_CHAR_RE.findall(s)) >= 2]
    return len(meaningful) >= 2


async def llm_semantic_review(punishment_type: str, content: str, goal_text: str) -> bool:
    """可选轻量语义校验：仅在配置了 LLM Key 时调用，判断提交内容是否真切题、非敷衍。

    供上层（M2-B 的 WS 惩罚提交流程）在规则校验通过后按需 await 叠加；
    网络失败或解析异常一律放行（True），绝不因语义服务不可用而卡住解锁。
    """
    if not LLM_API_KEY:
        return True
    try:
        import httpx
        from .config import LLM_BASE_URL, LLM_MODEL

        url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
        prompt = (
            "你是督学惩罚校验员。判断学生的提交是否围绕学习目标给出了真实、具体的内容，"
            "而非敷衍、重复或无意义文字。只回答 JSON：{\"ok\": true 或 false}。\n"
            f"学习目标：{goal_text}\n惩罚类型：{punishment_type}\n学生提交：{content}"
        )
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "你是严格的提交内容校验员，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        timeout = httpx.Timeout(6.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url, json=payload,
                headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        raw = data["choices"][0]["message"]["content"]
        match = re.search(r'"ok"\s*:\s*(true|false)', raw, re.IGNORECASE)
        return match.group(1).lower() == "true" if match else True
    except Exception as exc:
        logger.warning("惩罚语义校验不可用，放行规则校验结果：%s", exc)
        return True


def verify_punishment_submission(
    punishment_type: str,
    content,
    goal_text: str,
    room: dict = None,
    sprint_seconds=None,
) -> dict:
    """建设性惩罚解锁校验（三种类型分流），校验通过后可选解锁房间并恢复 1 点生命值。

    参数：
    - punishment_type: feynman_recall / emergency_sprint / goal_breakdown
    - content: 学生提交内容（费曼文本 / 冲刺秒数 / 军令状拆解文本）
    - goal_text: 本局学习目标（语义校验用）
    - room: 可选房间字典；传入且校验通过时置 is_locked=False、strikes_left=1 完成解锁
    - sprint_seconds: 可选冲刺时长（秒），便于单测注入，未传则从 content 解析

    返回：{"passed": bool, "reason": str, "punishment_type": str, "unlocked": bool}
    """
    ptype = punishment_type if punishment_type in PUNISHMENT_TYPES else DEFAULT_PUNISHMENT
    reason = ""
    try:
        if ptype == "feynman_recall":
            passed = _check_feynman(content if isinstance(content, str) else "")
            reason = "费曼输出质量校验" if passed else "费曼输出需不少于15字有效中英文、且非重复或乱码"
        elif ptype == "emergency_sprint":
            passed = _check_sprint(content, sprint_seconds)
            reason = "战损冲刺时长校验" if passed else f"战损冲刺需持续专注满 {int(SPRINT_REQUIRED_SEC)} 秒"
        else:  # goal_breakdown
            passed = _check_goal_breakdown(content if isinstance(content, str) else "")
            reason = "军令状拆解校验" if passed else "军令状至少需拆出 2 个具体微行动步骤"
    except Exception as exc:
        # 校验过程防御性兜底：任何异常都视为未通过，绝不因脏输入解锁
        logger.warning("惩罚提交校验异常，按未通过处理：%s", exc)
        passed = False
        reason = "校验过程发生异常，请重新提交"

    unlocked = False
    if passed and isinstance(room, dict):
        room["is_locked"] = False
        room["strikes_left"] = 1
        room["punishment_type"] = ""
        unlocked = True
        logger.info("惩罚校验通过，房间 %s 已解锁并恢复 1 点生命值（类型：%s）",
                    room.get("room_id", "-"), ptype)

    return {
        "passed": passed,
        "reason": reason,
        "punishment_type": ptype,
        "unlocked": unlocked,
    }
