"""
AI 剧本编译器（M2-A）

职责：
1. 内置 4 套经过打磨的离线 SceneConfig 剧本，严格符合 DEVELOPMENT_PLAN.md 第三章契约；
2. compile_scene 先读 scene_cache 缓存，未命中且配置了 LLM_API_KEY 时异步调用兼容
   OpenAI 的 /chat/completions 生成剧本，强制白名单归一化后落库返回；未配置 Key 或
   调用失败时按关键词匹配最贴近的内置剧本，绝不抛错阻断学习流程。

设计要点：
1. 所有对外返回的剧本都经过 _normalize_scene 归一化，保证前端渲染字段 100% 齐备；
2. LLM 超时/异常/非法 JSON 一律降级为离线剧本，学习流程永不因剧本生成中断；
3. 剧本缓存命中后同样做一次归一化，兼容历史版本的脏缓存；
4. 所有注释使用中文，本模块不落盘任何报告文件。
"""
import json

import httpx

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, logger
from .db import get_cached_scene, save_cached_scene

# ===== LLM 调用参数（模块级常量，便于单测 monkeypatch 压缩）=====
LLM_TIMEOUT_SEC = 8.0            # 剧本生成整体超时（秒），超时立即回退离线剧本
LLM_CONNECT_TIMEOUT_SEC = 3.0    # 建连超时（秒）

# ===== 渲染枚举白名单（与第三章契约严格一致）=====
HUD_STYLES = ("scifi", "military", "academy", "cyber")
PARTICLE_PRESETS = ("stars", "embers", "rain", "matrix", "dust")
AMBIENT_SOUNDS = ("spaceship_hum", "radar_tick", "library_rain", "fireplace", "none")

# 归一化缺省值（LLM 输出非法枚举时逐项回落）
_DEFAULT_HUD_STYLE = "scifi"
_DEFAULT_PARTICLE = "stars"
_DEFAULT_AMBIENT = "none"
_DEFAULT_PRIMARY = "#00E5FF"
_DEFAULT_ALERT = "#FF2A55"
_DEFAULT_GRADIENT = "linear-gradient(135deg, #050B14 0%, #0D1F33 100%)"
_DEFAULT_VOICE = "zh-CN-YunxiNeural"


# ========== 4 套内置离线剧本（严格符合 SceneConfig 契约）=========
_BUILTIN_SCENES = {
    "wandering_earth_moss": {
        "scene_id": "wandering_earth_moss",
        "source_work": "流浪地球",
        "scene_title": "领航员空间站·中控室",
        "commander_title": "智能量子体 MOSS",
        "commander_persona": "冷静、理性、略带紧迫感的空间站人工智能指挥官，称呼用户为领航员。",
        "opening_line": "领航员，行星发动机已进入推进轨道。请汇报本航段学习目标，准备投入战斗。",
        "theme": {
            "primary_color": "#00E5FF",
            "alert_color": "#FF2A55",
            "bg_gradient": "linear-gradient(135deg, #050B14 0%, #0D1F33 100%)",
            "hud_style": "scifi",
        },
        "atmosphere": {
            "particle_preset": "stars",
            "ambient_sound": "spaceship_hum",
            "voice_id": "zh-CN-YunxiNeural",
        },
        "strikes": {
            "strike_1": "警告：检测到领航员注意力偏移，请立即回归学习岗位。",
            "strike_2": "严重警告：生命维持与专注指数持续下降，这是最后一次口头通告！",
            "strike_3": "任务失守：领航员已脱离战位，启动紧急战损汇报程序。",
        },
        "victory_line": "任务圆满完成，领航员的专注数据已载入空间站日志。",
    },
    "military_outpost": {
        "scene_id": "military_outpost",
        "source_work": "战地前线哨所",
        "scene_title": "前线阵地·临时指挥所",
        "commander_title": "前线指挥官",
        "commander_persona": "果敢、严厉、雷厉风行的前线指挥官，称呼用户为战士，把学习任务当作作战命令。",
        "opening_line": "战士，阵地已就位，敌情不等人。报告你的作战目标，立即进入战斗状态！",
        "theme": {
            "primary_color": "#C8A24B",
            "alert_color": "#E23B2E",
            "bg_gradient": "linear-gradient(135deg, #1A1812 0%, #3A3120 100%)",
            "hud_style": "military",
        },
        "atmosphere": {
            "particle_preset": "embers",
            "ambient_sound": "radar_tick",
            "voice_id": "zh-CN-YunjianNeural",
        },
        "strikes": {
            "strike_1": "第一次警告：战士离开观察位，立刻归位，敌情就是命令！",
            "strike_2": "第二次警告：注意力涣散就是战场违纪，这不是演习！",
            "strike_3": "阵地失守：全队陷入危险，立即启动战损汇报！",
        },
        "victory_line": "作战任务完成，阵地守住了，你是一名合格的战士。",
    },
    "hogwarts_library": {
        "scene_id": "hogwarts_library",
        "source_work": "霍格沃茨魔法图书馆",
        "scene_title": "霍格沃茨·禁书区阅览室",
        "commander_title": "图书馆管理员平斯夫人",
        "commander_persona": "威严而博学的魔法图书馆管理员，称呼用户为年轻巫师，强调知识与专注的纪律。",
        "opening_line": "年轻巫师，安静。翻开你的课本，告诉我今天要研习哪一门魔法。",
        "theme": {
            "primary_color": "#B8912F",
            "alert_color": "#8E2B3C",
            "bg_gradient": "linear-gradient(135deg, #1B1526 0%, #2E2440 100%)",
            "hud_style": "academy",
        },
        "atmosphere": {
            "particle_preset": "dust",
            "ambient_sound": "library_rain",
            "voice_id": "zh-CN-XiaoxiaoNeural",
        },
        "strikes": {
            "strike_1": "提醒一次：目光离开了书页，知识可不会自己钻进脑子。",
            "strike_2": "第二次提醒：魔咒需要专心吟诵，请立刻收回心神！",
            "strike_3": "禁书区警报：分心已触犯校规，提交你的学习心得才能继续。",
        },
        "victory_line": "学业精进，这一章的魔法你已经牢牢掌握了。",
    },
    "cyber_matrix": {
        "scene_id": "cyber_matrix",
        "source_work": "赛博数据中枢",
        "scene_title": "数据中枢·核心矩阵",
        "commander_title": "引导程序 Architect",
        "commander_persona": "冷静精确的赛博空间引导程序，称呼用户为操作员，用数据流与入侵隐喻驱动学习。",
        "opening_line": "操作员，连接已建立，数据流正在加载。请输入本轮任务指令，进入矩阵核心。",
        "theme": {
            "primary_color": "#39FF14",
            "alert_color": "#FF2A6D",
            "bg_gradient": "linear-gradient(135deg, #02110A 0%, #0A2418 100%)",
            "hud_style": "cyber",
        },
        "atmosphere": {
            "particle_preset": "matrix",
            "ambient_sound": "none",
            "voice_id": "zh-CN-YunyangNeural",
        },
        "strikes": {
            "strike_1": "警告：操作员注意力信号衰减，请重新锁定目标节点。",
            "strike_2": "严重警告：数据流异常波动，这是最后一次同步提醒！",
            "strike_3": "连接中断：会话已被防火墙锁定，提交战果报告以重建链路。",
        },
        "victory_line": "任务完成，数据已写入核心矩阵，操作员表现优秀。",
    },
}

# 关键词 → 内置剧本 id：按匹配优先级排列，用于无 Key 时平滑匹配最贴近的离线剧本
_KEYWORD_ROUTES = (
    ("military_outpost", ("亮剑", "战地", "军事", "部队", "哨所", "战争", "士兵", "军营", "突击")),
    ("hogwarts_library", ("霍格沃茨", "哈利", "魔法", "巫师", "学院", "伏地魔", "gryffindor")),
    ("cyber_matrix", ("赛博", "黑客", "矩阵", "黑客帝国", "元宇宙", "代码", "网络", "cyber", "matrix")),
    ("wandering_earth_moss", ("流浪地球", "三体", "星际", "科幻", "太空", "宇宙", "空间站", "领航员")),
)
_DEFAULT_SCENE_ID = "wandering_earth_moss"


def _normalize_scene(raw: dict) -> dict:
    """把任意来源的剧本字典归一化为契约齐备的 SceneConfig：枚举白名单校验 + 缺失字段回落"""
    fallback = _BUILTIN_SCENES[_DEFAULT_SCENE_ID]
    raw = raw if isinstance(raw, dict) else {}

    def _text(value, default):
        """取非空字符串，否则回落默认值"""
        return value.strip() if isinstance(value, str) and value.strip() else default

    theme = raw.get("theme") if isinstance(raw.get("theme"), dict) else {}
    atmosphere = raw.get("atmosphere") if isinstance(raw.get("atmosphere"), dict) else {}
    strikes = raw.get("strikes") if isinstance(raw.get("strikes"), dict) else {}

    hud = theme.get("hud_style")
    particle = atmosphere.get("particle_preset")
    ambient = atmosphere.get("ambient_sound")

    return {
        "scene_id": _text(raw.get("scene_id"), fallback["scene_id"]),
        "source_work": _text(raw.get("source_work"), ""),
        "scene_title": _text(raw.get("scene_title"), fallback["scene_title"]),
        "commander_title": _text(raw.get("commander_title"), fallback["commander_title"]),
        "commander_persona": _text(raw.get("commander_persona"), fallback["commander_persona"]),
        "opening_line": _text(raw.get("opening_line"), fallback["opening_line"]),
        "theme": {
            "primary_color": _text(theme.get("primary_color"), _DEFAULT_PRIMARY),
            "alert_color": _text(theme.get("alert_color"), _DEFAULT_ALERT),
            "bg_gradient": _text(theme.get("bg_gradient"), _DEFAULT_GRADIENT),
            # 枚举白名单：非法取值一律回落到默认项，保证前端可渲染
            "hud_style": hud if hud in HUD_STYLES else _DEFAULT_HUD_STYLE,
        },
        "atmosphere": {
            "particle_preset": particle if particle in PARTICLE_PRESETS else _DEFAULT_PARTICLE,
            "ambient_sound": ambient if ambient in AMBIENT_SOUNDS else _DEFAULT_AMBIENT,
            "voice_id": _text(atmosphere.get("voice_id"), _DEFAULT_VOICE),
        },
        "strikes": {
            "strike_1": _text(strikes.get("strike_1"), fallback["strikes"]["strike_1"]),
            "strike_2": _text(strikes.get("strike_2"), fallback["strikes"]["strike_2"]),
            "strike_3": _text(strikes.get("strike_3"), fallback["strikes"]["strike_3"]),
        },
        "victory_line": _text(raw.get("victory_line"), fallback["victory_line"]),
    }


def list_presets() -> list:
    """返回 4 套内置剧本的归一化列表，供 GET /api/scenes/presets 直接消费"""
    return [_normalize_scene(scene) for scene in _BUILTIN_SCENES.values()]


def _match_builtin(work_name: str) -> dict:
    """按关键词匹配最贴近的内置剧本，无命中回退默认剧本；source_work 回填用户输入"""
    lowered = work_name.lower()
    scene_id = _DEFAULT_SCENE_ID
    for candidate_id, keywords in _KEYWORD_ROUTES:
        if any(keyword.lower() in lowered for keyword in keywords):
            scene_id = candidate_id
            break
    scene = _normalize_scene(_BUILTIN_SCENES[scene_id])
    scene["source_work"] = work_name
    return scene


def _extract_json(content: str):
    """从 LLM 回复中稳健提取 JSON 对象：兼容 ```json 代码围栏与前后杂文本"""
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```"):
        # 去掉首尾围栏行（```json / ```）
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


async def _call_llm(work_name: str):
    """异步调用兼容 OpenAI 的 /chat/completions 生成剧本；任何异常向上抛，由 compile_scene 兜底"""
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    system_prompt = (
        "你是游戏化督学场景编译器。根据用户给出的作品名称，生成一个沉浸式中学生学习场景的 SceneConfig JSON。"
        "必须严格输出如下 JSON 结构（不要输出解释文字、不要 markdown 围栏）："
        '{"scene_id":"英文小写下划线标识","source_work":"作品名","scene_title":"场景标题",'
        '"commander_title":"指挥官称谓","commander_persona":"指挥官人设一句话","opening_line":"开场台词",'
        '"theme":{"primary_color":"#RRGGBB","alert_color":"#RRGGBB",'
        '"bg_gradient":"linear-gradient(...)","hud_style":"枚举"},'
        '"atmosphere":{"particle_preset":"枚举","ambient_sound":"枚举","voice_id":"音色标识"},'
        '"strikes":{"strike_1":"第一次警告","strike_2":"第二次警告","strike_3":"第三次警告"},'
        '"victory_line":"通关台词"}。'
        f"其中 hud_style 只能取 {list(HUD_STYLES)} 之一；"
        f"particle_preset 只能取 {list(PARTICLE_PRESETS)} 之一；"
        f"ambient_sound 只能取 {list(AMBIENT_SOUNDS)} 之一。"
        "全部文案使用中文且贴合该作品氛围，称呼用户为学习者/战士/领航员等角色内称谓。"
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请为作品《{work_name}》编译督学场景。"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(LLM_TIMEOUT_SEC, connect=LLM_CONNECT_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if parsed is None:
        raise ValueError("LLM 返回的剧本不是合法 JSON 对象")
    return parsed


async def compile_scene(work_name: str) -> dict:
    """编译剧本：缓存命中 → 在线生成 → 关键词匹配离线回退，全程不抛错阻断"""
    name = str(work_name or "").strip()
    # 1) 先查缓存：命中后归一化一次以兼容历史脏缓存
    try:
        cached = get_cached_scene(name)
        if cached:
            obj = json.loads(cached)
            if isinstance(obj, dict):
                logger.info("剧本缓存命中：%s", name)
                return _normalize_scene(obj)
    except (ValueError, TypeError) as exc:
        logger.warning("剧本缓存解析失败，忽略并重新编译：%s，原因：%s", name, exc)
    except Exception as exc:
        logger.warning("读取剧本缓存异常，忽略并重新编译：%s，原因：%s", name, exc)

    # 2) 未命中且配置了 Key：在线生成并落库
    if LLM_API_KEY:
        try:
            raw = await _call_llm(name)
            scene = _normalize_scene(raw)
            scene["source_work"] = name
            save_cached_scene(name, json.dumps(scene, ensure_ascii=False))
            logger.info("剧本在线生成成功并已落库：%s", name)
            return scene
        except Exception as exc:
            logger.warning("剧本在线生成失败，回退离线剧本：%s，原因：%s", name, exc)

    # 3) 未配置 Key 或在线失败：匹配最贴近的离线剧本并落库（二次请求可直接命中缓存）
    scene = _match_builtin(name)
    try:
        save_cached_scene(name, json.dumps(scene, ensure_ascii=False))
    except Exception as exc:
        # 落库失败不影响本次返回，仅记录告警（绝不阻断学习流程）
        logger.warning("离线剧本落库失败，仍返回剧本：%s，原因：%s", name, exc)
    return scene
