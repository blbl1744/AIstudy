"""
实时流式对话指挥官（M2-B）

职责：以异步生成器 stream_commander_dialogue 按三种触发场景产出指挥官对话短句流：
1. proactive  开局领命：欢迎学习者、复述并拆解本轮学习目标；
2. distraction 走神纠偏：结合 goal_text 与剩余 strikes_left 给出具体可执行的归位动作；
3. chat        卡壳求助/互动：结合本轮目标给出简洁可操作的建议。

设计要点：
1. 配置了 LLM_API_KEY 时以 httpx.AsyncClient 流式请求兼容 OpenAI 的 /chat/completions
   （stream=True），按句末标点（。！？!?）切分逐句 yield ai_dialogue_chunk；
2. 未配置 Key 或请求异常时，平滑降级为带 commander_title / goal_text 变量的定制短句流，
   学习流程永不因对话服务不可用而中断；
3. TTS_ENABLED=="1" 且已安装 edge_tts 时，为每句额外生成 base64 音频并
   yield ai_audio_chunk；缺依赖或任何异常一律静默跳过，不影响文本流；
4. 每轮结束后把交互追加到 room["dialogue_history"]，仅保留最近 8 轮；
5. LLM 参数与阈值均为模块级常量，便于 pytest monkeypatch 压缩时延；
6. 所有注释使用中文，本模块不落盘任何报告文件。
"""
import base64
import json

import httpx

from . import config
from .config import logger

# 句末标点：流式文本按这些标点切分为短句
_SENTENCE_END = set("。！？!?")
# 对话历史保留轮数
DIALOGUE_HISTORY_LIMIT = 8
# LLM 调用超时（模块级便于测试 monkeypatch）
LLM_TIMEOUT_SEC = 8.0         # 整体超时（秒）
LLM_CONNECT_TIMEOUT_SEC = 3.0  # 建连超时（秒）

# 三种触发的系统人设要点（供 LLM system 消息使用，同时用于 trigger 白名单校验）
_TRIGGER_SYSTEM = {
    "proactive": (
        "这是开局领命场景：以指挥官口吻欢迎学习者、复述并拆解本轮学习目标为 2-3 个具体微行动，"
        "语气坚定、有代入感，控制在 3 句以内。"
    ),
    "distraction": (
        "这是走神纠偏场景：学习者分心了，用指挥官口吻给出一条具体、可立即执行的纠偏动作，"
        "并提醒剩余机会次数，语气严肃但不失鼓励，控制在 2 句以内。"
    ),
    "chat": (
        "这是学习者卡壳求助或互动场景：以指挥官口吻结合本轮目标给出简洁、可操作的建议，"
        "控制在 3 句以内。"
    ),
}


def _scene_fields(room) -> dict:
    """从房间提取对话所需变量：指挥官称谓、学习目标、剩余机会与音色"""
    room = room if isinstance(room, dict) else {}
    scene = room.get("scene_config") if isinstance(room.get("scene_config"), dict) else {}
    atmosphere = scene.get("atmosphere") if isinstance(scene.get("atmosphere"), dict) else {}
    return {
        "commander_title": scene.get("commander_title") or "督学指挥官",
        "goal_text": room.get("goal_text") or "本轮学习任务",
        "strikes_left": room.get("strikes_left", 3),
        "voice_id": atmosphere.get("voice_id") or "zh-CN-YunxiNeural",
    }


def _build_user_prompt(trigger: str, fields: dict, user_input: str) -> str:
    """按触发场景构造 user 消息，注入目标/剩余机会等变量"""
    goal = fields["goal_text"]
    if trigger == "proactive":
        return f"请以你的指挥官身份开局领命，本轮学习目标：{goal}。"
    if trigger == "distraction":
        return (
            f"学习者刚才分心了（剩余机会 {fields['strikes_left']} 次），"
            f"本轮学习目标：{goal}。请给出一条具体纠偏动作。"
        )
    return f"学习者求助：{(user_input or '').strip()}（本轮目标：{goal}）。请简洁作答。"


def _fallback_lines(trigger: str, fields: dict, user_input: str) -> list:
    """离线降级短句流：带 commander_title / goal_text（及剩余机会）变量的定制文案"""
    title = fields["commander_title"]
    goal = fields["goal_text"]
    left = fields["strikes_left"]
    if trigger == "proactive":
        return [
            f"{title}报到！本轮作战目标已锁定：{goal}。",
            f"听令：把「{goal}」拆成三段——通读要点、动手演练、复盘错题，逐段攻克。",
            "系统已就位，现在开始，我在旁全程盯着你。",
        ]
    if trigger == "distraction":
        return [
            f"{title}警告：检测到注意力偏移，当前剩余机会 {left} 次。",
            f"立刻坐直、视线回到屏幕，把「{goal}」推进一小步——哪怕只做一道题。",
            "归位后我会解除警报，行动！",
        ]
    text = (user_input or "").strip()
    hint = (
        f"关于「{text}」：别急着求快，先回到「{goal}」的下一步，把卡住的点写下来逐个击破。"
        if text else f"坚持住，围绕「{goal}」先完成最小的一步，再逐步展开。"
    )
    return [
        f"{title}收到你的求助。",
        hint,
        "需要我更具体一点，就再描述一下你卡在哪里。",
    ]


def _drain_sentences(buffer: str):
    """按句末标点切出完整短句，返回 (完整句列表, 未成句的残留片段)"""
    sentences = []
    start = 0
    for idx, ch in enumerate(buffer):
        if ch in _SENTENCE_END:
            segment = buffer[start:idx + 1].strip()
            if segment:
                sentences.append(segment)
            start = idx + 1
    return sentences, buffer[start:]


async def _stream_llm(trigger: str, user_prompt: str, fields: dict):
    """流式请求兼容 OpenAI 的 /chat/completions，逐块 yield 文本增量（异常向上抛）"""
    url = (getattr(config, "LLM_BASE_URL", "") or "").rstrip("/") + "/chat/completions"
    system = _TRIGGER_SYSTEM.get(trigger, _TRIGGER_SYSTEM["chat"])
    system += f"\n你的称谓是「{fields['commander_title']}」。学习者本轮目标：{fields['goal_text']}。"
    payload = {
        "model": getattr(config, "LLM_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(LLM_TIMEOUT_SEC, connect=LLM_CONNECT_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                    piece = obj["choices"][0]["delta"].get("content", "") or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                if piece:
                    yield piece


async def _audio_b64(text: str, voice_id: str):
    """为单句生成 base64 音频；未安装 edge_tts 或任何异常返回 None（静默跳过）"""
    try:
        import edge_tts  # 可选依赖，缺失时直接跳过
        communicate = edge_tts.Communicate(text, voice_id)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk.get("data") or b"")
        if not buf:
            return None
        return base64.b64encode(bytes(buf)).decode("ascii")
    except Exception as exc:
        logger.debug("语音合成跳过：%s", exc)
        return None


def _append_history(room, trigger: str, user_input: str, utterances: list) -> None:
    """把本轮交互追加到房间对话历史并截断保留最近 8 轮"""
    if not isinstance(room, dict):
        return
    history = room.setdefault("dialogue_history", [])
    history.append({
        "trigger": trigger,
        "user_input": (user_input or "")[:200],
        "text": "".join(utterances),
    })
    if len(history) > DIALOGUE_HISTORY_LIMIT:
        del history[:-DIALOGUE_HISTORY_LIMIT]


async def stream_commander_dialogue(room, trigger: str, user_input: str = ""):
    """异步生成器：按 trigger 产出指挥官对话块（ai_dialogue_chunk + 可选 ai_audio_chunk）"""
    trigger = trigger if trigger in _TRIGGER_SYSTEM else "chat"
    fields = _scene_fields(room)
    tts_on = str(getattr(config, "TTS_ENABLED", "0")) == "1"
    utterances = []

    async def _say(sentence: str):
        """产出单句文本块（含可选语音块），并登记到本轮台词"""
        sentence = (sentence or "").strip()
        if not sentence:
            return
        utterances.append(sentence)
        yield {"type": "ai_dialogue_chunk", "text": sentence}
        if tts_on:
            audio = await _audio_b64(sentence, fields["voice_id"])
            if audio:
                yield {"type": "ai_audio_chunk", "audio_b64": audio}

    try:
        produced = False
        # 1) 配置了 Key：流式请求 LLM，按句末标点切分逐句产出
        if getattr(config, "LLM_API_KEY", ""):
            buffer = ""
            try:
                async for piece in _stream_llm(trigger, _build_user_prompt(trigger, fields, user_input), fields):
                    produced = True
                    buffer += piece
                    sentences, buffer = _drain_sentences(buffer)
                    for sentence in sentences:
                        async for out in _say(sentence):
                            yield out
            except Exception as exc:
                logger.warning("对话指挥官 LLM 流式请求失败，将降级短句流：%s", exc)
            # 流结束后残留的非空片段同样作为一句产出
            if buffer.strip():
                async for out in _say(buffer.strip()):
                    yield out
        # 2) 未配置 Key / 请求失败且一句未产出：平滑降级定制短句流
        if not produced or not utterances:
            for line in _fallback_lines(trigger, fields, user_input):
                async for out in _say(line):
                    yield out
    finally:
        # 3) 无论正常结束还是中途异常，都登记本轮交互（截断保留最近 8 轮）
        _append_history(room, trigger, user_input, utterances)
