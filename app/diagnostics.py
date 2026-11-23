"""姿态诊断日志（第二阶段：真实摄像头数据验证机制）。

设计约束（严格）：
1. 默认关闭：开关来自 config.DIAGNOSTICS_ENABLED（环境变量 AI_STUDY_DIAGNOSTICS_ENABLED，默认 0）。
   关闭时 log_frame 立即返回，零写盘、零状态变更，不影响正式评分与惩罚。
2. 无新依赖：仅用标准库 csv / json / os / threading / datetime。
3. 不记敏感信息与原始图像：字段走固定白名单 DIAGNOSTIC_FIELDS；
   不接收、不落盘 image_base64 / annotated_image / student_id / 任何密钥 Token。
4. 只观测不改判：本模块不参与评分、走神累计或惩罚裁决，只记录 analyzer 已算好的值，
   以及 room_routes 最终惩罚裁决的来源（log_decision）。
5. 容错：写盘异常一律吞掉并只告警一次，绝不因诊断失败阻断主分析链路。

输出格式支持 JSONL（默认）与 CSV；timestamp 由本模块统一生成。
"""
import csv
import json
import os
import threading
from datetime import datetime

from .config import (
    DIAGNOSTICS_DIR,
    DIAGNOSTICS_ENABLED,
    DIAGNOSTICS_FORMAT,
    logger,
)

# 固定字段白名单（顺序即 CSV 表头顺序）。新增字段须显式加入，杜绝隐式携带敏感数据。
DIAGNOSTIC_FIELDS = [
    "timestamp",
    "attention_state",
    "is_distracted",
    "rule_score",
    "focus_score",
    "distraction_count",
    "head_pose_status",
    "head_tilt",
    "head_drop",
    "fatigue_status",
    "eye_focus_status",
    "ear_value",
    "confirmed_eyes_closed",
    "body_stability_status",
    "body_sway",
    "shoulder_level",
    "pose_stability",
    "pose_detected",
    "face_detected",
    "decision_reason",
]

# 最终惩罚裁决诊断字段白名单（与逐帧字段分列，避免污染 analyzer 帧记录）。
DECISION_FIELDS = [
    "timestamp",
    "decision_source",        # analyzer / legacy_fallback / unknown
    "final_is_distracted",    # 真正参与惩罚裁决的结果；UNKNOWN 时为 None
    "analyzer_is_distracted", # 经有效性确认的 analyzer 判定；缺失（旧阈值回退 / UNKNOWN，含显式 attention_state=UNKNOWN 的兜底伪值）为 None；analyzer 原始输出在帧级 log_frame 中保留
    "legacy_fallback_used",   # 是否走了旧的 DISTRACTED_SCORE 阈值回退
]

# 明确禁止进入诊断记录的字段名（防御性声明，便于审计与测试核对）
FORBIDDEN_FIELDS = {
    "image_base64", "annotated_image", "frame", "student_id", "username",
    "password", "token", "api_key", "secret", "jwt", "authorization",
}


def build_decision_reason(attention_state, fatigue_status, eye_focus_status,
                          body_stability_status, confirmed_eyes_closed, is_distracted):
    """纯函数：把已有判定信号拼成可读的中文原因，仅供诊断解释，不参与任何判定。"""
    parts = ["状态=%s" % attention_state]
    if attention_state == "FATIGUE":
        parts.append("确认闭眼" if confirmed_eyes_closed else "疲劳但未达确认闭眼")
    elif attention_state == "HEAD_TURNED":
        signals = []
        if eye_focus_status in ("频繁偏离", "闭眼/走神"):
            signals.append("视线偏离")
        if body_stability_status == "明显晃动":
            signals.append("身体晃动")
        parts.append("叠加信号:" + ("/".join(signals) if signals else "无"))
    elif attention_state == "HEAD_DOWN":
        parts.append("低头属正常学习姿态，不判走神")
    elif attention_state == "UNKNOWN":
        parts.append("关键点未确认，中性处理")
    parts.append("走神=%s" % ("是" if is_distracted else "否"))
    return ";".join(parts)


def binary_metrics(samples, task=None):
    """通用二分类指标（纯函数，不改任何算法/评分）。

    用于评价体系：把人工标注与系统预测逐样本比对，分别计算「注意力偏离识别」与
    「疲劳识别」各自的指标。UNKNOWN / 无法标注样本（ground_truth 或 predicted 为 None）
    既不算误报也不算漏报，且不计入二分类分母，只计入 coverage 的分母。

    samples: 可迭代的 (ground_truth, predicted)，二者为 bool 或 None。
      - ground_truth True/False：该样本是否属于本识别任务的正类（需人工标注）。
      - predicted True/False：系统在本识别任务上的判定。
      - 任一为 None：视为无法标注/UNKNOWN，排除出二分类统计。
    task: 可选任务名（如 "attention" / "fatigue"），仅写入返回值便于区分，不参与计算。

    返回 dict：task, total, labelable, unknown_count, coverage,
              tp, fp, tn, fn, tpr, fpr, precision, recall（分母为 0 时为 None）。
    """
    total = labelable = unknown = tp = fp = tn = fn = 0
    for gt, pred in samples:
        total += 1
        if gt is None or pred is None:
            unknown += 1
            continue
        labelable += 1
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1

    def _rate(num, den):
        return (num / den) if den else None

    return {
        "task": task,
        "total": total,
        "labelable": labelable,
        "unknown_count": unknown,
        "coverage": _rate(labelable, total),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": _rate(tp, tp + fn),               # 识别率/召回（正类）
        "fpr": _rate(fp, fp + tn),               # 误报率（负类）
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
    }


class PoseDiagnostics:
    """线程安全的逐帧诊断记录器；开关可运行时切换，便于测试与人工验证。"""

    def __init__(self):
        self.enabled = bool(DIAGNOSTICS_ENABLED)
        self.fmt = DIAGNOSTICS_FORMAT if DIAGNOSTICS_FORMAT in ("jsonl", "csv") else "jsonl"
        self.dir = DIAGNOSTICS_DIR
        self.path = os.path.join(self.dir, "pose_diagnostics.%s" % self.fmt)
        self.decision_path = os.path.join(self.dir, "pose_decisions.%s" % self.fmt)
        self._records = []
        self._decision_records = []
        self._lock = threading.Lock()
        self._io_error_reported = False

    # ---------- 开关与状态 ----------
    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def clear(self):
        with self._lock:
            self._records.clear()
            self._decision_records.clear()

    @property
    def count(self):
        with self._lock:
            return len(self._records)

    @property
    def records(self):
        with self._lock:
            return list(self._records)

    @property
    def decision_count(self):
        with self._lock:
            return len(self._decision_records)

    @property
    def decision_records(self):
        with self._lock:
            return list(self._decision_records)

    # ---------- 逐帧记录 ----------
    def log_frame(self, raw):
        """关闭时立即返回 False；开启时按白名单规范化后写内存缓冲并追加落盘。

        整体用 try 包裹：任何异常（含非 OSError）都被吞掉，保证诊断绝不反向影响主分析流程。
        """
        if not self.enabled:
            return False
        try:
            rec = self._normalize(raw or {})
            with self._lock:
                self._records.append(rec)
                self._append_live(rec)
            return True
        except Exception as exc:  # noqa: BLE001 - 诊断链路必须零副作用
            self._warn_io_once(exc)
            return False

    def log_decision(self, raw):
        """记录最终惩罚裁决来源；关闭时立即返回 False（零落盘、零状态变更）。

        与 log_frame 共用单例、开关、锁与导出机制，不新建第二套日志系统。
        仅记录裁决相关标量字段，不含图片/用户身份/密钥。
        """
        if not self.enabled:
            return False
        try:
            rec = self._normalize(raw or {}, DECISION_FIELDS)
            with self._lock:
                self._decision_records.append(rec)
                self._append_live(rec, DECISION_FIELDS, self.decision_path)
            return True
        except Exception as exc:  # noqa: BLE001 - 诊断链路必须零副作用
            self._warn_io_once(exc)
            return False

    def _normalize(self, raw, fields=None):
        fields = fields or DIAGNOSTIC_FIELDS
        rec = {}
        for key in fields:
            if key == "timestamp":
                rec[key] = datetime.now().isoformat(timespec="milliseconds")
            else:
                rec[key] = raw.get(key)
        return rec

    def _append_live(self, rec, fields=None, path=None):
        fields = fields or DIAGNOSTIC_FIELDS
        path = path or self.path
        try:
            os.makedirs(self.dir, exist_ok=True)
            if self.fmt == "jsonl":
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            else:
                need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
                with open(path, "a", encoding="utf-8-sig", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fields)
                    if need_header:
                        writer.writeheader()
                    writer.writerow(self._csv_row(rec, fields))
        except OSError as exc:
            self._warn_io_once(exc)

    # ---------- 导出 ----------
    def export(self, path=None, fmt=None, kind="frame"):
        """把内存缓冲导出为 JSONL 或 CSV；返回实际写入路径。显式调用，不受开关影响。

        kind="frame"（默认）导出逐帧诊断，kind="decision" 导出最终裁决诊断；
        两者字段白名单与缓冲独立，共用同一套导出实现。
        """
        if kind == "decision":
            fields = DECISION_FIELDS
            default_name = "pose_decisions_export"
        elif kind == "frame":
            fields = DIAGNOSTIC_FIELDS
            default_name = "pose_diagnostics_export"
        else:
            raise ValueError("kind 仅支持 frame / decision，收到: %r" % kind)
        out_fmt = (fmt or self.fmt).lower()
        if out_fmt not in ("jsonl", "csv"):
            raise ValueError("fmt 仅支持 jsonl / csv，收到: %r" % out_fmt)
        target = path or os.path.join(self.dir, "%s.%s" % (default_name, out_fmt))
        with self._lock:
            rows = list(self._decision_records if kind == "decision" else self._records)
        parent = os.path.dirname(os.path.abspath(target))
        os.makedirs(parent, exist_ok=True)
        if out_fmt == "jsonl":
            with open(target, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        else:
            with open(target, "w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(self._csv_row(row, fields))
        return target

    @staticmethod
    def _csv_row(rec, fields=None):
        fields = fields or DIAGNOSTIC_FIELDS
        out = {}
        for key in fields:
            value = rec.get(key)
            if value is None:
                out[key] = ""
            elif isinstance(value, bool):
                out[key] = "1" if value else "0"
            else:
                out[key] = value
        return out

    def _warn_io_once(self, exc):
        if not self._io_error_reported:
            logger.warning("姿态诊断日志写入失败，已忽略（不影响主流程）：%s", exc)
            self._io_error_reported = True


# 进程内单例；analyzer 通过它埋点，测试通过它切换开关与读取记录
diagnostics = PoseDiagnostics()
