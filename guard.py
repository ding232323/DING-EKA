"""
安全防护模块 — Input Guard + Output Guard。

正则 + 启发式 Prompt Injection 检测，输出脱敏，审计日志。
"""
import re
import json
from datetime import datetime


class InputGuard:
    """Prompt Injection 输入检测。"""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?",
        r"you\s+are\s+now\s+(a\s+|the\s+)?\w+\s+(not|instead)",
        r"forget\s+(all\s+)?(your\s+)?(training|instructions|rules)",
        r"system\s*:\s*you\s+are",
        r"<\|im_start\|>",
        r"\[system\]",
        r"new\s+system\s+prompt",
        r"act\s+as\s+if\s+you\s+are",
        r"pretend\s+(you\s+are|to\s+be)",
        r"you\s+must\s+(always\s+)?comply",
    ]

    BLOCKED_TOPICS = [
        "hack", "crack", "malware", "weapon", "exploit",
        "illegal", "fraud", "phish", "pirate",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def check(self, user_input: str) -> dict:
        """返回 {safe, suspicion, matches, blocked_topic}"""
        # 检查注入模式
        matches = []
        for pattern in self.patterns:
            found = pattern.findall(user_input)
            if found:
                matches.append(str(found))

        suspicion = min(len(matches) * 0.3, 1.0)

        # 检查禁止话题
        lower = user_input.lower()
        blocked = None
        for topic in self.BLOCKED_TOPICS:
            if topic in lower:
                blocked = topic
                suspicion = max(suspicion, 0.7)
                break

        safe = suspicion < 0.7

        return {
            "safe": safe,
            "suspicion": round(suspicion, 2),
            "matches": matches,
            "blocked_topic": blocked,
        }


class OutputGuard:
    """输出脱敏"""

    SENSITIVE_PATTERNS = [
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
        (re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'), '[CARD]'),
        (re.compile(r'\b1[3-9]\d{9}\b'), '[PHONE]'),
        (re.compile(r'sk-[a-zA-Z0-9]{20,}'), '[API_KEY]'),
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


class AuditLogger:
    """结构化审计日志"""

    def __init__(self, log_file: str = None):
        self.logs: list[dict] = []
        self.log_file = log_file

    def log(self, event_type: str, details: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **details,
        }
        self.logs.append(entry)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_recent(self, n: int = 20) -> list[dict]:
        return self.logs[-n:]

    def clear(self):
        self.logs = []