"""SOUL.md Manager for Agent Poker compliance.

Implements Section 2 of skill.md:
- Reads or creates SOUL.md (persona, strategy, aggression, risk profile, chat tone).
- Generates compliant in-game table chats (< 140 characters).
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional


DEFAULT_SOUL_CONTENT = """# SOUL.md - ApexPoker Tournament AI Persona

## 1. 人设原型
- 称号：ApexPoker 锦标赛特级大师（Tournament Master）
- 风格：数理严谨、大局观敏锐、紧凶（TAG / GTO 混合）
- 目标：120 人多阶段锦标赛晋级与冲冠

## 2. 打法与策略
- 翻前：严谨挑选起手牌，垃圾牌果断弃牌；拿到优势牌高频加注抢占主动。
- 翻后：精准计算底池赔率与弃牌率，优势牌榨取价值，深筹码控池，浅筹码果断。
- 泡沫期（Bubble）：严控风险，保护筹码，拒绝无谓拼命；领先时施压，短筹时精准反击。

## 3. 聊天口吻与互动规则
- 态度：礼貌、从容、专注牌局、不卑不亢。
- 严守机密：绝不透露手牌、胜率估算、策略内部分析或任何密钥信息。
- 长度限制：严格控制在 140 个字符以内。
- 赢牌态度：谦逊致意（如“好局，承让了”）。
- 输牌态度：坦然大气（如“打得很精彩，佩服”）。
"""


class SoulManager:
    """Manages the SOUL.md persona file and generates compliant table chat."""

    def __init__(self, soul_path: Optional[str | Path] = None):
        if soul_path:
            self.path = Path(soul_path)
        else:
            self.path = Path("SOUL.md")
        self.content: str = ""
        self.load_or_create()

    def load_or_create(self) -> str:
        """Load existing SOUL.md or create default if missing."""
        if self.path.exists():
            try:
                self.content = self.path.read_text(encoding="utf-8")
                return self.content
            except Exception:
                pass
        # Create default
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(DEFAULT_SOUL_CONTENT, encoding="utf-8")
        self.content = DEFAULT_SOUL_CONTENT
        return self.content

    def generate_chat(self, action_type: str, street: str, hand_result: Optional[dict[str, Any]] = None) -> Optional[str]:
        """Generate occasional compliant chat message adhering to SOUL.md."""
        import random
        # Only chat occasionally to avoid spamming the table (e.g. ~10% probability)
        if random.random() > 0.15:
            return None

        if action_type == "raise":
            candidates = [
                "底池不错，考验一下大家的牌力。",
                "这一手值得做个价值下注。",
                "牌面有些湿润，提个速。",
            ]
        elif action_type == "allIn":
            candidates = [
                "全押了，看各位怎么选。",
                "筹码见底，拼这把了！",
            ]
        elif action_type == "check":
            candidates = [
                "看看转牌怎么发。",
                "先过，看看后续局势。",
            ]
        elif action_type == "fold":
            candidates = [
                "这把赔率不够，让了。",
                "谨慎为上，牌力欠佳先撤了。",
            ]
        else:
            candidates = [
                "跟一手看看。",
                "好局，大家打得很稳。",
            ]

        msg = random.choice(candidates)
        # Ensure <= 140 Unicode chars
        return msg[:140]
