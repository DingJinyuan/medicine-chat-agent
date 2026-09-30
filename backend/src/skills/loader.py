"""Skill 渐进式加载器（仿 Claude Code Agent Skills 机制）。

reasoning 等节点运行时按需读取 SKILL.md，去掉 YAML frontmatter 后返回 Markdown body，
作为节点的 system prompt。改 SKILL.md 立即生效，无需改代码。
"""
from __future__ import annotations

import re
from pathlib import Path

SKILLS_DIR = Path(__file__).parent

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def load_skill(name: str) -> str:
    """按需加载指定 skill 的 Markdown body（渐进式加载）。

    Args:
        name: skill 目录名（必须与 SKILL.md 所在目录名一致）。

    Returns:
        str: SKILL.md 去掉 frontmatter 后的 body 文本。

    Raises:
        FileNotFoundError: skill 目录或 SKILL.md 不存在。
    """
    skill_file = SKILLS_DIR / name / "SKILL.md"
    if not skill_file.exists():
        raise FileNotFoundError(f"Skill '{name}' not found at {skill_file}")
    text = skill_file.read_text(encoding="utf-8")
    return _FRONTMATTER_RE.sub("", text, count=1).strip()
