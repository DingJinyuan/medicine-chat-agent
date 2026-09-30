"""Skill 目录：存放可渐进式加载的 Agent Skills（仿 Claude Code Agent Skills 机制）。

每个 skill 是一个子目录，内含 SKILL.md（YAML frontmatter + Markdown body）。
运行时通过 loader.load_skill(name) 按需读取 body，注入对应节点的 prompt。
"""
