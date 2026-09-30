"""L1 安全测试:限流 + 鉴权(护栏/注入/脱敏已在 test_boundary 覆盖)。

补充计划遗漏点:限流窗口/独立计数、鉴权三态(空 key 关闭 / 匹配通过 / 不匹配 401)。
"""
import asyncio
import time

import pytest

from src.common.security import RateLimiter, require_api_key
from src.common.errors import AppError
from src.config import settings


class TestRateLimiter:
    def test_blocks_over_limit(self):
        rl = RateLimiter(limit_per_minute=1)
        assert rl.check("client-a") is True
        assert rl.check("client-a") is False

    def test_independent_clients(self):
        rl = RateLimiter(limit_per_minute=1)
        assert rl.check("client-a") is True
        assert rl.check("client-b") is True

    def test_window_resets(self, monkeypatch):
        rl = RateLimiter(limit_per_minute=1)
        fake_time = [0.0]
        monkeypatch.setattr(time, "time", lambda: fake_time[0])
        assert rl.check("client-a") is True
        fake_time[0] = 61.0  # 跨过 60s 窗口,旧 hit 过期
        assert rl.check("client-a") is True

    def test_evicts_expired_clients(self, monkeypatch):
        # 淘汰机制:client_id 过多时清理已过期的,防内存泄漏
        rl = RateLimiter(limit_per_minute=1)
        fake_time = [0.0]
        monkeypatch.setattr(time, "time", lambda: fake_time[0])
        for i in range(10001):
            rl._hits[f"client-{i}"] = [-100.0]  # 时间戳 -100,已过期(远超 60s 窗口)
        rl.check("new-client")  # 触发淘汰
        assert "client-0" not in rl._hits  # 过期 client 被清理
        assert "new-client" in rl._hits


class TestRequireApiKey:
    def test_empty_key_disables_auth(self, monkeypatch):
        monkeypatch.setattr(settings, "app_api_key", "")
        asyncio.run(require_api_key(x_api_key="anything"))  # 不抛异常

    def test_matching_key_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "app_api_key", "correct-key")
        asyncio.run(require_api_key(x_api_key="correct-key"))

    def test_mismatch_key_raises_401(self, monkeypatch):
        monkeypatch.setattr(settings, "app_api_key", "correct-key")
        with pytest.raises(AppError):
            asyncio.run(require_api_key(x_api_key="wrong-key"))
