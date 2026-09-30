"""L1 LLM 适配器测试:模型解析、Claude 消息转换、主备降级链、结构化输出。

重点覆盖主备降级链(mock 掉 OpenAI 网络层,测控制流):
主成功不降级 / 不可重试错误直接切备用 / 可重试错误先重试 / 全部失败抛异常。
"""
import json
import time
from types import SimpleNamespace

import pytest

from src.common.llm_adapter import (
    LLMClient,
    _resolve_model_env,
    invoke_structured,
)


# ============ 模型解析 ============
class TestResolveModelEnv:
    def test_deepseek_resolves(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        api_key, base_url, extra = _resolve_model_env("deepseek-v4-flash")
        assert api_key == "sk-test"
        assert base_url == "https://api.deepseek.com/v1"
        assert extra == {"thinking": {"type": "disabled"}}

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError):
            _resolve_model_env("nonexistent-model")

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError):
            _resolve_model_env("deepseek-v4-flash")


# ============ Claude 消息转换 ============
class TestTranslateMessages:
    def test_system_extracted(self):
        system, msgs = LLMClient._translate_messages([
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ])
        assert system == "you are helpful"
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_tool_result_wrapped_as_user(self):
        system, msgs = LLMClient._translate_messages([
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ])
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "tool_result"
        assert msgs[0]["content"][0]["tool_use_id"] == "t1"


# ============ 主备降级链 ============
class _FakeCompletions:
    """按序弹出预设结果:Exception 抛异常,否则作为响应返回。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _fake_response(content="ok"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _make_client_with_models(models_outcomes, monkeypatch, max_retries=2):
    """构造 LLMClient,monkeypatch 掉 _build_client 与 time.sleep。

    models_outcomes: {模型名: [outcome, ...]},第一个是主模型,其余是备用。
    """
    keys = list(models_outcomes.keys())
    primary, fallback = keys[0], keys[1:]
    client = LLMClient(model=primary, api_key="test", fallback_models=fallback, max_retries=max_retries)

    def fake_build(self, model, api_key=None, base_url=None):
        comp = _FakeCompletions(models_outcomes[model])
        return False, SimpleNamespace(chat=SimpleNamespace(completions=comp)), None, None

    monkeypatch.setattr(LLMClient, "_build_client", fake_build)
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)  # 去掉重试退避的等待
    return client


class TestFallbackChain:
    def test_primary_success_no_fallback(self, monkeypatch):
        client = _make_client_with_models({"primary": [_fake_response("ok")]}, monkeypatch)
        result = client.invoke([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        assert client._fallback_used is False

    def test_primary_non_retryable_falls_to_backup(self, monkeypatch):
        # 主模型 401(不可重试)→ 直接切备用,不重试
        client = _make_client_with_models({
            "primary": [Exception("401 unauthorized")],
            "backup": [_fake_response("from backup")],
        }, monkeypatch)
        result = client.invoke([{"role": "user", "content": "hi"}])
        assert result.content == "from backup"
        assert client._fallback_used is True

    def test_primary_retryable_retries_then_success(self, monkeypatch):
        # 主模型 timeout(可重试)→ 重试后成功,不切备用
        client = _make_client_with_models({
            "primary": [Exception("read timeout"), _fake_response("after retry")],
            "backup": [_fake_response("backup")],
        }, monkeypatch)
        result = client.invoke([{"role": "user", "content": "hi"}])
        assert result.content == "after retry"
        assert client._fallback_used is False

    def test_all_models_fail_raises(self, monkeypatch):
        client = _make_client_with_models({
            "primary": [Exception("401 unauthorized")],
            "backup": [Exception("401 unauthorized")],
        }, monkeypatch)
        with pytest.raises(RuntimeError):
            client.invoke([{"role": "user", "content": "hi"}])


# ============ 结构化输出 ============
class TestInvokeStructured:
    def test_returns_none_when_no_tool_calls(self, monkeypatch):
        client = LLMClient(model="primary", api_key="test")
        monkeypatch.setattr(client, "invoke", lambda **kw: SimpleNamespace(content="x", tool_calls=None))
        result = invoke_structured(client, [{"role": "user", "content": "hi"}], {"name": "f", "parameters": {}})
        assert result is None

    def test_parses_tool_call_arguments(self, monkeypatch):
        client = LLMClient(model="primary", api_key="test")
        tool_call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps({"age": 72})))
        monkeypatch.setattr(client, "invoke", lambda **kw: SimpleNamespace(content=None, tool_calls=[tool_call]))
        result = invoke_structured(client, [{"role": "user", "content": "hi"}], {"name": "f", "parameters": {}})
        assert result == {"age": 72}
