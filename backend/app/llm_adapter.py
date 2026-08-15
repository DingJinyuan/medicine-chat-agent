import json
import os, time
import random
import structlog
from pathlib import Path
# OpenAI官方SDK类型，用于标准化返回消息结构（统一OpenAI / Claude输出格式）
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall, Function,
)
# 读取.env环境变量文件
from dotenv import load_dotenv

# 加载 backend/.env（显式路径，避免工作目录不同导致找不到）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# 全局日志实例，记录调用耗时、token消耗、模型返回内容
logger = structlog.get_logger("medisense")

# ===================== 模型厂商配置注册表 =====================
# 前缀匹配规则：通过模型名开头前缀自动识别厂商，自动读取对应环境变量
# key：模型前缀，value：对应密钥环境变量名、接口地址环境变量、厂商专属额外请求参数
MODEL_PROFILES = {
    "deepseek": {
        "api_key": "DEEPSEEK_API_KEY",       # .env内密钥变量名
        "base_url": "DEEPSEEK_BASE_URL",     # .env内接口地址变量名
        "extra_body": {"thinking": {"type": "disabled"}}, # DeepSeek专属参数：关闭思考过程
    },
    "glm": {
        "api_key": "GLM_API_KEY",
        "base_url": "GLM_BASE_URL",
    },
    "gemini": {
        "api_key": "GOOGLE_API_KEY",
        "base_url": "GOOGLE_BASE_URL",
    },
    "gpt": {
        "api_key": "OPENAI_API_KEY",
        # OpenAI官方接口无需自定义base_url，省略
    },
    "qwen": {
        "api_key": "QWEN_API_KEY",
        "base_url": "QWEN_BASE_URL",
        "extra_body": {"enable_thinking": False}, # 通义千问专属参数：关闭思维链输出
    },
    "kimi": {
        "api_key": "MOONSHOT_API_KEY",
        "base_url": "MOONSHOT_BASE_URL",
    },
    "minimax": {
        "api_key": "MINIMAX_API_KEY",
        "base_url": "MINIMAX_BASE_URL",
    },
    "ollama": {
        "api_key": "OLLAMA_API_KEY",  # 本地Ollama随便填字符串即可，不需要真实密钥
        "base_url": "OLLAMA_BASE_URL",
    },
    "claude-open": { # Claude 通过 OpenAI 兼容中转
        "api_key": "CLAUDE_OPEN_API_KEY",
        "base_url": "CLAUDE_OPEN_BASE_URL",
    },
    # ========== 新增全局中转配置 ==========
    "proxy": {
        "api_key": "PROXY_API_KEY",  # 中转平台给你的全局密钥
        "base_url": "PROXY_BASE_URL",  # 中转平台统一接口地址，固定结尾 /v1
    }
}

def _resolve_model_env(model: str) -> tuple[str, str | None, dict | None]:
    """
    根据模型名称前缀，自动解析对应的API密钥、接口地址、额外参数
    :param model: 完整模型名称，如 deepseek-v4-flash
    :return: (api_key, base_url, extra_body)
    """
    # 遍历厂商注册表，进行前缀匹配
    for prefix, profile in MODEL_PROFILES.items():
        if model.lower().startswith(prefix):
            # 1. profile["api_key"] 拿到的是【环境变量的变量名】，不是密钥本身
            # 比如deepseek配置里的 "DEEPSEEK_API_KEY"
            api_key_env = profile["api_key"]
            # 读取.env中该变量的值，如果为空，直接抛异常提前报错
            if not os.getenv(api_key_env):
                raise ValueError(
                    f"模型 '{model}' 匹配到 '{prefix}'，"
                    f"但环境变量 {api_key_env} 未设置"
                )

            # 2. 读取自定义接口地址（国内厂商/中转地址必备）
            base_url = None
            if "base_url" in profile:
                base_url_env = profile["base_url"]
                base_url = os.getenv(base_url_env)
                if not base_url:
                    raise ValueError(
                        f"模型 '{model}' 匹配到 '{prefix}'，"
                        f"但环境变量 {base_url_env} 未设置"
                    )

            # 3. 返回密钥、接口地址、厂商专属额外请求体
            return (
                os.getenv(api_key_env),
                base_url,
                profile.get("extra_body"),
            )

    # 所有前缀都不匹配，抛出异常
    all_prefixes = list(MODEL_PROFILES.keys()) + ["claude"]
    raise ValueError(
        f"未知模型 '{model}'，支持的前缀: {all_prefixes}。"
        f"请在 MODEL_PROFILES 中添加配置，或显式传入 api_key/base_url。"
    )

# ===================== 异常分类工具 =====================
def _is_retryable_error(error: Exception) -> bool:
    """
    判断异常是否可重试。网络超时、限流、服务端错误 → 可重试。
    认证失败、参数错误、JSON 解析失败 → 不可重试，应直接降级到备用模型。
    """
    error_str = str(error).lower()
    # 不可重试的错误类型
    non_retryable_patterns = [
        "jsondecodeerror", "json", "invalid json",
        "401", "unauthorized", "authentication", "invalid api key",
        "400", "bad request", "invalid parameter",
        "402", "payment required", "insufficient",
        "model not found", "not found",
    ]
    for pattern in non_retryable_patterns:
        if pattern in error_str:
            return False

    # JSON 解析失败直接不可重试
    if isinstance(error, json.JSONDecodeError):
        return False

    # 其余（超时、连接、5xx、429 限流等）可重试
    return True

# ===================== 统一LLM客户端主类 =====================
# 核心设计：统一入参、统一返回OpenAI标准消息结构，上层Agent完全不用区分底层模型是OpenAI/Claude/DeepSeek
class LLMClient:
    def __init__(
        self,
        model: str = None,          # 模型名称，优先传参，其次读取环境变量LLM_MODEL_ID
        api_key: str = None,        # 手动传入密钥（优先级高于.env）
        base_url: str = None,       # 手动传入接口地址
        timeout: int = None,        # 请求超时时间
        fallback_models: list[str] | None = None,  # 主模型失败时的备用模型列表
        max_retries: int = 2,       # 单模型最大重试次数（含首次调用）
    ):
        # 模型名称 + 备用模型列表
        self.model = model or os.getenv("LLM_MODEL_ID")
        if not self.model:
            raise ValueError("未指定模型。请使用 --model 参数或在 .env 中设置 LLM_MODEL_ID")
        self.fallback_models = fallback_models or []
        self.max_retries = max_retries
        self._current_model = self.model  # 追踪当前实际使用的模型

        # 超时时间：入参优先，默认读取.env LLM_TIMEOUT，兜底60秒
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        # 初始化主模型客户端
        self._init_client(self.model, api_key, base_url)

        # 全局Token统计指标，用于成本核算
        self.total_prompt_tokens = 0      # 累计输入token
        self.total_completion_tokens = 0  # 累计输出token
        self.total_calls = 0              # 累计调用次数
        self._fallback_used = False       # 标记是否触发过降级

    def _init_client(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """初始化指定模型的客户端，支持主模型和备用模型"""
        self._current_model = model
        self._is_claude = model.lower().startswith("claude")

        if self._is_claude:
            try:
                import anthropic
            except ImportError:
                raise ImportError("Claude 模型需要安装 anthropic SDK: pip install anthropic")
            _api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not _api_key:
                raise ValueError("Claude 模型需要设置环境变量 ANTHROPIC_API_KEY")
            self._anthropic_client = anthropic.Anthropic(api_key=_api_key, timeout=self.timeout)
            self.extra_body = None
            self.client = None
        else:
            if api_key and base_url:
                self.extra_body = None
            else:
                api_key, base_url, self.extra_body = _resolve_model_env(model)
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)

    def reset_stats(self):
        """重置token调用统计数据，适合单次研究任务开始前清零"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    @property
    def active_model(self) -> str:
        """返回当前实际使用的模型名（可能已降级到备用模型）"""
        return self._current_model

    def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0,
        tools=None,
        tool_choice=None,
        max_tokens: int = 4096,
        tag: str = "",
    ) -> ChatCompletionMessage:
        """
        统一对外调用入口，内置主备降级：
        主模型失败 → 依次尝试 fallback_models → 全部失败则抛异常
        """
        models_to_try = [self.model] + self.fallback_models
        last_error = None

        for attempt, model in enumerate(models_to_try):
            # 切换模型时重新初始化客户端
            if attempt > 0:
                logger.warning("llm_fallback", tag=tag, model=model, attempt=attempt, total=len(models_to_try) - 1)
                try:
                    self._init_client(model)
                except Exception as e:
                    logger.error("llm_fallback_init_failed", tag=tag, model=model, error=str(e))
                    last_error = e
                    continue
                self._fallback_used = True

            # 每个模型最多重试 max_retries 次
            for retry in range(self.max_retries):
                try:
                    if tool_choice is None:
                        tool_choice = "auto" if tools else None

                    start = time.time()

                    if self._is_claude:
                        result = self._invoke_claude(messages, temperature, tools, tool_choice, max_tokens, tag)
                    else:
                        response = self.client.chat.completions.create(
                            messages=messages,
                            model=self._current_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tool_choice=tool_choice,
                            tools=tools if tools else None,
                            extra_body=self.extra_body,
                        )
                        elapsed = time.time() - start
                        result = response.choices[0].message
                        self._track_usage(
                            response.usage.prompt_tokens,
                            response.usage.completion_tokens,
                            elapsed, tag, result,
                        )

                    return result

                except Exception as e:
                    last_error = e
                    # 判断异常是否可重试
                    retryable = _is_retryable_error(e)

                    if not retryable:
                        # 不可重试（认证失败、参数错误、JSON 解析失败等），直接跳到下一个备用模型
                        logger.error("llm_non_retryable_error", tag=tag, model=self._current_model, error=str(e))
                        break

                    if retry < self.max_retries - 1:
                        # 指数退避 + 随机抖动（封顶 32s），防止重试雪崩
                        base = min(1.0 * (2 ** retry), 32.0)
                        delay = base + random.uniform(0, base * 0.25)
                        logger.warning("llm_retry", tag=tag, model=self._current_model, retry=retry + 1, max_retries=self.max_retries - 1, error=str(e))
                        time.sleep(delay)
                    else:
                        logger.error("llm_all_retries_failed", tag=tag, model=self._current_model, max_retries=self.max_retries, error=str(e))
                        break  # 跳出重试循环，尝试下一个备用模型

        raise RuntimeError(
            f"[LLM:{tag}] 所有模型调用失败 (已尝试 {len(models_to_try)} 个模型)。最后错误: {last_error}"
        )

    # ===================== Claude 模型私有适配层 =====================
    def _invoke_claude(self, messages, temperature, tools, tool_choice, max_tokens, tag):
        """Claude模型底层调用，完成 OpenAI消息格式 → Claude私有格式 转换"""
        system, anthropic_msgs = self._translate_messages(messages)

        # 工具格式转换：OpenAI Function格式 → Claude Tool格式
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]

        # 工具调用策略映射
        anthropic_tc = None
        if anthropic_tools:
            if tool_choice is None or tool_choice == "auto":
                anthropic_tc = {"type": "auto"}
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                anthropic_tc = {"type": "tool", "name": tool_choice["function"]["name"]}
            elif tool_choice == "none":
                anthropic_tools = None

        # 组装Claude请求参数
        kwargs = {
            "model": self.model,
            "messages": anthropic_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if anthropic_tc:
            kwargs["tool_choice"] = anthropic_tc

        start = time.time()
        response = self._anthropic_client.messages.create(**kwargs)
        elapsed = time.time() - start

        # 将Claude返回结果 反向转换为OpenAI标准ChatCompletionMessage对象
        content = None
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                # 工具调用结构体对齐OpenAI格式
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=block.id,
                        type="function",
                        function=Function(
                            name=block.name,
                            arguments=json.dumps(block.input, ensure_ascii=False),
                        ),
                    )
                )

        msg = ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )
        # 统计日志
        self._track_usage(
            response.usage.input_tokens,
            response.usage.output_tokens,
            elapsed, tag, msg,
        )
        return msg

    @staticmethod
    def _translate_messages(messages):
        """
        静态工具方法：OpenAI标准消息数组 转 Claude 消息格式
        OpenAI：system/user/assistant/tool 扁平数组
        Claude：system字段独立，tool结果需要包装为user消息，assistant工具调用为特殊block结构
        :return: (system提示词, claude格式消息列表)
        """
        system_parts = []
        result = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            # 1. 提取system角色，Claude单独参数传入，不放入messages列表
            if role == "system":
                system_parts.append(msg["content"])
                i += 1

            # 2. user角色直接兼容，格式无需改动
            elif role == "user":
                result.append({"role": "user", "content": msg["content"]})
                i += 1

            # 3. assistant角色：文本 + 工具调用，转为Claude的block数组结构
            elif role == "assistant":
                blocks = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        args = tc["function"]["arguments"]
                        blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(args) if isinstance(args, str) else args,
                        })
                if not blocks:
                    blocks.append({"type": "text", "text": ""})
                result.append({"role": "assistant", "content": blocks})
                i += 1

            # 4. tool工具返回结果：Claude要求必须包装成user角色的tool_result块
            elif role == "tool":
                tool_results = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": messages[i]["tool_call_id"],
                        "content": messages[i]["content"] or "",
                    })
                    i += 1
                result.append({"role": "user", "content": tool_results})

            else:
                i += 1

        system = "\n\n".join(system_parts) if system_parts else None
        return system, result

    # ===================== 通用日志&统计 =====================
    def _track_usage(self, prompt_tokens, completion_tokens, elapsed, tag, msg):
        """
        统计单次调用Token、耗时，写入日志
        :param prompt_tokens: 输入token
        :param completion_tokens: 输出token
        :param elapsed: 耗时秒数
        :param tag: Agent标记（如Supervisor）
        :param msg: 模型返回消息
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1
        total = prompt_tokens + completion_tokens
        # INFO级别打印耗时、Token总量
        logger.info(
            "llm_response",
            tag=tag, elapsed=elapsed, total_tokens=total,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )
        # DEBUG级别打印前500字符模型返回内容，方便调试
        logger.debug("llm_output", tag=tag, content=str(msg.content)[:500])
