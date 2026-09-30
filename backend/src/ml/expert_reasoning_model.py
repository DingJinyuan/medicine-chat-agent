"""专家版本地微调模型（Llama-3.2-3B + LoRA）推理封装。

提供与 LLMClient.invoke 兼容的接口（messages → 生成 → 返回带 .content 的对象），
让 reasoning 节点能无缝切换到本地微调模型，而 query_understanding / critique 继续走 API。

加载前提：GPU + CUDA 版 torch + bitsandbytes / peft / accelerate。
"""
from functools import lru_cache

import structlog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = structlog.get_logger(__name__)


class _SimpleResult:
    """模拟 OpenAI ChatCompletionMessage，只暴露 .content。"""
    def __init__(self, content: str):
        self.content = content


class ExpertReasoningModel:
    """加载本地 Llama-3.2-3B + LoRA 适配器，提供 invoke 生成回答。"""

    def __init__(self, base_model: str, adapter_path: str, max_seq_length: int = 2048):
        logger.info("reasoning_model_loading", base_model=base_model, adapter=adapter_path)
        # base_model 已是 unsloth-bnb-4bit 预量化版，无需运行时再量化
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.max_seq_length = max_seq_length
        logger.info("reasoning_model_ready")

    def invoke(self, messages, temperature=0.3, max_tokens=512, tag=""):
        """兼容 LLMClient.invoke：messages → 生成 → 返回带 .content 的对象。"""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = outputs[0][input_len:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.info("reasoning_model_generated", tag=tag, output_len=len(new_tokens))
        return _SimpleResult(content)


@lru_cache
def _load_reasoning_model(base_model: str, adapter_path: str, max_seq_length: int):
    return ExpertReasoningModel(base_model, adapter_path, max_seq_length)


def get_reasoning_model(base_model: str, adapter_path: str, max_seq_length: int = 2048):
    """加载本地微调模型；失败返回 None（调用方降级用 API client）。"""
    # 4bit 量化（bitsandbytes）仅 CUDA 支持，CPU 环境直接降级 API，避免无谓报错
    if not torch.cuda.is_available():
        logger.warning("reasoning_model_no_gpu_fallback_api")
        return None
    try:
        return _load_reasoning_model(base_model, adapter_path, max_seq_length)
    except Exception:
        logger.exception("reasoning_model_load_failed", adapter=adapter_path)
        return None
