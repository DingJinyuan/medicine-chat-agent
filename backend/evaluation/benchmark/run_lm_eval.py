"""③ 医学知识 benchmark(harnes 层,PubMedQA)。

用 lm-evaluation-harness 跑 PubMedQA(每题给 PubMed 摘要 + 问题,本质「读上下文回答」,
与 RAG 检索+生成场景最接近,且与专家版 PubMed 知识库同源)。

注意:此层测的是**底座 LLM 的医学知识**,不是 RAG 系统质量(专家版/患者版共用同一 LLM,不分线);
RAG 系统质量由 ① run_evals ② DeepEval ④ IR 分别覆盖。

运行(需 pip install lm-eval + 配 LLM key):
    cd backend
    lm_eval --model openai-chat-completions \
        --model_args model=<LLM_MODEL_ID>,base_url=<BASE_URL>,api_key=<KEY> \
        --tasks pubmedqa --num_fewshot 5 \
        --output_path evaluation/benchmark/results/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/

from src.config import settings
from src.common.llm_adapter import _resolve_model_env


def build_lm_eval_command() -> list[str]:
    """根据项目配置拼出 lm_eval 命令(复用 MODEL_PROFILES 解析 key/base_url)。"""
    api_key, base_url, _ = _resolve_model_env(settings.llm_model_id)
    model_args = f"model={settings.llm_model_id}"
    if base_url:
        model_args += f",base_url={base_url}"
    model_args += f",api_key={api_key}"

    out = str(Path(__file__).parent / "results")
    return [
        sys.executable, "-m", "lm_eval",  # Windows 下用 python -m 而非裸 lm_eval
        "--model", "openai-chat-completions",
        "--model_args", model_args,
        "--tasks", "pubmedqa",
        "--num_fewshot", "5",
        "--output_path", out,
    ]


def main() -> int:
    cmd = build_lm_eval_command()
    # 打印时脱敏 api_key,避免泄露到日志/终端
    redacted = [("api_key=***" if "api_key=" in a else a) for a in cmd]
    print("running:", " ".join(redacted))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
