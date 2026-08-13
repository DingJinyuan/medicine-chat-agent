"""Loads the LoRA-finetuned triage classifier (see finetuning/train_lora.py)
and exposes a single `classify` call used by the LangGraph `classify_triage` node.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import structlog
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class TriageResult:
    label: str
    confidence: float


class TriageClassifier:
    def __init__(self, adapter_path: str, base_model: str):
        label_map = json.loads(Path(adapter_path, "label_map.json").read_text())
        self.id2label = {int(k): v for k, v in label_map["id2label"].items()}

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(self.id2label)
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    @torch.no_grad()
    def classify(self, text: str) -> TriageResult:
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        return TriageResult(label=self.id2label[idx], confidence=float(probs[idx].item()))


logger = structlog.get_logger("medisense")


class ConservativeClassifier:
    """安全兜底：模型加载失败时，把所有输入判为 emergency。

    刻意的安全优先选择——分类器是安全关键件，挂了宁可全拒（返回急救建议）
    也不漏放真正紧急的情况。代价是后端退化为「只能答急救」，接受此行为。
    """
    def classify(self, text: str) -> TriageResult:
        return TriageResult(label="emergency", confidence=1.0)


@lru_cache
def _load_triage_classifier(adapter_path: str, base_model: str) -> TriageClassifier:
    """只缓存「成功加载」的结果；失败抛异常，不进缓存"""
    return TriageClassifier(adapter_path, base_model)


def get_triage_classifier(adapter_path: str, base_model: str):
    try:
        return _load_triage_classifier(adapter_path, base_model)
    except Exception:
        logger.exception("triage_classifier_load_failed", adapter_path=adapter_path)
        return ConservativeClassifier()   # 失败不缓存，下次请求再试
