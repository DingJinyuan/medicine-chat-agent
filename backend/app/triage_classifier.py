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
    """
        分诊分类输出结果数据类

        Fields:
            label: 分类标签字符串，例如 "emergency" / "normal"
            confidence: 该标签对应的预测置信度，取值 [0, 1]
        """
    label: str
    confidence: float


class TriageClassifier:
    """
    LoRA微调后的文本分诊分类器
    用于医疗场景：对用户输入文本做序列分类，输出标签与置信度，供LangGraph做条件路由
    """
    def __init__(self, adapter_path: str, base_model: str):
        """
                初始化LoRA分诊分类器
                1. 读取label映射文件 id2label
                2. 加载tokenizer（从LoRA适配器目录读取）
                3. 加载基础分类模型，再注入LoRA适配器权重
                4. 设置模型为eval评估模式，关闭dropout等训练行为
                id2label      "0": "emergency", "1": "urgent", "2": "routine", "3": "self_care"
                Args:
                    adapter_path: LoRA适配器本地目录路径，内含adapter权重、label_map.json
                    base_model: 基座模型名称/本地路径
                """
        # 读取标签映射文件，json中key为字符串，转int作为模型输出id
        label_map = json.loads(Path(adapter_path, "label_map.json").read_text())
        self.id2label = {int(k): v for k, v in label_map["id2label"].items()}
        # 加载分词器：分词器随LoRA适配器一同存放
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        # 加载基座序列分类模型，设置分类类别数量
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(self.id2label)
        )
        # 将LoRA适配器权重挂载到基座模型上
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    @torch.no_grad()
    def classify(self, text: str) -> TriageResult:
        # 文本分词：截断、补padding，最大长度64，返回PyTorch tensor
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
        # 前向推理拿到logits；原始分数
        logits = self.model(**inputs).logits
        # softmax转成概率分布，取第0条样本（batch_size=1） 加起来等于1
        probs = torch.softmax(logits, dim=-1)[0]
        # 获取概率最大的类别下标 选最大
        idx = int(torch.argmax(probs).item())
        # 下标映射标签，取出对应置信度，转为python float返回
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
