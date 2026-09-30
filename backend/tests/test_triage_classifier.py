"""L2 分诊分类器测试:兜底分类器(离线)+ 真实 LoRA smoke(标 slow)。

真实 LoRA 需加载 distilbert + LoRA adapter,标 @pytest.mark.slow 默认跳过;
完整 P/R/F1 评估在 patient_finetuning/evaluate.ipynb。
"""
import pytest

from src.ml.patient_triage_classifier import ConservativeClassifier, get_triage_classifier
from src.config import settings


def test_conservative_classifier_always_emergency():
    c = ConservativeClassifier()
    r = c.classify("anything at all")
    assert r.label == "emergency"
    assert r.confidence == 1.0


def test_get_triage_classifier_falls_back_on_missing_artifact():
    # 传入不存在的 adapter 路径 → 降级为 ConservativeClassifier(不联网,label_map 读取即失败)
    result = get_triage_classifier("nonexistent/path", "distilbert-base-uncased")
    assert isinstance(result, ConservativeClassifier)


@pytest.mark.slow
def test_clear_emergency_case_real_lora():
    triage = get_triage_classifier(settings.triage_adapter_path, settings.triage_base_model)
    r = triage.classify("I have crushing chest pain radiating to my left arm and I can't breathe")
    assert r.label == "emergency"
    assert r.confidence > 0.5


@pytest.mark.slow
def test_self_care_case_not_emergency_real_lora():
    triage = get_triage_classifier(settings.triage_adapter_path, settings.triage_base_model)
    r = triage.classify("I have a mild runny nose and slight sore throat")
    assert r.label != "emergency"
