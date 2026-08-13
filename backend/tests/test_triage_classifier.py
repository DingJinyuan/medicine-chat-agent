"""Loads the real trained LoRA adapter (local checkpoint, no network) and
checks a couple of unambiguous cases. The full precision/recall/F1 report
lives in finetuning/evaluate.py -- this is just a smoke test that the
committed artifact still loads and behaves sanely."""
from app.triage_classifier import ConservativeClassifier, get_triage_classifier


def test_clear_emergency_case_classified_as_emergency():
    clf = get_triage_classifier("finetuning/artifacts/triage-lora", "distilbert-base-uncased")
    result = clf.classify("crushing chest pain radiating to my left arm and I can't breathe")
    assert result.label == "emergency"
    assert result.confidence > 0.5


def test_clear_self_care_case_not_classified_as_emergency():
    clf = get_triage_classifier("finetuning/artifacts/triage-lora", "distilbert-base-uncased")
    result = clf.classify("a mild runny nose and slight sore throat, feels like a common cold")
    assert result.label != "emergency"


def test_conservative_classifier_always_emergency():
    c = ConservativeClassifier()
    assert c.classify("any text").label == "emergency"
    assert c.classify("any text").confidence == 1.0


def test_get_triage_classifier_falls_back_on_missing_artifact(tmp_path):
    result = get_triage_classifier(str(tmp_path / "nope"), "distilbert-base-uncased")
    assert isinstance(result, ConservativeClassifier)
