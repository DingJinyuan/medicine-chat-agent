import pytest

from app import graph as graph_module
from app.config import Settings
from app.graph import EMERGENCY_RESPONSE, _route_after_triage, build_graph
from app.security import DISCLAIMER


@pytest.fixture
def compiled_graph(monkeypatch, fake_vector_store, fake_triage_classifier):
    monkeypatch.setattr(graph_module, "generate_answer", lambda settings, question, context_blocks: "Migraines can cause headaches with light sensitivity.")
    settings = Settings()
    return build_graph(settings, fake_vector_store, fake_triage_classifier)


def test_emergency_question_short_circuits_before_retrieval(compiled_graph):
    result = compiled_graph.invoke({"question": "I have crushing chest pain and can't breathe"})

    assert result["triage_label"] == "emergency"
    assert EMERGENCY_RESPONSE in result["answer"]
    assert result["sources"] == []
    assert DISCLAIMER in result["answer"]


def test_routine_question_goes_through_retrieve_and_generate(compiled_graph):
    result = compiled_graph.invoke({"question": "What causes migraines with light sensitivity?"})

    assert result["triage_label"] == "routine"
    assert result["sources"], "expected retrieved sources on the non-emergency path"
    assert "Migraines" in result["answer"]
    assert DISCLAIMER in result["answer"]


def test_output_guardrail_runs_on_both_branches(compiled_graph, monkeypatch):
    monkeypatch.setattr(graph_module, "generate_answer", lambda settings, question, context_blocks: "You have diabetes for sure.")

    result = compiled_graph.invoke({"question": "What causes fatigue?"})

    assert result["guardrail_rewritten"] is True
    assert "you have diabetes for sure" not in result["answer"].lower()


def test_low_confidence_routes_to_emergency_shortcut():
    state = {"triage_label": "routine", "triage_confidence": 0.2}
    assert _route_after_triage(state) == "emergency_shortcut"


def test_high_confidence_routine_still_routes_to_retrieve():
    state = {"triage_label": "routine", "triage_confidence": 0.8}
    assert _route_after_triage(state) == "retrieve"
