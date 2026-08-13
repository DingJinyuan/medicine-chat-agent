"""LangGraph state machine for one chat turn.

    classify_triage --conditional--> emergency_shortcut --> output_guardrail --> END
                     \\-------------> retrieve --> generate --> output_guardrail --> END

The emergency branch is a genuine conditional (not just sequential steps):
a high-confidence "emergency" triage prediction skips retrieval and
generation entirely and returns a fixed, safe response -- the LLM never
sees emergency-flagged input. Both branches converge on `output_guardrail`
so the disclaimer/dosage/diagnosis rewrite rules apply unconditionally.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app import metrics as m
from app.config import Settings
from app.llm_client import generate_answer
from app.security import enforce_medical_guardrails, scan_for_injection, wrap_untrusted
from app.triage_classifier import TriageClassifier

EMERGENCY_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_THRESHOLD = 0.4

EMERGENCY_RESPONSE = (
    "What you're describing sounds like it could be a medical emergency. "
    "Please call your local emergency number (911 in the US) or go to the "
    "nearest emergency room right now. Do not wait for an online response."
)


class ChatState(TypedDict, total=False):
    question: str
    triage_label: str
    triage_confidence: float
    context_blocks: list[str]
    sources: list[dict]
    injection_flagged: bool
    answer: str
    guardrail_rewritten: bool


def _classify_triage_node(triage_classifier: TriageClassifier):
    def node(state: ChatState) -> ChatState:
        result = triage_classifier.classify(state["question"])
        m.TRIAGE_LABELS.labels(label=result.label).inc()
        return {**state, "triage_label": result.label, "triage_confidence": result.confidence}

    return node


def _route_after_triage(state: ChatState) -> str:
    if state["triage_label"] == "emergency" and state["triage_confidence"] >= EMERGENCY_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"
    if state["triage_confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"   # 任何标签置信度太低，走保守兜底
    return "retrieve"


def _emergency_shortcut_node(state: ChatState) -> ChatState:
    return {**state, "answer": EMERGENCY_RESPONSE, "context_blocks": [], "sources": [], "injection_flagged": False}


def _retrieve_node(retriever, top_k: int):
    def node(state: ChatState) -> ChatState:
        docs_with_scores = retriever.similarity_search_with_score(state["question"], k=top_k)

        context_blocks = []
        sources = []
        any_flagged = False
        for doc, score in docs_with_scores:
            scan = scan_for_injection(doc.page_content)
            any_flagged = any_flagged or scan.flagged
            context_blocks.append(wrap_untrusted(doc.metadata.get("topic", "unknown"), doc.page_content))
            sources.append(
                {
                    "topic": doc.metadata.get("topic", "unknown"),
                    "url": doc.metadata.get("url", ""),
                    "text": doc.page_content,
                    "score": float(score),
                }
            )

        return {**state, "context_blocks": context_blocks, "sources": sources, "injection_flagged": any_flagged}

    return node


def _generate_node(settings: Settings):
    def node(state: ChatState) -> ChatState:
        answer = generate_answer(settings, state["question"], state["context_blocks"])
        return {**state, "answer": answer}

    return node


def _output_guardrail_node(state: ChatState) -> ChatState:
    result = enforce_medical_guardrails(state["answer"])
    return {**state, "answer": result.text, "guardrail_rewritten": result.rewritten}


def build_graph(settings: Settings, retriever, triage_classifier: TriageClassifier):
    graph = StateGraph(ChatState)

    graph.add_node("classify_triage", _classify_triage_node(triage_classifier))
    graph.add_node("emergency_shortcut", _emergency_shortcut_node)
    graph.add_node("retrieve", _retrieve_node(retriever, settings.retrieval_top_k))
    graph.add_node("generate", _generate_node(settings))
    graph.add_node("output_guardrail", _output_guardrail_node)

    graph.set_entry_point("classify_triage")
    graph.add_conditional_edges(
        "classify_triage",
        _route_after_triage,
        {"emergency_shortcut": "emergency_shortcut", "retrieve": "retrieve"},
    )
    graph.add_edge("emergency_shortcut", "output_guardrail")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()
