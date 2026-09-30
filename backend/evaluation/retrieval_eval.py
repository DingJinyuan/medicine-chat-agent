"""真实检索质量评估(患者版 MedlinePlus,完整 IR 指标)。

两套查询对比:
- 简单查询(主题名明确,golden_dataset 的 groundedness 用例)
- 模糊查询(不点主题名,只描述症状,考验检索的语义理解能力)

标注方式:topic 近似(相关 = 该主题下的所有 chunk)。
运行:cd backend && python -m evaluation.retrieval_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from sqlalchemy import text

from src.ingestion.embedder import Embedder
from src.service.patient_retrieval_service import retrieve_medlineplus
from src.database.connection import get_db

from evaluation.ir_metrics import recall_at_k, hit_at_k, mrr, ndcg_at_k

TOP_K = 5

# 模糊查询:只描述症状,不点主题名
HARD_QUERIES = [
    {"question": "I've been feeling really thirsty and urinating much more than usual lately, what could be causing this?", "relevant_topic": "Diabetes"},
    {"question": "I get bad throbbing headaches where bright light and loud sounds really bother me", "relevant_topic": "Migraine"},
    {"question": "I keep wheezing and my chest feels tight, especially at night or in the early morning", "relevant_topic": "Asthma"},
    {"question": "Right after eating peanuts my lips tingled and I got itchy bumps on my skin", "relevant_topic": "Food Allergy"},
    {"question": "I've been sneezing a lot and my nose is completely stuffed up for the past few days", "relevant_topic": "Common Cold"},
]


def topic_chunk_ids(topic: str) -> set:
    with get_db() as db:
        rows = db.execute(
            text("SELECT id FROM chunks WHERE meta->>'topic' = :topic"),
            {"topic": topic},
        ).fetchall()
        return {r[0] for r in rows}


def evaluate(queries: list[dict], embedder, top_k: int = TOP_K) -> list[dict]:
    """对一组查询跑检索,返回每条查询的 IR 指标。"""
    out = []
    for q in queries:
        relevant_ids = topic_chunk_ids(q["relevant_topic"])
        retrieved = retrieve_medlineplus(q["question"], embedder, top_k=top_k)
        retrieved_ids = [c["id"] for c in retrieved]
        out.append({
            "question": q["question"],
            "topic": q["relevant_topic"],
            "hit": hit_at_k(retrieved_ids, relevant_ids, top_k),
            "recall": recall_at_k(retrieved_ids, relevant_ids, top_k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, top_k),
        })
    return out


def _print(rows: list[dict], top_k: int):
    print(f"{'question':<60} {'topic':<13} {'hit@k':>6} {'recall@k':>9} {'mrr':>6} {'ndcg@k':>7}")
    print("-" * 105)
    for r in rows:
        print(f"{r['question'][:58]:<60} {r['topic']:<13} {r['hit']:>6.2f} {r['recall']:>9.4f} {r['mrr']:>6.3f} {r['ndcg']:>7.4f}")
    print("-" * 105)
    n = len(rows)
    print(f"平均 hit@{top_k}:    {sum(r['hit'] for r in rows) / n:.2f}")
    print(f"平均 recall@{top_k}:  {sum(r['recall'] for r in rows) / n:.4f}")
    print(f"平均 MRR:      {sum(r['mrr'] for r in rows) / n:.4f}")
    print(f"平均 NDCG@{top_k}:  {sum(r['ndcg'] for r in rows) / n:.4f}")


def main() -> int:
    # 简单查询:从 golden_dataset 的 groundedness 用例拿
    data_dir = Path(__file__).resolve().parents[1] / "data" / "eval"
    cases = json.loads((data_dir / "golden_dataset.json").read_text(encoding="utf-8"))
    easy = [{"question": c["question"], "relevant_topic": c["expected_topic"]}
            for c in cases if c.get("expected_topic")]

    print("loading embedder...")
    embedder = Embedder()

    print("\n===== 简单查询(主题名明确) =====")
    easy_rows = evaluate(easy, embedder)
    _print(easy_rows, TOP_K)

    print("\n===== 模糊查询(只描述症状,不点主题名) =====")
    hard_rows = evaluate(HARD_QUERIES, embedder)
    _print(hard_rows, TOP_K)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
