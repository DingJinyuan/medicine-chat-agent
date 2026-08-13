import re

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DIM = 384


class FakeEmbeddings(Embeddings):
    """Deterministic, network-free stand-in for fastembed.

    Uses hashed word features so texts sharing vocabulary land close in
    similarity space -- real enough to exercise ranking logic without
    downloading a model in every test run.
    """

    def _vec(self, text: str) -> list[float]:
        vector = np.zeros(DIM, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            idx = hash(word) % DIM
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeTriageClassifier:
    """Deterministic stand-in for the real LoRA classifier: routes on a
    keyword so graph/API tests can exercise both branches without loading torch."""

    def classify(self, text: str):
        from app.triage_classifier import TriageResult

        if "chest pain" in text.lower() or "can't breathe" in text.lower():
            return TriageResult(label="emergency", confidence=0.95)
        return TriageResult(label="routine", confidence=0.8)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Diabetes causes high blood sugar and increased thirst.", metadata={"topic": "Diabetes", "url": "https://medlineplus.gov/diabetes.html"}),
        Document(page_content="Migraines are recurring headaches often with light sensitivity.", metadata={"topic": "Migraine", "url": "https://medlineplus.gov/migraine.html"}),
        Document(page_content="The common cold causes a runny nose and mild sore throat.", metadata={"topic": "Common Cold", "url": "https://medlineplus.gov/commoncold.html"}),
    ]


@pytest.fixture
def fake_vector_store(tmp_path, fake_embeddings, sample_documents):
    from app.vector_store import build_or_load_vector_store

    return build_or_load_vector_store(
        str(tmp_path / "index"),
        embedding_model="fake",
        documents=sample_documents,
        embeddings=fake_embeddings,
    )


@pytest.fixture
def fake_triage_classifier():
    return FakeTriageClassifier()
