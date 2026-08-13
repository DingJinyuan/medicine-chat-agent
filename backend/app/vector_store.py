"""LangChain-based retrieval: a FAISS vectorstore over fastembed embeddings.

fastembed (not sentence-transformers) keeps the embedding dependency small.
Here it's wrapped behind LangChain's
`Embeddings` interface so the rest of the pipeline (the LangGraph `retrieve`
node) can use a standard LangChain retriever instead of a hand-rolled search
method.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


FASTEMBED_CACHE_DIR = str(Path(__file__).resolve().parent.parent / ".fastembed_cache")


class FastEmbedEmbeddings(Embeddings):
    """Lazy wrapper around fastembed so import/model-download cost is paid once.

    Explicit `cache_dir` outside /tmp -- fastembed defaults to a /tmp-based
    cache, but some platforms (Render included) mount a fresh, empty /tmp at
    container runtime, hiding whatever got baked into the image there during
    the Docker build. The embedder loads lazily on first real query, so this
    silently breaks only the first chat request, not startup/health checks.
    """

    def __init__(self, model_name: str, cache_dir: str = FASTEMBED_CACHE_DIR):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(v) for v in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._load().embed([text]))[0].tolist()


def build_or_load_vector_store(
    index_path: str,
    embedding_model: str,
    documents: list[Document] | None = None,
    embeddings: Embeddings | None = None,
):
    """Load a persisted FAISS index if present, else build one from `documents`.

    `embeddings` can be injected (e.g. a fast hash-based fake) so tests don't
    have to download the real fastembed model.
    """
    from langchain_community.vectorstores import FAISS

    embeddings = embeddings or FastEmbedEmbeddings(embedding_model)
    path = Path(index_path)

    if (path / "index.faiss").exists():
        return FAISS.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True
        )

    if not documents:
        raise FileNotFoundError(
            f"no vector index at {index_path} and no documents provided to build one"
        )

    store = FAISS.from_documents(documents, embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    return store
