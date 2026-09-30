"""L1 专家版摄取测试(离线,mock db / embedder)。"""
from src.ingestion.expert_ingest import ExpertIngestionPipeline
from src.ingestion.chunker import TextChunker
from src.database.models import Chunk


class _FakeQuery:
    def filter(self, *a, **k):
        return self

    def first(self):
        return None  # 模拟 doc 不存在


class _FakeSession:
    def __init__(self):
        self.added_docs = []
        self.added_chunks = []

    def query(self, *cols):
        return _FakeQuery()

    def add(self, obj):
        if isinstance(obj, Chunk):
            self.added_chunks.append(obj)
        else:
            self.added_docs.append(obj)

    def flush(self):
        pass


class _FakeEmbedder:
    def embed_chunks(self, chunks, batch_size=32):
        return [[0.0] * 768 for _ in chunks]


def _make_pipeline():
    # 用 __new__ 跳过 __init__(避免加载真实 Embedder / PDFParser / PubMedFetcher)
    p = ExpertIngestionPipeline.__new__(ExpertIngestionPipeline)
    p.chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    p.embedder = _FakeEmbedder()
    return p


def test_ingest_stores_document_and_chunks():
    p = _make_pipeline()
    db = _FakeSession()
    content = " ".join(f"Medical sentence number {i} about treatment." for i in range(30))
    p._ingest(db, doc_id="pubmed_1", source="pubmed", title="T",
              content=content, meta={}, chunk_meta={})
    assert len(db.added_docs) == 1
    assert db.added_docs[0].doc_id == "pubmed_1"
    assert len(db.added_chunks) >= 1


def test_exists_returns_false_when_absent():
    p = _make_pipeline()
    db = _FakeSession()
    assert p._exists(db, "pubmed_1") is False


# ============ run_pubmed 的 PMID 去重 ============
class _FakeDB:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, *a):
        pass


class _Article:
    def __init__(self, pmid):
        self.pmid = pmid
        self.title = "T"
        self.abstract = "A"
        self.authors = []
        self.publication_date = "2025"
        self.journal = "J"
        self.doi = None
        self.keywords = []


class _FakeFetcher:
    def __init__(self, articles):
        self._articles = articles

    def search(self, q, max_results):
        return [a.pmid for a in self._articles]

    def fetch_articles(self, pmids):
        return self._articles


def test_run_pubmed_dedups_pmids(monkeypatch):
    import src.ingestion.expert_ingest as ei
    p = _make_pipeline()
    p.fetcher = _FakeFetcher([_Article("1"), _Article("1"), _Article("2")])  # pmid 1 重复
    db = _FakeSession()
    monkeypatch.setattr(ei, "init_db", lambda: None)
    monkeypatch.setattr(ei, "get_db", lambda: _FakeDB(db))
    p.run_pubmed(queries=["diabetes"], max_results=10)
    # PMID 去重后只 ingest 2 篇(1 和 2)
    assert len(db.added_docs) == 2
