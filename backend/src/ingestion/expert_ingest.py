import structlog
from tqdm import tqdm
from typing import List

from src.config import settings
from src.database.connection import get_db, init_db
from src.database.models import Document, Chunk as ChunkModel
from src.ingestion.pubmed_fetcher import PubMedFetcher
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import Embedder

logger = structlog.get_logger(__name__)


DEFAULT_QUERIES = [
    "heart failure diagnosis management clinical guidelines",
    "type 2 diabetes mellitus treatment evidence based",
    "sepsis diagnosis criteria treatment protocol",
    "community acquired pneumonia antibiotic treatment",
    "hypertension cardiovascular risk reduction",
    "chronic kidney disease management progression",
    "acute myocardial infarction treatment outcomes",
    "stroke diagnosis thrombolysis management",
]


class ExpertIngestionPipeline:
    """专家版数据摄取 pipeline：PubMed API / PDF → Chunker → Embedder → pgvector。"""

    def __init__(self):
        self.fetcher = PubMedFetcher()
        self.parser = PDFParser()
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedder = Embedder()

    def _exists(self, db, doc_id: str) -> bool:
        return db.query(Document).filter(
            Document.doc_id == doc_id
        ).first() is not None

    def _ingest(self, db, doc_id, source, title, content,
                meta, chunk_meta, **doc_fields):
        """核心入库单元：存 Document + 切块 + 嵌入 + 存 Chunk。"""
        doc = Document(
            source=source, doc_id=doc_id, title=title,
            full_text=content, meta=meta, **doc_fields
        )
        db.add(doc)
        db.flush()  # 拿到自动生成的主键 doc.id

        chunks = self.chunker.chunk_text(content, doc_id, chunk_meta)
        if not chunks:
            return

        embeddings = self.embedder.embed_chunks(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            db.add(ChunkModel(
                document_id=doc.id,
                doc_id=doc_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embedding,
                token_count=chunk.token_count,
                meta=chunk.metadata,
            ))

        logger.debug("ingested_chunks", doc_id=doc_id, chunk_count=len(chunks))

    def run_pubmed(self, queries: List[str] = None, max_results: int = 50):
        if queries is None:
            queries = DEFAULT_QUERIES
        init_db()
        logger.info("pubmed_ingestion_start")

        all_articles = []
        for q in queries:
            pmids = self.fetcher.search(q, max_results)
            all_articles.extend(self.fetcher.fetch_articles(pmids))

        # 按 PMID 去重
        seen, unique = set(), []
        for a in all_articles:
            if a.pmid not in seen:
                seen.add(a.pmid)
                unique.append(a)

        logger.info("unique_articles", count=len(unique))
        skipped = 0

        with get_db() as db:
            for article in tqdm(unique, desc="PubMed ingestion"):
                doc_id = f"pubmed_{article.pmid}"
                if self._exists(db, doc_id):
                    skipped += 1
                    continue
                self._ingest(
                    db=db, doc_id=doc_id, source="pubmed",
                    title=article.title,
                    content=f"{article.title}\n\n{article.abstract}",
                    meta={"keywords": article.keywords, "pmid": article.pmid},
                    chunk_meta={"source": "pubmed", "pmid": article.pmid,
                                "journal": article.journal},
                    authors=article.authors,
                    abstract=article.abstract,
                    publication_date=article.publication_date,
                    journal=article.journal,
                    doi=article.doi,
                )

        logger.info("pubmed_ingestion_done", skipped=skipped)

    def run_pdf(self):
        logger.info("pdf_ingestion_start")

        documents = self.parser.parse_directory()
        skipped = 0

        with get_db() as db:
            for doc in tqdm(documents, desc="PDF ingestion"):
                if self._exists(db, doc.doc_id):
                    skipped += 1
                    continue
                self._ingest(
                    db=db, doc_id=doc.doc_id, source="pdf",
                    title=doc.title, content=doc.content,
                    meta=doc.metadata,
                    chunk_meta={"source": "pdf",
                                "filename": doc.metadata.get("filename", "")},
                )

        logger.info("pdf_ingestion_done", skipped=skipped)

    def run(self, queries: List[str] = None):
        """完整 pipeline：init DB → PubMed → PDF。"""
        logger.info("ingestion_init_db")
        init_db()
        self.run_pubmed(queries or DEFAULT_QUERIES)
        self.run_pdf()
        logger.info("ingestion_complete")
