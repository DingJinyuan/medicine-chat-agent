from typing import List, Dict

import structlog
from sentence_transformers import CrossEncoder

from src.config import settings

logger = structlog.get_logger(__name__)


class Reranker:
    """Cross-encoder 精排器：对 (查询, 候选块) 逐对打分，重排后返回 top-k。

    懒加载：首次 rerank 时才加载模型；加载失败则降级为不精排，
    直接按粗召回（RRF）顺序返回，保证流水线不中断。
    """

    def __init__(self):
        self.model = None

    def _load(self):
        if self.model is None:
            logger.info("reranker_loading", model=settings.rerank_model)
            logger.info("reranker_first_run_download")
            self.model = CrossEncoder(settings.rerank_model)
            logger.info("reranker_ready")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """对候选块按相关性重新打分排序，返回 top_k；失败降级为原顺序"""
        if not chunks:
            return []
        try:
            self._load()
            pairs = [(query, c["content"]) for c in chunks]
            scores = self.model.predict(pairs)
            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            # 回写 cross-encoder 精排分数，否则下游拿到的仍是 RRF 粗排分数
            result = []
            for c, s in ranked[:top_k]:
                c = dict(c)
                c["score"] = float(s)
                result.append(c)
            return result
        except Exception as e:
            logger.warning("reranker_failed_skip", error=str(e))
            return chunks[:top_k]
