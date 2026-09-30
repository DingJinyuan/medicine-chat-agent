# Sentence‑Transformers 向量嵌入框架，加载预训练嵌入模型
from sentence_transformers import SentenceTransformer
# 类型注解：列表
from typing import List
# 结构化日志
import structlog
# tqdm 进度条工具，可视化批量向嵌入进度
from tqdm import tqdm

logger = structlog.get_logger(__name__)
# 项目全局配置文件，读取模型名称、向量维度等环境参数
from src.config import settings
# 导入切块数据模型 TextChunk（上一段代码定义的切块实体）
from src.ingestion.chunker import TextChunk


class Embedder:
    """
    使用生物医学专用模型生成稠密向量Embedding

    选用模型: NeuML/pubmedbert-base-embeddings
      - 在PubMed生物医学摘要数据集上训练完成
      - 输出768维向量
      - 在临床医疗文本任务上效果显著优于通用嵌入模型
      - 完全本地运行、免费开源，无任何接口调用费用

    向量开启L2‑归一化：余弦相似度 = 点积运算
    可以加速 pgvector 向量数据库近似最近邻ANN检索查询速度。
    """

    def __init__(self):
        """初始化：加载生物医学嵌入模型到内存/GPU"""
        logger.info("embedder_loading", model=settings.embedding_model)
        logger.info("embedder_first_run_download")
        # 从HuggingFace加载预训练sentence‑transformer模型
        self.model = SentenceTransformer(settings.embedding_model)
        logger.info("embedder_ready", dim=settings.embedding_dimension)

    def embed_chunks(
            self, chunks: List[TextChunk], batch_size: int = 32
    ) -> List[List[float]]:
        """
        批量对切块文本生成向量嵌入
        :param chunks: TextChunk切块对象列表
        :param batch_size: 一次送入模型的文本批次大小，调大可以提速，占用更多显存
        :return: 向量列表，每个向量是768维浮点数列表
        """
        # 提取所有切块内的纯文本内容
        texts = [chunk.content for chunk in chunks]
        all_embeddings = []

        # 以步长 batch_size 循环分批处理，tqdm展示进度条
        #range(开始,结束,步长)   tqdm 是进度条工具。  leave=False 进度条跑完之后**自动消失**
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,  # L2归一化向量
                show_progress_bar=False,  # 关闭内置进度条，使用外层tqdm
                convert_to_numpy=True,  # 返回numpy数组
            )
            # numpy数组转为Python list存入结果  chunks[0] ↔ all_embeddings[0]
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        将用户检索Query生成查询向量，用于向量库相似度召回
        :param query: 用户输入检索问题字符串
        :return: 单条768维浮点数向量
        """
        return self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()


