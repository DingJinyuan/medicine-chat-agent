# 正则表达式库，用于句子切分
import re
# tiktoken OpenAI官方分词库，精确统计token数量
import tiktoken
# 类型注解
from typing import List, Dict, Any
# Pydantic结构化数据模型，封装切片后的文本块
from pydantic import BaseModel
# 结构化日志
import structlog

logger = structlog.get_logger(__name__)


class TextChunk(BaseModel):
    """
    文本切块之后的单元数据模型
    RAG知识库中存入向量库的最小存储对象
    """
    doc_id: str                # 归属的原始文档ID，用来溯源
    chunk_index: int           # 当前切片在文档内的序号
    content: str               # 切块的文本内容
    token_count: int           # 当前切块token数量
    metadata: Dict[str, Any] = {}  # 附加元数据：页码、文件名等溯源信息

class TextChunker:
    """
    基于Token、带重叠窗口的文档切分器

    为什么选用Token切分而不是字符切分？
       嵌入模型的限制单位是Token，不是字符。
       使用Token计数可以得到精准、稳定的块大小，不会超出Embedding模型上限。

    为什么设置重叠overlap？
       避免上下文信息在切块边界丢失。
       如果一句话刚好被两块切开，重叠机制会让这句话同时出现在相邻两个块中，
       提升检索召回完整相关段落的成功率。
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        初始化切分器
        :param chunk_size: 每个文本块最大token数
        :param chunk_overlap: 相邻两块之间重叠的token数量
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # cl100k_base：GPT‑3.5 / GPT‑4 使用的分词器
        # 作为绝大多数大模型token计数的近似参考 把文本切成大模型看得懂的 **token（令牌）**。
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """统计一段文本对应的token数量"""
        return len(self.tokenizer.encode(text))

    def _split_sentences(self, text: str) -> List[str]:
        """
        句子边界分割函数
        正则规则：仅在句号/感叹号/问号之后、下一个单词首字母为大写时才切分句子
        目的：防止把医学缩写 Dr.、mg.、kg. 错误识别为句子结束标记
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # 去除首尾空白字符，过滤空字符串
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
            self,
            text: str,
            doc_id: str,
            metadata: Dict[str, Any] = {}
    ) -> List[TextChunk]:
        """
        主切分函数：将长文档文本切分成多个带重叠的文本块
        :param text: 待切分完整文档文本
        :param doc_id: 原始文档唯一id
        :param metadata: 溯源元数据（页码、文件名等）
        :return: TextChunk对象列表
        """
        # 第一步：把全文拆分成独立句子列表
        sentences = self._split_sentences(text)
        chunks = []
        current = []  # 正在组装的当前块句子缓存
        current_tokens = 0  # 当前缓存累计token数
        chunk_index = 0  # 切块编号计数器

        for sentence in sentences:
            # 计算当前这句句子的token数量
            s_tokens = self._count_tokens(sentence)

            # 单句本身超过块上限：先把当前缓存落盘，再按字符硬切超长句，防止单块无限膨胀
            if s_tokens > self.chunk_size:
                if current:
                    chunks.append(TextChunk(
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        content=" ".join(current),
                        token_count=current_tokens,
                        metadata=metadata
                    ))
                    chunk_index += 1
                    current, current_tokens = [], 0
                for i in range(0, len(sentence), self.chunk_size):
                    piece = sentence[i:i + self.chunk_size].strip()
                    if piece:
                        chunks.append(TextChunk(
                            doc_id=doc_id,
                            chunk_index=chunk_index,
                            content=piece,
                            token_count=self._count_tokens(piece),
                            metadata=metadata
                        ))
                        chunk_index += 1
                continue

            # 判断：加入该句子后超过块大小上限，先把当前缓存生成一个切块
            if current_tokens + s_tokens > self.chunk_size and current:
                chunks.append(TextChunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    content=" ".join(current),
                    token_count=current_tokens,
                    metadata=metadata
                ))
                chunk_index += 1

                # ========== 生成重叠部分 ==========
                overlap, overlap_tokens = [], 0
                # 倒序遍历当前块的句子，从末尾往前收集句子直到达到overlap上限
                for s in reversed(current):
                    t = self._count_tokens(s)
                    if overlap_tokens + t <= self.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_tokens += t
                    else:
                        break
                # 重叠句子作为下一个新块的开头
                current = overlap
                current_tokens = overlap_tokens

            # 将当前句子加入缓存
            current.append(sentence)
            current_tokens += s_tokens

        # 循环结束，处理剩余未生成块的最后一部分文本
        if current:
            chunks.append(TextChunk(
                doc_id=doc_id,
                chunk_index=chunk_index,
                content=" ".join(current),
                token_count=current_tokens,
                metadata=metadata
            ))

        logger.debug("chunked", doc_id=doc_id, chunk_count=len(chunks))
        return chunks
