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

# FastEmbed模型本地缓存目录，避免容器临时目录丢失预下载模型
FASTEMBED_CACHE_DIR = str(Path(__file__).resolve().parent.parent / ".fastembed_cache")


class FastEmbedEmbeddings(Embeddings):
    """
    FastEmbed 延迟加载包装器，适配 LangChain Embeddings 标准接口
    程序启动的时候，不去加载 Embedding 大模型；等到第一次真正要算向量（第一次调用 embed 推理）的时候，才把模型加载进内存
    采用懒加载策略：首次调用嵌入推理时才加载模型，避免启动耗时、避免健康检查拖慢服务。
    自定义持久化缓存目录，规避容器平台(Render)每次重启清空 /tmp 默认缓存导致模型重复下载的问题。
    """

    def __init__(self, model_name: str, cache_dir: str = FASTEMBED_CACHE_DIR):
        """
                初始化 FastEmbed 嵌入器（仅保存配置，不加载模型）

                Args:
                    model_name: FastEmbed 嵌入模型名称
                    cache_dir: 模型权重本地缓存路径
                """
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
        #.embed(texts) FastEmbed 原生方法    返回值：生成器（generator），不是 list  流式产出向量
        #必须用 list() 消费这个生成器，才能拿到里面的 numpy 向量数组
        return [list(v) for v in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._load().embed([text]))[0].tolist()


def build_or_load_vector_store(
    index_path: str,
    embedding_model: str,
    documents: list[Document] | None = None,
    embeddings: Embeddings | None = None,
):
    """
       加载或构建 FAISS 向量索引
       优先读取本地持久化索引；无索引则根据传入文档新建并落地保存。
       支持依赖注入 embeddings，方便单元测试使用假向量模型，无需下载真实权重。

       Args:
           index_path: FAISS 索引本地存储路径
           embedding_model: FastEmbed 模型名称（默认构造嵌入器使用）
           documents: 用于新建索引的 LangChain Document 列表
           embeddings: 可注入自定义 Embeddings 实例，用于测试解耦

       Returns:
           可直接检索的 FAISS 向量库实例
       FAISS 本地持久化会输出两个文件：
            index.faiss：faiss 库自己的二进制格式，只存向量（浮点数组）
            index.pkl：pickle 文件，存 Document 对象、page_content、metadata（你的 topic、url 元数据）

       """
    from langchain_community.vectorstores import FAISS
    #外部传入了 embeddings（测试用 fake）→ 用外部的     如果没传 → 自动初始化真实 FastEmbed 嵌入模型
    embeddings = embeddings or FastEmbedEmbeddings(embedding_model)
    path = Path(index_path)
    #allow_dangerous_deserialization=True FAISS 本地加载需要反序列化 pickle 文件，langchain 强制要求显式开启。我们读取自己保存的本地文件，是安全的
    if (path / "index.faiss").exists():
        return FAISS.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True
        )

    if not documents:
        raise FileNotFoundError(
            f"no vector index at {index_path} and no documents provided to build one"
        )
    #循环每一个 Document，调用 embeddings 把文本块转为向量，构建 FAISS 内存索引对象 store
    store = FAISS.from_documents(documents, embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    return store
