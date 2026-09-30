# SQLAlchemy 字段类型
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
# ORM 声明式基类，所有数据表模型继承它
from sqlalchemy.orm import DeclarativeBase
# pgvector 向量字段类型；Vector(768) 代表768‑维稠密向量
from pgvector.sqlalchemy import Vector
# 生成创建记录的UTC时间戳
from datetime import datetime


class Base(DeclarativeBase):
    """
    ORM模型的顶层基类
    所有数据库表(Document、Chunk)都继承Base
    init_db()中 Base.metadata.create_all(bind=engine) 依靠它扫描全部模型并建表
    """
    pass


class Document(Base):
    """
    源文档元数据表
    一对多关系：1条 Document(原始文档) → 多条 Chunk(文本切块)
    一条记录对应：一篇PubMed文献 / 一份PDF临床指南
    """
    __tablename__ = "documents"   # PostgreSQL中真实表名

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 自增主键，数据库内部id

    source = Column(String(50), nullable=False)
    # 文档来源标记：'pubmed' 医学文献 / 'pdf' 本地指南PDF

    doc_id = Column(String(100), unique=True, nullable=False)
    # 业务唯一文档ID，例如 pdf_heart_guideline、pubmed_39123456
    # unique=True：防止同一文档重复入库

    title = Column(Text)
    # 文档标题

    authors = Column(JSON)
    # JSON字段存储作者名字字符串列表 ["Alice","Bob"]

    abstract = Column(Text)
    # PubMed文献摘要；PDF指南可以为空

    publication_date = Column(String(20))
    # 发表日期字符串，例如 "2025‑Jan"

    journal = Column(String(500))
    # 期刊名称

    doi = Column(String(200))
    # 文献DOI编号

    full_text = Column(Text)
    # 文档完整正文（PDF解析出来的全文；PubMed一般无全文，仅摘要）

    meta = Column(JSON)
    # 通用JSON元数据存储：关键词、文件名、页码信息、pmid等杂项溯源字段

    created_at = Column(DateTime, default=datetime.utcnow)
    # 记录入库UTC时间戳


class Chunk(Base):
    """
    文本切块+向量存储表
    RAG语义检索阶段**主要查询这张表**
    每一行 = 一个TextChunk切块 + 768维Embedding向量
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 切块自增主键

    document_id = Column(Integer, nullable=False)
    # 外键，关联 documents.id，指向所属原始文档
    # ⚠️ 当前代码没有声明ForeignKey约束，只是逻辑外键

    doc_id = Column(String(100), nullable=False)
    # 反冗余字段(denormalized)：存放文档业务doc_id
    # 好处：检索Chunk时不需要JOIN documents表就拿到文档标识，查询更快

    chunk_index = Column(Integer, nullable=False)
    # 当前切块在原始文档里的序号 0,1,2…

    content = Column(Text, nullable=False)
    # 切块原始文本内容，检索命中之后展示给LLM

    embedding = Column(Vector(768))
    # pgvector向量字段 768维，NeuML/pubmedbert‑base‑embeddings输出向量

    token_count = Column(Integer)
    # 该切块token数量，来自TextChunker

    meta = Column(JSON)
    # 切块级别溯源元数据，页码等信息

    created_at = Column(DateTime, default=datetime.utcnow)
    # 切块入库时间戳

# -------- 新增：聊天会话两张表 --------
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    source = Column(String(20), nullable=False, default="patient")  # 'expert' / 'patient'，区分两版
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)