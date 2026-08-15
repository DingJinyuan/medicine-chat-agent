"""Fetch and chunk MedlinePlus health-topic summaries into LangChain Documents.

MedlinePlus (National Library of Medicine, part of NIH) content is US
government work and public domain -- this sidesteps the licensing ambiguity
of third-party medical QA datasets, where answer text is often stripped for
non-government sources.

API docs: https://medlineplus.gov/webservices.html
"""
from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# MedlinePlus查询接口地址
MEDLINEPLUS_ENDPOINT = "https://wsearch.nlm.nih.gov/ws/query"

# Curated set of common condition/topic terms -- broad enough for a demo
# triage/RAG corpus without trying to cover all of medicine.
DEFAULT_TOPICS = [
    "diabetes", "hypertension", "asthma", "migraine", "influenza", "common cold",
    "urinary tract infection", "anxiety", "depression", "back pain", "allergies",
    "pneumonia", "bronchitis", "sinusitis", "gastroenteritis", "acid reflux",
    "irritable bowel syndrome", "constipation", "diarrhea", "food poisoning",
    "strep throat", "ear infection", "conjunctivitis", "eczema", "psoriasis",
    "acne", "urinary incontinence", "kidney stones", "gallstones", "anemia",
    "hypothyroidism", "hyperthyroidism", "high cholesterol", "obesity",
    "osteoarthritis", "rheumatoid arthritis", "gout", "osteoporosis",
    "carpal tunnel syndrome", "sciatica", "tendinitis", "concussion",
    "sprains and strains", "fractures", "chest pain", "heart attack",
    "stroke", "heart failure", "arrhythmia", "deep vein thrombosis",
    "chronic obstructive pulmonary disease", "sleep apnea", "insomnia",
    "shingles", "chickenpox", "measles", "mononucleosis", "hepatitis",
    "hiv and aids", "sexually transmitted diseases", "menstrual cramps",
    "polycystic ovary syndrome", "menopause", "pregnancy", "morning sickness",
    "erectile dysfunction", "prostate enlargement", "kidney disease",
    "chronic fatigue syndrome", "fibromyalgia", "vertigo", "tinnitus",
    "seasonal affective disorder", "adhd", "autism spectrum disorder",
    "panic disorder", "bipolar disorder", "eating disorders",
    "substance use disorder", "alcohol use disorder", "smoking and tobacco",
    "skin cancer", "breast cancer", "colon cancer", "lung cancer",
    "prostate cancer", "melanoma", "food allergy", "lactose intolerance",
    "celiac disease", "appendicitis", "hemorrhoids", "varicose veins",
    "cellulitis", "athlete's foot", "ringworm", "head lice", "scabies",
    "nosebleeds", "dehydration", "heat exhaustion and heat stroke",
    "frostbite", "hypothermia", "food safety",
]


def _strip_html(raw: str) -> str:
    """
    私有工具函数：清洗接口返回富文本

    1. 解析HTML转义字符
    2. 移除全部HTML标签
    3. 去除首尾空白

    Args:
        raw: 带有HTML标签与转义符的原始字符串

    Returns:
        清洗完成的纯文本
    """
    #解析 HTML 转义实体   &lt; → <；&gt; → >；&amp; → &；&quot; → "
    unescaped = html.unescape(raw)
    #正则删除所有 HTML 标签
    return re.sub(r"<[^>]+>", "", unescaped).strip()


def fetch_medlineplus_topic(term: str, client: httpx.Client) -> dict | None:
    """
    抓取单个健康主题的标题与完整摘要

    Args:
        term: 检索关键词（疾病/症状英文名称）
        client: 复用的httpx客户端，减少TCP连接开销

    Returns:
        dict: {"topic":主题名, "url":官方链接, "summary":清洗后摘要}
        None: 查询无结果、标题或摘要缺失时返回
    """
    #client：外部传入复用的httpx.Client，复用 TCP 连接，批量抓取时减少握手开销
    #db=healthTopics：指定查询健康主题数据库 term：疾病英文关键词，例如migraine   retmax=1：只拿第一条匹配结果，不需要多页
    resp = client.get(MEDLINEPLUS_ENDPOINT, params={"db": "healthTopics", "term": term, "retmax": 1})
    resp.raise_for_status()  #HTTP 状态码 >=400 直接抛出httpx.HTTPError
    """
    <nlmSearchResult count="1">
  <document url="https://medlineplus.gov/migraine.html">
    <content name="title"><b>Migraine</b></content>
    <content name="FullSummary">
      <p>Migraine is a type of headache...</p>
    </content>
    <content name="organization">MedlinePlus</content>
    <content name="date">2025‑03‑12</content>
    <!-- 还有很多其他content字段 -->
  </document>
</nlmSearchResult>
"""
    root = ET.fromstring(resp.text)  #把接口返回 XML 原始字符串，解析成 ElementTree 节点树
    doc = root.find(".//document")   #在整个 XML 树任意深度查找<document>节点
    """
    无关查询结果
    <nlmSearchResult count="0">
  <script/>
</nlmSearchResult>
    """
    if doc is None:
        return None

    title = None
    summary = None
    for content in doc.findall("content"):
        name = content.get("name")
        if name == "title" and title is None:
            title = _strip_html(content.text or "")
        elif name == "FullSummary" and summary is None:
            summary = _strip_html(content.text or "")

    if not title or not summary:
        return None

    return {"topic": title, "url": doc.get("url", ""), "summary": summary}


def fetch_all_topics(terms: list[str] = DEFAULT_TOPICS) -> list[dict]:
    """
        批量抓取全部预设健康主题
        单个主题发生网络异常、XML解析异常直接跳过，不中断整体任务。

        Args:
            terms: 待抓取关键词列表，默认使用DEFAULT_TOPICS

        Returns:
            成功抓取的主题字典列表
        """

    results: list[dict] = []
    with httpx.Client(timeout=20.0) as client:
        for term in terms:
            try:
                #dict字典：查询成功拿到标题 + 摘要 → 添加进结果列表     None：没找到 document /title 或 summary 为空（业务空结果，不是异常），不会 append，直接丢弃。
                topic = fetch_medlineplus_topic(term, client)
            except (httpx.HTTPError, ET.ParseError):  #httpx.HTTPError 包含超时、连接失败、4xx/5xx ET.ParseError：接口返回畸形 XML，ET.fromstring()解析失败
                continue
            if topic:
                results.append(topic)
    return results


def load_or_fetch_corpus(cache_path: str, terms: list[str] = DEFAULT_TOPICS) -> list[dict]:
    """
        带本地JSON缓存的语料加载入口
        缓存文件存在直接读取本地文件；不存在则线上抓取并持久化缓存，减少API调用。
        Args:
            cache_path: 本地缓存json文件路径
            terms: 待抓取关键词列表

        Returns:
            原始健康主题数据字典列表
        """
    path = Path(cache_path)
    if path.exists():
        return json.loads(path.read_text())

    topics = fetch_all_topics(terms)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(topics, indent=2))
    return topics


def build_documents(topics: list[dict], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    """
        将原始健康摘要切分，构建LangChain标准Document对象，可直接向量化入库
        Args:
            topics: 原始抓取得到的主题字典列表
            chunk_size: 单个文本块最大字符数
            chunk_overlap: 分片重叠字符数，保障上下文连续性

        Returns:
            Document列表，metadata携带topic主题、url官方溯源链接
        """
    #LangChain 优先按：段落 → 换行 → 句子 → 单词 递归切割  尽量不破坏语义（比粗暴按字数切割强太多）
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents: list[Document] = []
    for t in topics:
        for chunk in splitter.split_text(t["summary"]):
            documents.append(Document(page_content=chunk,
                                      metadata={"topic": t["topic"],
                                                "url": t["url"]}))
    return documents
