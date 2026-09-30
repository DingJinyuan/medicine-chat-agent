# HTTP网络请求库，调用NCBI‑E‑utilities接口
import requests
# 用于API请求限流休眠，遵守PubMed访问速率限制
import time
# Python内置XML解析模块，解析efetch返回的XML文献数据
import xml.etree.ElementTree as ET
# 类型注解
from typing import List, Optional
import structlog

logger = structlog.get_logger(__name__)
# Pydantic结构化模型，封装单篇PubMed文献信息
from pydantic import BaseModel
# 读取项目全局配置（API‑Key、批次大小等）
from src.config import settings
#爬虫详情https://www.ncbi.nlm.nih.gov/books/NBK25500/#chapter1.Searching_a_Database
"""
<PubmedArticleSet>

    <!-- 每一篇文献 = 1个 PubmedArticle 节点 -->
    <PubmedArticle>

        <!-- 1. PMID 文献编号 -->
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">38901234</PMID>

            <!-- 2. 文章核心信息 -->
            <Article PubModel="Print">
                <!-- 标题 -->
                <ArticleTitle>Diabetes treatment in adults ...</ArticleTitle>

                <!-- 结构化摘要，就是你解析的 AbstractText -->
                <Abstract>
                    <AbstractText Label="BACKGROUND">Diabetes is a chronic disease…</AbstractText>
                    <AbstractText Label="METHODS">We enrolled 300 patients…</AbstractText>
                    <AbstractText Label="RESULTS">Significant improvement was observed…</AbstractText>
                    <AbstractText Label="CONCLUSION">The therapy is effective…</AbstractText>
                </Abstract>

                <!-- 作者列表 -->
                <AuthorList>
                    <Author>
                        <LastName>Smith</LastName>
                        <ForeName>John A</ForeName>
                    </Author>
                    <Author>
                        <CollectiveName>World‑Health‑Organization</CollectiveName>
                    </Author>
                </AuthorList>

                <!-- 期刊信息 -->
                <Journal>
                    <Title>Journal of Clinical Medicine</Title>
                    <JournalIssue>
                        <PubDate>
                            <Year>2025</Year>
                            <Month>Jan</Month>
                            <Day>15</Day>
                        </PubDate>
                    </JournalIssue>
                </Journal>
            </Article>

            <!-- 关键词 -->
            <KeywordList>
                <Keyword>Diabetes Mellitus</Keyword>
                <Keyword>Insulin therapy</Keyword>
            </KeywordList>

            <!-- Mesh 医学主题词（另外一套标签，你当前代码没读） -->
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>Diabetes Mellitus</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
        </MedlineCitation>

        <!-- DOI、PMC等各种编号放在 ArticleIdList -->
        <PubmedData>
            <ArticleIdList>
                <ArticleId IdType="pubmed">38901234</ArticleId>
                <ArticleId IdType="doi">10.1234/jcm.2024‑1234</ArticleId>
                <ArticleId IdType="pmc">PMC12345678</ArticleId>
            </ArticleIdList>
        </PubmedData>

    </PubmedArticle>

    <!-- 第二篇文献继续再来一个 <PubmedArticle> -->
    <PubmedArticle>
        ...
    </PubmedArticle>

</PubmedArticleSet>
"""
class PubMedArticle(BaseModel):
    """
    PubMed单篇文献结构化数据模型
    """
    pmid: str                 # PubMed文献唯一编号
    title: str                # 文献标题
    abstract: str             # 摘要正文
    authors: List[str]        # 作者姓名列表
    publication_date: str     # 发表日期 格式 YYYY‑MM
    journal: str              # 期刊名称
    doi: Optional[str] = None # 文献DOI编号，可空
    keywords: List[str] = []  # 关键词列表

class PubMedFetcher:
    """
    通过NCBI E‑utilities API 抓取PubMed文献摘要数据
    内置速率限制、分页批量拉取、XML解析容错处理

    NCBI官方接口速率约束：
      无API密钥 : 最多 3次请求 / 秒
      配置API密钥: 最多10次请求 / 秒（免费申请）
    """
    # NCBI E‑utilities 接口根地址
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self):
        """初始化请求会话，读取密钥，配置请求延迟、请求头"""
        self.api_key = settings.ncbi_api_key
        # 根据是否有密钥设置休眠延迟，限流防封禁
        self.delay = 0.1 if self.api_key else 0.35
        # 创建持久Session，复用TCP连接，提升请求效率
        self.session = requests.Session()
        # 设置UA标识，告知NCBI请求来源为你的临床Agent项目
        self.session.headers.update({"User-Agent": "Medicine-ai/2.0 (research)"})

    def search(self, query: str, max_results: int = 100) -> List[str]:
        """

        PubMed文献检索，仅返回匹配文献的PMID编号列表
        :param query: PubMed检索字符串（支持PubMed高级检索语法）
        :param max_results: 需要返回的最大文献数量
        :return: pmid字符串列表
        """
        logger.info("pubmed_searching", query=query, max=max_results)

        params = {
            "db": "pubmed", "term": query,
            "retmax": max_results, "retmode": "json",
        }
        # 如果配置了密钥，则携带密钥参数发送请求
        if self.api_key:
            params["api_key"] = self.api_key

        # 调用esearch搜索接口
        resp = self.session.get(f"{self.BASE_URL}esearch.fcgi", params=params)
        resp.raise_for_status()  # HTTP状态码非200直接抛出异常
        data = resp.json()

        pmids = data["esearchresult"]["idlist"] #就是你要的 PMID 编号列表
        total = data["esearchresult"]["count"] #数据库里总共搜到多少篇相关文献
        logger.info("pubmed_found", total=total, fetching=len(pmids))
        return pmids

    def fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        根据PMID编号列表，批量拉取文献完整元数据与摘要
        :param pmids: pmid编号列表
        :return: PubMedArticle结构化文献对象列表
        """
        articles = []
        batch_size = settings.pubmed_batch_size

        # 按批次切分pmid列表分批请求
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            b_num = i // batch_size + 1
            b_total = (len(pmids) + batch_size - 1) // batch_size
            logger.info("pubmed_fetching_batch", batch=b_num, total=b_total)
            #**结构化摘要（Background/Methods）**必须用 XML
            params = {
                "db": "pubmed", "id": ",".join(batch),
                "rettype": "xml", "retmode": "xml",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            # 请求efetch接口获取XML格式文献详情
            resp = self.session.get(f"{self.BASE_URL}efetch.fcgi", params=params)
            resp.raise_for_status()
            # 解析当前批次XML，转为文献对象，追加结果列表
            articles.extend(self._parse_xml(resp.text))
            # 请求后休眠，遵守NCBI限流规则
            time.sleep(self.delay)

        logger.info("pubmed_fetched", count=len(articles))
        return articles

    def _parse_xml(self, xml_text: str) -> List[PubMedArticle]:
        """
        批量解析efetch返回的完整XML字符串
        :param xml_text: 原始XML响应文本
        :return: PubMedArticle列表
        """
        articles = []
        root = ET.fromstring(xml_text)  #把一大段字符串 XML → 转换成可以查找遍历的 XML 树对象 `root`。
        # XPath遍历所有PubmedArticle节点 .//当前节点自动穿透
        for elem in root.findall(".//PubmedArticle"):
            try:
                a = self._parse_single(elem)
                if a:
                    articles.append(a)
            except Exception as e:
                # 单篇文献解析失败不会中断整个批次，跳过损坏文献
                logger.warning("pubmed_skip_malformed", error=str(e))
        return articles
    def _parse_single(self, elem) -> Optional[PubMedArticle]:
        """
        解析单一篇PubmedArticle XML节点，转为结构化对象
        :param elem: 单篇文献XML元素节点
        :return: PubMedArticle；无PMID /无摘要返回None跳过
        """
        pmid_elem = elem.find(".//PMID")
        if pmid_elem is None:
            return None

        # ==========解析结构化摘要（Background/Methods/Results等带标签摘要）==========
        abstract_parts = elem.findall(".//AbstractText")
        abstract = " ".join([
            f"{p.get('Label')}: {p.text or ''}" if p.get('Label')
            else (p.text or "")
            for p in abstract_parts
        ]).strip()

        if not abstract:
            return None  # 没有摘要的文献直接跳过，不入库

        # ==========解析作者列表==========
        authors = []
        for author in elem.findall(".//Author"):
            last = author.findtext("LastName", "")
            first = author.findtext("ForeName", "")
            name = f"{first} {last}".strip() if first else last
            if name:
                authors.append(name)

        # ==========解析发表日期==========
        pub_date = elem.find(".//PubDate")
        year = pub_date.findtext("Year", "") if pub_date is not None else ""
        month = pub_date.findtext("Month", "") if pub_date is not None else ""

        # ==========提取DOI编号==========
        doi = next(
            (e.text for e in elem.findall(".//ArticleId")
             if e.get("IdType") == "doi"), None
        )

        return PubMedArticle(
            pmid=pmid_elem.text,
            title=(elem.findtext(".//ArticleTitle") or "Unknown").strip(),
            abstract=abstract,
            authors=authors,
            publication_date=f"{year}-{month}".strip("-"),
            journal=elem.findtext(".//Journal/Title", "Unknown Journal"),
            doi=doi,
            keywords=[kw.text for kw in elem.findall(".//Keyword") if kw.text],
        )