"""L1 PubMed 抓取测试(离线:测 XML 解析纯函数 + mock 网络)。"""
import xml.etree.ElementTree as ET

from src.ingestion.pubmed_fetcher import PubMedFetcher

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>38901234</PMID>
      <Article>
        <ArticleTitle>Diabetes treatment in adults</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Diabetes is a chronic disease.</AbstractText>
          <AbstractText Label="CONCLUSION">Therapy is effective.</AbstractText>
        </Abstract>
        <AuthorList><Author><LastName>Smith</LastName><ForeName>John</ForeName></Author></AuthorList>
        <Journal>
          <Title>Journal of Clinical Medicine</Title>
          <JournalIssue><PubDate><Year>2025</Year><Month>Jan</Month></PubDate></JournalIssue>
        </Journal>
      </Article>
      <KeywordList><Keyword>Diabetes Mellitus</Keyword></KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList><ArticleId IdType="doi">10.1234/jcm.2024-1234</ArticleId></ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


class TestParseXML:
    def test_parse_xml_returns_article(self):
        fetcher = PubMedFetcher()
        articles = fetcher._parse_xml(SAMPLE_XML)
        assert len(articles) == 1
        a = articles[0]
        assert a.pmid == "38901234"
        assert a.title == "Diabetes treatment in adults"
        assert "BACKGROUND: Diabetes is a chronic disease." in a.abstract
        assert a.authors == ["John Smith"]
        assert a.journal == "Journal of Clinical Medicine"
        assert a.publication_date == "2025-Jan"
        assert a.doi == "10.1234/jcm.2024-1234"
        assert a.keywords == ["Diabetes Mellitus"]

    def test_parse_xml_skips_article_without_pmid(self):
        xml = ("<PubmedArticleSet><PubmedArticle><Article>"
               "<ArticleTitle>No PMID</ArticleTitle>"
               "<Abstract><AbstractText>text</AbstractText></Abstract>"
               "</Article></PubmedArticle></PubmedArticleSet>")
        fetcher = PubMedFetcher()
        assert fetcher._parse_xml(xml) == []

    def test_parse_single_no_abstract_returns_none(self):
        elem = ET.fromstring(
            "<PubmedArticle><MedlineCitation><PMID>123</PMID></MedlineCitation></PubmedArticle>"
        )
        fetcher = PubMedFetcher()
        assert fetcher._parse_single(elem) is None


class TestSearch:
    def test_search_returns_pmids(self, monkeypatch):
        fetcher = PubMedFetcher()

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"esearchresult": {"idlist": ["1", "2"], "count": "2"}}

        monkeypatch.setattr(fetcher.session, "get", lambda url, params=None: _FakeResp())
        assert fetcher.search("diabetes", max_results=10) == ["1", "2"]
