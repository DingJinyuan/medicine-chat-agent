import httpx

from app.ingestion import _strip_html, build_documents, fetch_medlineplus_topic

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <document rank="1" url="https://medlineplus.gov/diabetes.html">
    <content name="title">&lt;span class="qt0"&gt;Diabetes&lt;/span&gt;</content>
    <content name="FullSummary">&lt;p&gt;What is &lt;span class="qt0"&gt;diabetes&lt;/span&gt;? It is a disease.&lt;/p&gt;</content>
  </document>
</nlmSearchResult>
"""

EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult></nlmSearchResult>
"""


def test_strip_html_removes_tags_and_unescapes_entities():
    assert _strip_html("&lt;span&gt;Diabetes&lt;/span&gt; &amp; more") == "Diabetes & more"


def test_fetch_medlineplus_topic_parses_title_and_summary():
    def handler(request):
        return httpx.Response(200, text=SAMPLE_XML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    topic = fetch_medlineplus_topic("diabetes", client)

    assert topic is not None
    assert topic["topic"] == "Diabetes"
    assert topic["url"] == "https://medlineplus.gov/diabetes.html"
    assert "disease" in topic["summary"]
    assert "<" not in topic["summary"]


def test_fetch_medlineplus_topic_returns_none_when_no_document():
    def handler(request):
        return httpx.Response(200, text=EMPTY_XML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert fetch_medlineplus_topic("nonexistent-term", client) is None


def test_build_documents_chunks_and_preserves_metadata():
    topics = [{"topic": "Diabetes", "url": "https://medlineplus.gov/diabetes.html", "summary": "word " * 500}]
    docs = build_documents(topics, chunk_size=200, chunk_overlap=20)

    assert len(docs) > 1
    for doc in docs:
        assert doc.metadata["topic"] == "Diabetes"
        assert doc.metadata["url"] == "https://medlineplus.gov/diabetes.html"
        assert len(doc.page_content) <= 220
