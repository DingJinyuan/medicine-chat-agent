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
    unescaped = html.unescape(raw)
    return re.sub(r"<[^>]+>", "", unescaped).strip()


def fetch_medlineplus_topic(term: str, client: httpx.Client) -> dict | None:
    resp = client.get(MEDLINEPLUS_ENDPOINT, params={"db": "healthTopics", "term": term, "retmax": 1})
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    doc = root.find(".//document")
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
    results: list[dict] = []
    with httpx.Client(timeout=20.0) as client:
        for term in terms:
            try:
                topic = fetch_medlineplus_topic(term, client)
            except (httpx.HTTPError, ET.ParseError):
                continue
            if topic:
                results.append(topic)
    return results


def load_or_fetch_corpus(cache_path: str, terms: list[str] = DEFAULT_TOPICS) -> list[dict]:
    path = Path(cache_path)
    if path.exists():
        return json.loads(path.read_text())

    topics = fetch_all_topics(terms)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(topics, indent=2))
    return topics


def build_documents(topics: list[dict], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents: list[Document] = []
    for t in topics:
        for chunk in splitter.split_text(t["summary"]):
            documents.append(Document(page_content=chunk, metadata={"topic": t["topic"], "url": t["url"]}))
    return documents
