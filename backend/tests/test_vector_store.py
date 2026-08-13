from app.vector_store import build_or_load_vector_store


def test_search_ranks_lexically_similar_chunk_first(fake_vector_store):
    results = fake_vector_store.similarity_search_with_score("headache with light sensitivity", k=3)
    top_doc, _ = results[0]
    assert top_doc.metadata["topic"] == "Migraine"


def test_search_preserves_source_metadata(fake_vector_store):
    results = fake_vector_store.similarity_search_with_score("runny nose sore throat", k=1)
    doc, _ = results[0]
    assert doc.metadata["topic"] == "Common Cold"
    assert doc.metadata["url"].startswith("https://medlineplus.gov/")


def test_persisted_index_reloads_with_same_results(tmp_path, fake_embeddings, sample_documents):
    index_path = str(tmp_path / "index")
    build_or_load_vector_store(index_path, "fake", documents=sample_documents, embeddings=fake_embeddings)

    reloaded = build_or_load_vector_store(index_path, "fake", embeddings=fake_embeddings)
    results = reloaded.similarity_search_with_score("blood sugar diabetes", k=1)
    assert results[0][0].metadata["topic"] == "Diabetes"


def test_build_raises_without_documents_or_existing_index(tmp_path, fake_embeddings):
    import pytest

    with pytest.raises(FileNotFoundError):
        build_or_load_vector_store(str(tmp_path / "missing"), "fake", embeddings=fake_embeddings)
