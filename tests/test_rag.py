from erp_copilot.rag import Retriever, tokenize


def test_tokenize_drops_stopwords_and_stems() -> None:
    assert tokenize("The invoices are overdue") == ["invoic", "overdu"]


def test_chunks_have_stable_refs() -> None:
    refs = {c.ref for c in Retriever.from_dir().chunks}
    assert "invoice-holds#how-to-release-a-hold" in refs
    assert len(refs) >= 20


def test_top_hit_is_relevant() -> None:
    hit = Retriever.from_dir().search("who approves a credit memo above 1000", k=1)[0]
    assert hit.chunk.doc_id == "credit-memo-policy"
    assert 0 < hit.coverage <= 1


def test_unknown_terms_return_nothing() -> None:
    assert Retriever.from_dir().search("weather forecast Sao Paulo") == []
