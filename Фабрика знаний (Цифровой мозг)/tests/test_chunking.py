from kf.chunking import chunk_text


def test_short_text_is_single_chunk():
    text = "Короткая заметка."
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert chunks == ["Короткая заметка."]


def test_splits_long_text_on_paragraph_boundaries():
    para_a = "A" * 300
    para_b = "B" * 300
    text = f"{para_a}\n\n{para_b}"
    chunks = chunk_text(text, max_chars=400, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].strip() == para_a
    assert chunks[1].strip() == para_b


def test_chunks_have_overlap():
    text = "X" * 1000
    chunks = chunk_text(text, max_chars=400, overlap=50)
    assert len(chunks) > 1
    tail_of_first = chunks[0][-50:]
    head_of_second = chunks[1][:50]
    assert tail_of_first == head_of_second


def test_no_chunk_exceeds_max_chars():
    text = "слово " * 1000
    chunks = chunk_text(text, max_chars=300, overlap=30)
    assert all(len(c) <= 300 for c in chunks)


def test_empty_text_returns_no_chunks():
    assert chunk_text("", max_chars=500, overlap=50) == []


def test_merges_short_paragraphs_up_to_limit():
    paragraphs = ["Пункт один.", "Пункт два.", "Пункт три."]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=1000, overlap=0)
    assert len(chunks) == 1
    assert all(p in chunks[0] for p in paragraphs)
