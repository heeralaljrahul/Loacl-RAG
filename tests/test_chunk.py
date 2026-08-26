from rag.chunk import Chunk, chunk_blocks, embed_text
from rag.loaders import Block


def test_headings_become_breadcrumbs():
    blocks = [Block("# Handbook\n\n## Leave policy\n\n### Carry-over\n\n"
                    "Up to five days may be carried over." + " filler." * 40)]
    chunks = chunk_blocks(blocks, chunk_chars=600, overlap_chars=100)
    assert chunks
    assert chunks[0].heading == "Handbook > Leave policy > Carry-over"


def test_sibling_heading_pops_the_stack():
    text = ("# Doc\n\n## A\n\n" + "alpha " * 60 + "\n\n## B\n\n" + "beta " * 60)
    chunks = chunk_blocks([Block(text)], chunk_chars=400, overlap_chars=0)
    headings = {c.heading for c in chunks}
    assert "Doc > A" in headings and "Doc > B" in headings
    assert "Doc > A > B" not in headings


def test_a_heading_starts_a_new_chunk():
    text = ("# Doc\n\n## First\n\n" + "one " * 80 + "\n\n## Second\n\n" + "two " * 80)
    chunks = chunk_blocks([Block(text)], chunk_chars=5000, overlap_chars=0)
    assert len(chunks) == 2, "content under two headings must not be merged"


def test_oversized_paragraph_is_split_on_sentences():
    para = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_blocks([Block(para)], chunk_chars=500, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 520 for c in chunks)


def test_overlap_carries_context_forward():
    text = "\n\n".join(f"Paragraph {i} with some words in it." for i in range(40))
    with_overlap = chunk_blocks([Block(text)], chunk_chars=300, overlap_chars=120)
    without = chunk_blocks([Block(text)], chunk_chars=300, overlap_chars=0)
    assert sum(len(c.text) for c in with_overlap) > sum(len(c.text) for c in without)


def test_pages_are_preserved():
    chunks = chunk_blocks([Block("alpha " * 50, page=3), Block("beta " * 50, page=4)],
                          chunk_chars=200, overlap_chars=0)
    assert {c.page for c in chunks} <= {3, 4}
    assert 3 in {c.page for c in chunks}


def test_embed_text_prefixes_title_and_heading():
    chunk = Chunk("body text", "Leave policy", None, 0)
    assert embed_text("Handbook", chunk).startswith("Handbook > Leave policy")


def test_no_text_is_ever_lost():
    """The invariant that matters: every sentence in the source ends up in at
    least one chunk. A short section under its own heading is the case that
    used to vanish."""
    text = (
        "# Doc\n\n## Long section\n\n" + "filler sentence here. " * 60 +
        "\n\n### Tiny subsection\n\nOnly one short line lives here.\n\n"
        "## Another\n\n" + "more filler. " * 40
    )
    chunks = chunk_blocks([Block(text)], chunk_chars=600, overlap_chars=80)
    joined = "\n".join(c.text for c in chunks)
    assert "Only one short line lives here." in joined
    for sentence in ("filler sentence here.", "more filler."):
        assert sentence in joined
    assert any(c.heading.endswith("Tiny subsection") for c in chunks)


def test_tiny_leftover_merges_into_a_sibling_of_the_same_section():
    text = "# Doc\n\n## Only\n\n" + ("word " * 130) + "\n\ntail.\n"
    chunks = chunk_blocks([Block(text)], chunk_chars=600, overlap_chars=0,
                          min_chunk_chars=120)
    assert all(len(c.text) >= 120 for c in chunks)
    assert "tail." in "\n".join(c.text for c in chunks)
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_title_is_not_repeated_in_the_breadcrumb():
    """The document's own H1 usually becomes both the title and the first
    breadcrumb component; embedding "Handbook > Handbook > Leave" wastes
    tokens and puts noise in front of the cross-encoder."""
    from rag.text import trail
    chunk = Chunk("body", "Handbook 2026 > Leave policy", None, 0)
    assert embed_text("Handbook 2026", chunk).startswith("Handbook 2026 > Leave policy\n")
    assert trail("Handbook 2026", "Handbook 2026 > Leave") == "Handbook 2026 > Leave"
    assert trail("Other", "Handbook > Leave") == "Other > Handbook > Leave"
    assert trail("Handbook", "") == "Handbook"
