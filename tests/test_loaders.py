import json

from rag.loaders import is_supported, load


def test_markdown_roundtrip(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# Title\n\nBody text.", encoding="utf-8")
    assert load(path)[0].text.startswith("# Title")


def test_csv_repeats_the_header_in_every_block(tmp_path):
    path = tmp_path / "t.csv"
    rows = ["name,role"] + [f"person{i},engineer" for i in range(90)]
    path.write_text("\n".join(rows), encoding="utf-8")
    blocks = load(path)
    assert len(blocks) == 3
    assert all(b.text.startswith("name | role") for b in blocks)


def test_json_is_flattened_to_path_value_lines(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"user": {"name": "Ada", "tags": ["x", "y"]}}), encoding="utf-8")
    text = load(path)[0].text
    assert "user.name: Ada" in text
    assert "user.tags[1]: y" in text


def test_html_headings_become_markdown(tmp_path):
    path = tmp_path / "t.html"
    path.write_text("<html><body><h2>Section</h2><p>Para.</p>"
                    "<script>bad()</script></body></html>", encoding="utf-8")
    text = load(path)[0].text
    assert "## Section" in text and "Para." in text and "bad()" not in text


def test_unsupported_extension():
    from pathlib import Path
    assert not is_supported(Path("x.bin"))
    assert is_supported(Path("x.pdf")) and is_supported(Path("x.py"))


def test_encoding_fallback(tmp_path):
    path = tmp_path / "t.txt"
    path.write_bytes("café — dash".encode("cp1252"))
    assert "caf" in load(path)[0].text
