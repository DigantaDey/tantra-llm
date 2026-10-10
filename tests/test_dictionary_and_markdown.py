from __future__ import annotations

import pytest

from tantra.dictionary import EsotericDictionary, Term
from tantra.markdown import DocumentMeta, reflow, render_document, split_pages, strip_page_numbers


@pytest.fixture(scope="module")
def dictionary() -> EsotericDictionary:
    return EsotericDictionary.default()


def test_default_lexicon_is_loaded_and_unique(dictionary):
    assert len(dictionary) >= 50
    texts = [t.text for t in dictionary]
    assert len(texts) == len(set(texts))
    assert "ন্যাস" in dictionary and "বীজমন্ত্র" in dictionary and "মুদ্রা" in dictionary


def test_lookup_recovers_a_one_edit_corruption(dictionary):
    # ন্যাশ is a common mis-read of ন্যাস; the anchor must find the intended term.
    hits = dictionary.lookup("ন্যাশ")
    assert hits[0] == ("ন্যাস", 1)


def test_exact_match_has_distance_zero(dictionary):
    assert dictionary.lookup("মুদ্রা")[0] == ("মুদ্রা", 0)


def test_unrelated_word_has_no_hits(dictionary):
    assert dictionary.lookup("বইটি") == []
    assert dictionary.lookup("") == []


def test_terms_in_text(dictionary):
    found = dictionary.terms_in("আমি মুদ্রা ও ন্যাস শিখেছি, বইটি পড়ো।")
    assert found == {"মুদ্রা", "ন্যাস"}


def test_malformed_tsv_is_rejected(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("একটি\tঅসম্পূর্ণ\n", encoding="utf-8")
    with pytest.raises(ValueError):
        EsotericDictionary.from_tsv(bad)


def test_custom_dictionary_respects_max_distance(tmp_path):
    path = tmp_path / "terms.tsv"
    path.write_text("# comment\nশব্দ\tsabda\tword\tcore\n", encoding="utf-8")
    d = EsotericDictionary.from_tsv(path, max_dist=1)
    assert d.lookup("শব্")[0] == ("শব্দ", 1)
    assert d.lookup("শ") == []  # 3 edits away
    assert d.get("শব্দ") == Term("শব্দ", "sabda", "word", "core")


# ----------------------------------------------------------------- markdown
def test_split_pages_on_form_feed_and_drops_trailing_empty():
    assert split_pages("a\fb\f\n") == ["a", "b"]
    assert split_pages("only one page") == ["only one page"]


def test_strip_page_numbers_only_at_edges():
    lines = ["৭", "", "কথা ১২ আছে।", "12", ""]
    assert strip_page_numbers(lines) == ["কথা ১২ আছে।"]


def test_reflow_joins_wrapped_lines_and_keeps_verse_breaks():
    paragraphs = reflow(["এক লাইন যা", "শেষ হয়।", "", "দ্বিতীয় পদ।", "লাইন দুই"])
    assert paragraphs[0] == "এক লাইন যা শেষ হয়।"
    assert paragraphs[1] == "দ্বিতীয় পদ।  \nলাইন দুই"


def test_reflow_escapes_markdown_leaders():
    assert reflow(["# এটি শিরোনাম নয়"]) == ["\\# এটি শিরোনাম নয়"]


def test_front_matter_is_valid_json_strings():
    meta = DocumentMeta(identifier="x", title='তন্ত্র "সার"', year=1900, languages=("bengali",))
    md = render_document(meta, [["প্রথম।"]], {"pages": 1})
    assert md.startswith("---\n")
    assert 'title: "তন্ত্র \\"সার\\""' in md
    assert "## Page 1" in md and "correction_pages: 1" in md
