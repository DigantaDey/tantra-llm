"""Document-level correction: raw archive text in, Markdown and plain text out."""

from __future__ import annotations

from tantra.bengali import normalize
from tantra.dictionary import EsotericDictionary
from tantra.markdown import DocumentMeta
from tantra.pipeline import correct_document
from tests.fakes import controlled_corrector

META = DocumentMeta(
    identifier="sadhana1898",
    title="সাধনা",
    year=1898,
    languages=("bengali",),
    licenseurl="https://creativecommons.org/publicdomain/mark/1.0/",
    source_url="https://archive.org/details/sadhana1898",
)
# Page 1 starts with a page number and ends with an English note; page 2 ends with a page number.
RAW = "12\nন্যাশ ও শশ নশ\nSome English note 1898\n\f\nতন্ত্রসাধনায় ন্যাশ\n3\n"


def _corrector(bonus: float = 3.0):
    return controlled_corrector(bonus, EsotericDictionary.default())


def test_pages_are_split_on_form_feed_and_rendered_in_order():
    result = correct_document(RAW, _corrector(), META)
    assert result.pages == 2
    md = result.markdown
    assert md.index("## Page 1") < md.index("## Page 2")
    assert md.startswith("---\n") and 'identifier: "sadhana1898"' in md
    assert "# সাধনা" in md


def test_bengali_lines_are_corrected_and_counted():
    result = correct_document(RAW, _corrector(), META)
    assert "ন্যাস ও শশ নশ" in result.plain_text  # the lexicon term is restored
    # Input 'য' + nukta is composed to U+09DF by normalize(); compare in the normalized form.
    assert normalize("তন্ত্রসাধনায় ন্যাস") in result.plain_text
    assert "\u09df" in result.plain_text
    assert "ন্যাশ" not in result.plain_text
    assert result.lines_corrected == 2
    assert result.stats.dictionary_anchors == 2
    assert result.extra["dictionary_used"] == "yes"


def test_page_number_lines_are_removed_at_page_edges():
    result = correct_document(RAW, _corrector(), META)
    lines = result.plain_text.split("\n")
    assert "12" not in lines and "3" not in lines
    assert "correction_pages: 2" in result.markdown


def test_latin_lines_pass_through_untouched():
    result = correct_document(RAW, _corrector(), META)
    assert "Some English note 1898" in result.plain_text
    assert result.lines_passed_through == 1  # only the English line was not sent to the model


def test_dictionary_can_be_disabled_per_document():
    result = correct_document(RAW, _corrector(3.0), META, use_dictionary=False)
    assert "ন্যাশ" in result.plain_text  # no bonus applied, so the model's own reading stays
    assert result.stats.dictionary_anchors == 0
    assert result.extra["dictionary_used"] == "no"


def test_empty_document_renders_front_matter_and_no_pages():
    result = correct_document("", _corrector(), META)
    assert result.pages == 1 and result.lines_corrected == 0
    assert result.plain_text.strip() == ""
    assert "## Page" not in result.markdown
