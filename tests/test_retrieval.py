from __future__ import annotations

from tantra.llm import ChatSession, GenerationParams
from tantra.retrieval import (
    CorpusIndex,
    build_grounded_prompt,
    format_sources,
    split_passages,
)
from tests.fakes import EchoBackend

KUNDALINI_DOC = (
    "কুণ্ডলিনী শক্তি মূলাধার চক্রে কুণ্ডলিত অবস্থায় থাকে। সাধক প্রাণায়াম ও আসনের দ্বারা এই শক্তিকে জাগ্রত করেন।\n\n"
    "ষট্চক্র ভেদ করে কুণ্ডলিনী সহস্রারে উঠে যায়। এই অবস্থায় সাধক সমাধিলাভ করেন বলে শাস্ত্রে বর্ণিত আছে।\n\n"
    "সহস্রার পদ্মে শিব-শক্তির মিলন ঘটে এবং অমৃতস্রোত নামে পরিচিত আনন্দের অনুভূতি হয়।"
)
MANTRA_DOC = (
    "বীজমন্ত্র হলো দেবতার সংক্ষিপ্ত রূপ। তন্ত্রশাস্ত্রে প্রতিটি দেবতার নিজস্ব বীজমন্ত্র আছে।\n\n"
    "ন্যাস ও মুদ্রার সহায়তায় মন্ত্রজপ করা হয়। গুরুর নির্দেশ ছাড়া বীজমন্ত্র জপ নিষিদ্ধ।"
)
UNRELATED_DOC = "এই দলিলে জেলার রাজস্ব হিসাব লেখা আছে। কৃষি ও বাণিজ্যের পরিসংখ্যান এখানে সংরক্ষিত।"


def _index() -> CorpusIndex:
    return CorpusIndex.from_texts(
        [
            ("kundalini1898", "কুণ্ডলিনী তত্ত্ব", KUNDALINI_DOC),
            ("mantra1905", "বীজমন্ত্র পদ্ধতি", MANTRA_DOC),
            ("revenue1920", "রাজস্ব রিপোর্ট", UNRELATED_DOC),
        ]
    )


# ----------------------------------------------------------------- passage splitting
def test_split_passages_merges_small_paragraphs_and_caps_size():
    text = "\n\n".join(f"অনুচ্ছেদ {i}।" for i in range(10))
    passages = split_passages(text, max_chars=30)
    assert passages and all(len(p) <= 30 + 40 for p in passages)  # bounded, with merge tolerance

    long_para = "কুণ্ডলিনী শক্তি জাগ্রত হয়। " * 80  # one enormous paragraph
    pieces = split_passages(long_para, max_chars=200)
    assert all(len(p) <= 200 + 30 for p in pieces)
    assert "".join(p.replace("\n\n", "") for p in pieces).count("কুণ্ডলিনী") == long_para.count("কুণ্ডলিনী")


# ----------------------------------------------------------------- indexing
def test_index_counts_documents_and_passages():
    index = _index()
    assert index.document_ids == ["kundalini1898", "mantra1905", "revenue1920"]
    # each test document is under the passage budget, so paragraphs merge into one passage each
    assert len(index) == 3
    assert [p.ordinal for p in index.passages] == [1, 1, 1]


def test_empty_directory_gives_an_empty_index(tmp_path):
    assert len(CorpusIndex.from_dir(tmp_path)) == 0
    assert len(CorpusIndex.from_dir(tmp_path / "missing")) == 0
    assert CorpusIndex.from_dir(tmp_path).search("কুণ্ডলিনী") == []


def test_from_dir_reads_titles_from_the_markdown_front_matter(tmp_path):
    (tmp_path / "doc1.txt").write_text(KUNDALINI_DOC, encoding="utf-8")
    (tmp_path / "doc1.md").write_text(
        '---\nidentifier: "doc1"\ntitle: "কুণ্ডলিনী তত্ত্ব ১৮৯৮"\n---\n\n# heading\n', encoding="utf-8"
    )
    (tmp_path / "doc2.txt").write_text(MANTRA_DOC, encoding="utf-8")  # no .md: falls back to the stem
    index = CorpusIndex.from_dir(tmp_path)
    assert index.document_ids == ["doc1", "doc2"]
    assert index.passages[0].title == "কুণ্ডলিনী তত্ত্ব ১৮৯৮"
    assert index.passages[-1].title == "doc2"


# ----------------------------------------------------------------- retrieval
def test_search_finds_the_relevant_document_in_bengali():
    index = _index()
    hits = index.search("কুণ্ডলিনী জাগরণ কীভাবে হয়?", k=2)
    assert hits, "expected at least one hit"
    assert hits[0].passage.doc_id == "kundalini1898"
    assert "কুণ্ডলিনী" in hits[0].passage.text


def test_search_ranks_the_topical_document_above_the_offtopic_one():
    index = _index()
    hits = index.search("বীজমন্ত্র জপের নিয়ম কী", k=3)
    assert hits[0].passage.doc_id == "mantra1905"
    revenue_hits = [h for h in hits if h.passage.doc_id == "revenue1920"]
    assert not revenue_hits  # revenue never shares a topical term


def test_search_returns_empty_without_overlap_or_stopwords_only():
    index = _index()
    assert index.search("মাইক্রোসফট উইন্ডোজ") == []
    assert index.search("কী কি কোথায়") == []  # stopword-only query


def test_search_respects_k_and_deterministic_order():
    index = _index()
    assert len(index.search("কুণ্ডলিনী মন্ত্র", k=1)) == 1
    again = index.search("কুণ্ডলিনী মন্ত্র", k=3)
    assert [(h.passage.doc_id, h.score) for h in again] == [
        (h.passage.doc_id, h.score) for h in index.search("কুণ্ডলিনী মন্ত্র", k=3)
    ]


# ----------------------------------------------------------------- prompt building
def test_build_grounded_prompt_injects_citations_and_budget():
    index = _index()
    hits = index.search("কুণ্ডলিনী কী?", k=2)
    prompt = build_grounded_prompt("BASE PROMPT", hits, max_chars=10_000)
    assert prompt.startswith("BASE PROMPT")
    assert "[1]" in prompt and "[2]" in prompt
    assert "kundalini1898" in prompt
    assert "cite" in prompt.lower()


def test_build_grounded_prompt_respects_the_character_budget():
    index = _index()
    hits = index.search("কুণ্ডলিনী", k=3)
    prompt = build_grounded_prompt("BASE", hits, max_chars=150)
    assert prompt.startswith("BASE")
    assert len(prompt) < 150 + 600  # instructions + at least one passage


def test_build_grounded_prompt_without_hits_returns_the_base():
    assert build_grounded_prompt("BASE", []) == "BASE"


def test_format_sources_lists_hit_titles():
    index = _index()
    hits = index.search("কুণ্ডলিনী", k=2)
    line = format_sources(hits)
    assert line.startswith("[1]")
    assert "কুণ্ডলিনী তত্ত্ব" in line


# ----------------------------------------------------------------- chat integration
def test_chat_session_uses_the_grounded_prompt_for_that_turn_only():
    backend = EchoBackend("model")
    session = ChatSession(backend, system_prompt="plain system")
    params = GenerationParams(max_tokens=32)
    hits = _index().search("কুণ্ডলিনী কী?", k=2)
    grounded = build_grounded_prompt("plain system", hits)
    assert "".join(session.ask("প্রশ্ন", params, system_prompt=grounded)).strip()
    assert backend.calls[0][0]["content"] == grounded
    # the next turn falls back to the session prompt
    assert "".join(session.ask("আরেকটি প্রশ্ন", params)).strip()
    assert backend.calls[1][0]["content"] == "plain system"
