from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tantra.bengali import normalize
from tantra.cli import clean as clean_cli
from tantra.cli import scrape as scrape_cli
from tantra.cli.chat import find_default_model, pick_threads
from tantra.cli.chat import main as chat_main
from tests.fakes import FakeArchive

ROOT = Path(__file__).resolve().parent.parent
BENGALI_LINES = [
    "তন্ত্রসাধনায় বীজমন্ত্র ও ন্যাস এবং মুদ্রা সম্পর্কে একটি দীর্ঘ বাক্য লেখা হলো।",
    "কুণ্ডলিনী শক্তি ষট্চক্র ভেদ করে সহস্রারে পৌঁছায় বলে বিশ্বাস করা হয়।",
    "শ্মশানে বসে সাধক জপ করেন এবং ভৈরবীর আশীর্বাদ প্রার্থনা করেন।",
    "মূলাধার থেকে সহস্রার পর্যন্ত চক্রগুলির ধ্যান করা হয় নিয়মিত।",
]


def test_pick_threads_uses_two_or_four():
    assert pick_threads(1) == 2
    assert pick_threads(2) == 2
    assert pick_threads(4) == 4
    assert pick_threads(16) == 4


def test_find_default_model_prefers_q4_k_m(tmp_path):
    assert find_default_model(tmp_path) is None
    (tmp_path / "a-Q3_K_L.gguf").write_bytes(b"GGUF")
    (tmp_path / "b-Q4_K_M.gguf").write_bytes(b"GGUF")
    assert find_default_model(tmp_path).name == "b-Q4_K_M.gguf"


def test_find_default_model_prefers_the_newest_q4_k_m(tmp_path):
    (tmp_path / "older-Q4_K_M.gguf").write_bytes(b"GGUF")
    (tmp_path / "newer-Q4_K_M.gguf").write_bytes(b"GGUF")
    os.utime(tmp_path / "older-Q4_K_M.gguf", (1_000_000, 1_000_000))
    assert find_default_model(tmp_path).name == "newer-Q4_K_M.gguf"


def test_chat_fails_gracefully_without_a_model(tmp_path, capsys):
    code = chat_main(["--model", str(tmp_path / "missing.gguf")])
    assert code == 1
    assert "model not found" in capsys.readouterr().out


def test_scrape_year_range_validation():
    with pytest.raises(SystemExit):
        scrape_cli._year_range([1935, 1870])
    assert scrape_cli._year_range([1870, 1935]) == (1870, 1935)


def test_scrape_help_runs():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scraper.py"), "--help"], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0 and "download" in out.stdout


def _scrape_args(**overrides):
    base = {
        "command": "search",
        "limit": 10,
        "year_range": [1870, 1935],
        "allow_undated": False,
        "require_marked_pd": False,
        "query": None,
        "show_rejected": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_scrape_search_renders_rights_and_accepts_unmarked(capsys):
    from tantra.scraper import evaluate, parse_candidate

    records = [
        {
            "identifier": "unmarked1910",
            "title": "Diksha O Sadhana",
            "year": 1910,
            "language": "ben",
            "subject": "tantra",
            "licenseurl": "",
        },
        {
            "identifier": "english1900",
            "title": "A tantric cookbook",
            "year": 1900,
            "language": "eng",
            "subject": "tantra",
        },
    ]
    archive = FakeArchive(records)
    assert scrape_cli._cmd_search(_scrape_args(), archive) == 0
    out = capsys.readouterr().out
    assert "unmarked1910" in out and "unmarked" in out
    assert "accepted" in out
    assert "english1900" not in out  # rejected hit is hidden without --show-rejected
    # the default policy accepts the unmarked scan; the strict flag would not
    cand = parse_candidate(records[0])
    assert evaluate(cand).accepted
    assert not evaluate(cand, rights_policy="marked").accepted


def test_scrape_search_strict_mode_rejects_unmarked(capsys):
    records = [
        {
            "identifier": "unmarked1910",
            "title": "Diksha O Sadhana",
            "year": 1910,
            "language": "ben",
            "subject": "tantra",
            "licenseurl": "",
        }
    ]
    archive = FakeArchive(records)
    assert scrape_cli._cmd_search(_scrape_args(require_marked_pd=True, show_rejected=True), archive) == 0
    out = capsys.readouterr().out
    assert "not marked public domain" in out
    assert "unmarked1910" in out  # shown because --show-rejected


def test_scrape_download_rejects_and_skips_specific_identifier(capsys, tmp_path):
    records = [
        {
            "identifier": "modern1999",
            "title": "Modern Sadhana",
            "year": 1999,  # outside the default range
            "language": "ben",
            "subject": "tantra",
        }
    ]
    archive = FakeArchive(records)
    args = argparse.Namespace(
        command="download",
        limit=10,
        year_range=[1870, 1935],
        allow_undated=False,
        require_marked_pd=False,
        query=None,
        identifier=["modern1999"],
        formats="djvu_txt",
        out=tmp_path,
        delay=0,
        retries=1,
        dry_run=False,
    )
    assert scrape_cli._cmd_download(args, archive) == 1
    assert "outside 1870-1935" in capsys.readouterr().out
    assert archive.download_calls == []


def _train_tiny(tmp_path: Path, corpus: Path, ckpt: Path) -> int:
    return clean_cli.main(
        [
            "train",
            "--corpus",
            str(corpus),
            "--out",
            str(ckpt),
            "--steps",
            "6",
            "--batch-size",
            "8",
            "--threads",
            "1",
            "--d-model",
            "32",
            "--layers",
            "1",
            "--heads",
            "2",
            "--max-len",
            "64",
            "--copies",
            "1",
            "--max-tags",
            "64",
            "--warmup",
            "2",
        ]
    )


def test_end_to_end_train_correct_calibrate_evaluate(tmp_path):
    corpus = tmp_path / "baseline"
    corpus.mkdir()
    (corpus / "a.txt").write_text("\n".join(BENGALI_LINES * 8), encoding="utf-8")
    ckpt = tmp_path / "denoiser.pt"
    assert _train_tiny(tmp_path, corpus, ckpt) == 0
    assert ckpt.exists()

    raw = tmp_path / "raw" / "sadhana1898"
    raw.mkdir(parents=True)
    (raw / "metadata.json").write_text(
        json.dumps(
            {
                "identifier": "sadhana1898",
                "title": "সাধনা",
                "year": 1898,
                "languages": ["bengali"],
                "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/",
                "source_url": "https://archive.org/details/sadhana1898",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    page1 = "\n".join(["12", *BENGALI_LINES[:2], "", "Some English note 1898"])
    page2 = "\n".join(BENGALI_LINES[2:])
    (raw / "sadhana1898_djvu.txt").write_text(f"{page1}\f{page2}\f", encoding="utf-8")

    processed = tmp_path / "processed"
    assert (
        clean_cli.main(
            [
                "correct",
                "--checkpoint",
                str(ckpt),
                "--input",
                str(tmp_path / "raw"),
                "--out-dir",
                str(processed),
            ]
        )
        == 0
    )
    md = (processed / "sadhana1898.md").read_text(encoding="utf-8")
    assert md.startswith("---\n") and 'identifier: "sadhana1898"' in md
    assert "## Page 1" in md and "## Page 2" in md
    assert "Some English note 1898" in md  # Latin lines pass through untouched
    assert "\n12\n" not in md  # leading page number removed

    calib = tmp_path / "calibration.txt"
    assert clean_cli.main(["calibration", "--processed", str(processed), "--out", str(calib)]) == 0
    assert calib.read_text(encoding="utf-8").strip()

    metrics = tmp_path / "metrics.json"
    assert (
        clean_cli.main(
            [
                "evaluate",
                "--checkpoint",
                str(ckpt),
                "--corpus",
                str(corpus),
                "--noise",
                "1",
                "--max-windows",
                "20",
                "--json",
                str(metrics),
            ]
        )
        == 0
    )
    rows = json.loads(metrics.read_text(encoding="utf-8"))
    assert rows and {"cer_noisy", "cer_model", "cer_dictionary", "edit_recall_model"} <= set(rows[0])
    assert rows[0]["gold_edit_positions"] > 0  # the synthetic noise always creates edits to recover
    assert rows[0]["edit_recall_model"] is None or 0.0 <= rows[0]["edit_recall_model"] <= 1.0


def test_correct_refuses_a_non_bengali_document(tmp_path, capsys):
    chkpt_dir = tmp_path / "x"
    corpus = tmp_path / "baseline"
    corpus.mkdir()
    (corpus / "a.txt").write_text("\n".join(BENGALI_LINES * 4), encoding="utf-8")
    ckpt = chkpt_dir / "d.pt"
    assert _train_tiny(tmp_path, corpus, ckpt) == 0
    raw = tmp_path / "raw" / "english1900"
    raw.mkdir(parents=True)
    (raw / "english1900_djvu.txt").write_text("This is only English text.\n" * 5, encoding="utf-8")
    assert (
        clean_cli.main(
            [
                "correct",
                "--checkpoint",
                str(ckpt),
                "--input",
                str(tmp_path / "raw"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert "not mostly Bengali" in capsys.readouterr().out
    assert not (tmp_path / "out" / "english1900.md").exists()


def test_correct_without_a_checkpoint_cleans_structurally(tmp_path):
    """--no-model is the zero-setup path: no denoiser, no baseline corpus, still
    produces processed .md/.txt good enough for calibration and grounded chat."""
    raw = tmp_path / "raw" / "sadhana1898"
    raw.mkdir(parents=True)
    (raw / "metadata.json").write_text(
        json.dumps({"identifier": "sadhana1898", "title": "সাধনা", "year": 1898}, ensure_ascii=False),
        encoding="utf-8",
    )
    (raw / "sadhana1898_djvu.txt").write_text(
        f"{'12'}\n{BENGALI_LINES[0]}\n\n{BENGALI_LINES[1]}\f{BENGALI_LINES[2]}\n", encoding="utf-8"
    )
    out = tmp_path / "processed"
    assert clean_cli.main(["correct", "--no-model", "--input", str(raw.parent), "--out-dir", str(out)]) == 0
    text = (out / "sadhana1898.txt").read_text(encoding="utf-8")
    # compare against the NFC/composed form: that is what the structural pass normalises to
    assert normalize(BENGALI_LINES[0]).split("।")[0] in text
    assert "\n12\n" not in text  # leading bare page number is stripped structurally
    md = (out / "sadhana1898.md").read_text(encoding="utf-8")
    assert "no model (structure only)" in md
    calib = tmp_path / "calibration.txt"
    assert clean_cli.main(["calibration", "--processed", str(out), "--out", str(calib)]) == 0
    assert calib.read_text(encoding="utf-8").strip()


def test_train_without_corpus_gives_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as info:
        clean_cli.main(["train", "--corpus", str(tmp_path / "nope")])
    assert "corpus not found" in str(info.value)


# ----------------------------------------------------------------- shell entry point
def test_quantize_script_help_and_validation(tmp_path):
    script = str(ROOT / "quantize.sh")
    bash = ["bash", script]
    assert subprocess.run(bash + ["--help"], capture_output=True, text=True).returncode == 0
    missing = subprocess.run(bash, capture_output=True, text=True)
    assert missing.returncode == 1 and "--hf-dir is required" in missing.stderr
    bad_type = subprocess.run(bash + ["--hf-dir", str(tmp_path), "--types", "Q9"], capture_output=True, text=True)
    assert bad_type.returncode == 1
    fake_hf = tmp_path / "hf"
    fake_hf.mkdir()
    (fake_hf / "config.json").write_text("{}", encoding="utf-8")
    plan = subprocess.run(
        bash
        + [
            "--hf-dir",
            str(fake_hf),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--calibration",
            str(tmp_path / "none.txt"),
        ],
        capture_output=True,
        text=True,
    )
    assert plan.returncode == 0 and "dry run" in plan.stdout
    assert "continuing without an i-matrix" in plan.stderr
