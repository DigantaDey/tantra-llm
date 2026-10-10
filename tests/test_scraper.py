import json
import tempfile
import unittest
from pathlib import Path

import scraper


class MetadataTests(unittest.TestCase):
    def test_publication_year_uses_metadata_not_identifier(self):
        self.assertEqual(scraper.publication_year({"date": "1898"}), 1898)
        self.assertEqual(scraper.publication_year({"year": ["1904"], "date": "1920"}), 1904)
        self.assertIsNone(scraper.publication_year({"publicdate": "2020-01-01"}))
        self.assertIsNone(scraper.publication_year({"identifier": "tantra_1899"}))

    def test_bengali_language_names_and_codes(self):
        self.assertTrue(scraper.has_bengali_language({"language": ["English", "Bengali"]}))
        self.assertTrue(scraper.has_bengali_language({"language": "ben"}))
        self.assertTrue(scraper.has_bengali_language({"language": "বাংলা"}))
        self.assertFalse(scraper.has_bengali_language({"language": "Sanskrit"}))

    def test_rights_are_not_assumed_from_age(self):
        self.assertEqual(scraper.rights_status({"date": "1890"}), "unverified")
        self.assertEqual(
            scraper.rights_status({"rights": "Public Domain"}),
            "public-domain-indicated",
        )
        self.assertEqual(
            scraper.rights_status({"access-restricted-item": "true"}),
            "restricted",
        )
        self.assertFalse(scraper.is_eligible({"access-restricted-item": "true"}))

    def test_metadata_score_prioritizes_bengali_historical_items(self):
        bengali = {
            "title": "তন্ত্র সাধনা",
            "language": "Bengali",
            "date": "1905",
            "mediatype": "texts",
            "rights": "Public Domain",
        }
        generic = {"title": "A modern book about Tantra", "date": "2005", "mediatype": "texts"}
        self.assertGreater(scraper.metadata_score(bengali), scraper.metadata_score(generic))
        self.assertTrue(scraper.is_eligible(bengali, strict_language=True))
        self.assertFalse(scraper.is_eligible(generic, strict_language=True))
        self.assertFalse(scraper.is_eligible(generic, require_public_domain=True))


class FileSelectionTests(unittest.TestCase):
    def test_selects_djvu_text_and_preferred_pdf_scan(self):
        files = [
            {"name": "book.pdf", "size": "1000", "source": "original"},
            {"name": "book.djvu", "size": "3000", "source": "original"},
            {"name": "book_djvu.txt", "size": "100"},
            {"name": "readme.txt", "size": "999"},
            {"name": "book_files.txt", "size": "9999"},
            {"name": "../outside.pdf", "size": "9000"},
            {"name": "book_scandata.xml", "size": "800"},
        ]
        selected = scraper.select_files(files)
        self.assertEqual(selected["scan"], ["book.pdf"])
        self.assertEqual(selected["ocr"], ["book_djvu.txt"])
        all_scans = scraper.select_files(files, all_scans=True)
        self.assertEqual(set(all_scans["scan"]), {"book.pdf", "book.djvu"})

    def test_identifier_guard_blocks_path_traversal(self):
        self.assertIsNone(scraper._safe_identifier("../private"))
        self.assertEqual(scraper._safe_identifier("safe_id-1890"), "safe_id-1890")
        self.assertFalse(scraper._safe_archive_path("../outside.pdf"))
        self.assertFalse(scraper._safe_archive_path("C:\\outside.pdf"))


class DiscoveryTests(unittest.TestCase):
    def test_search_results_are_deduplicated_and_ranked(self):
        class FakeIA:
            calls = 0

            def search_items(self, query, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return iter(
                        [
                            {
                                "identifier": "tantra_bengali",
                                "title": "তন্ত্র সাধনা",
                                "language": ["Bengali"],
                                "date": "1902",
                                "mediatype": "texts",
                            },
                            {"identifier": "english_tantra", "title": "Tantra today", "mediatype": "texts"},
                        ]
                    )
                return iter(
                    [
                        {
                            "identifier": "tantra_bengali",
                            "subject": ["Tantra"],
                            "rights": "Public Domain",
                        }
                    ]
                )

        fake = FakeIA()
        results = scraper.discover_candidates(fake)
        self.assertEqual(fake.calls, 2)
        identifiers = [record["identifier"] for record in results]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(identifiers[0], "tantra_bengali")
        merged = next(record for record in results if record["identifier"] == "tantra_bengali")
        self.assertEqual(merged["rights_status"], "public-domain-indicated")

    def test_all_search_failures_are_reported_to_caller(self):
        class BrokenIA:
            @staticmethod
            def search_items(*args, **kwargs):
                raise OSError("offline")

        with self.assertRaisesRegex(RuntimeError, "All Archive.org searches failed"):
            scraper.discover_candidates(BrokenIA)


class DownloadTests(unittest.TestCase):
    def test_downloads_one_scan_and_one_ocr_into_identifier_folder(self):
        class FakeItem:
            def __init__(self):
                self.metadata = {"title": "তন্ত্র", "date": "1900", "identifier": "book_1900"}
                self.files = [
                    {"name": "book.pdf", "size": "1024", "source": "original"},
                    {"name": "book_djvu.txt", "size": "128"},
                ]

            def download(self, *, files, destdir, no_directory, **kwargs):
                self.requested = files
                self.no_directory = no_directory
                root = Path(destdir) / "book_1900"
                for name in files:
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"test")
                return []

        item = FakeItem()

        class FakeIA:
            @staticmethod
            def get_item(identifier):
                if identifier != "book_1900":
                    raise AssertionError(identifier)
                return item

        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "raw"
            record = scraper.download_candidate(
                FakeIA,
                {
                    "identifier": "book_1900",
                    "metadata": {"title": "তন্ত্র", "language": "Bengali"},
                    "score": 10,
                },
                raw_dir,
                verbose=False,
            )
            self.assertEqual(record["status"], "downloaded")
            self.assertEqual(len(record["downloaded_files"]), 2)
            self.assertEqual(item.requested, ["book.pdf", "book_djvu.txt"])
            self.assertFalse(item.no_directory)
            stored = json.loads((raw_dir / "book_1900" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["identifier"], "book_1900")

    def test_dry_run_does_not_create_download_directory(self):
        class FakeItem:
            def __init__(self):
                self.metadata = {"title": "বই"}
                self.files = [{"name": "book.pdf", "size": "1"}]

        class FakeIA:
            @staticmethod
            def get_item(identifier):
                return FakeItem()

        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "raw"
            record = scraper.download_candidate(
                FakeIA,
                {"identifier": "book", "metadata": {"title": "বই"}, "score": 1},
                raw_dir,
                dry_run=True,
            )
            self.assertEqual(record["status"], "planned")
            self.assertFalse(raw_dir.exists())


if __name__ == "__main__":
    unittest.main()
