import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import _messages_for_turn, load_llama_model, recommended_threads, validate_gguf


class RuntimeTests(unittest.TestCase):
    def test_thread_policy_is_two_or_four(self):
        self.assertEqual(recommended_threads(1), 2)
        self.assertEqual(recommended_threads(2), 2)
        self.assertEqual(recommended_threads(4), 4)
        self.assertEqual(recommended_threads(16), 4)

    def test_gguf_magic_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.gguf"
            model.write_bytes(b"GGUF" + b"\0" * 32)
            validate_gguf(model)
            model.write_bytes(b"NOPE" + b"\0" * 32)
            with self.assertRaises(ValueError):
                validate_gguf(model)
            with self.assertRaises(FileNotFoundError):
                validate_gguf(Path(temporary) / "missing.gguf")

    def test_llama_runtime_is_explicitly_cpu_only_and_thread_bounded(self):
        class FakeLlama:
            def __init__(self, **options):
                self.options = options

        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.gguf"
            model_path.write_bytes(b"GGUF" + b"\0" * 32)
            with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
                model = load_llama_model(
                    model_path,
                    threads=2,
                    context_size=2048,
                    batch_size=128,
                )
        self.assertEqual(model.options["n_threads"], 2)
        self.assertEqual(model.options["n_threads_batch"], 2)
        self.assertEqual(model.options["n_gpu_layers"], 0)
        self.assertTrue(model.options["use_mmap"])

    def test_history_window_retains_only_requested_turns(self):
        history = []
        for index in range(5):
            history.extend(
                [
                    {"role": "user", "content": f"q{index}"},
                    {"role": "assistant", "content": f"a{index}"},
                ]
            )
        messages = _messages_for_turn("system", history, "new", 2)
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[1]["content"], "q3")
        self.assertEqual(messages[-1]["content"], "new")
        self.assertEqual(_messages_for_turn("system", history, "new", 0)[1]["content"], "new")


if __name__ == "__main__":
    unittest.main()
