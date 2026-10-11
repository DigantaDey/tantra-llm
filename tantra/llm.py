"""Local LLM runtime: a thin, testable wrapper around llama-cpp-python plus chat-history management."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant for Bengali Tantric and Sadhana literature. "
    "Answer in the language of the question (Bengali or English). "
    "When you are not certain of a mantra, ritual detail, or quotation, "
    "say so plainly instead of inventing it. "
    "Keep answers focused and scholarly."
)


class Backend(Protocol):
    """Anything that can stream a chat completion and count tokens."""

    def stream_chat(self, messages: Sequence[dict[str, str]], params: GenerationParams) -> Iterator[str]: ...

    def count_tokens(self, text: str) -> int: ...

    @property
    def context_size(self) -> int: ...


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int = 384
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    seed: int | None = None


class ModelNotFoundError(FileNotFoundError):
    pass


class RuntimeMissingError(ImportError):
    pass


def gpu_offload_supported() -> bool:
    """True when the installed llama-cpp-python can move layers to a GPU (Metal on macOS, CUDA, …)."""
    try:
        import llama_cpp
    except ImportError:
        return False
    checker = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    return bool(checker()) if callable(checker) else False


def resolve_gpu_layers(n_gpu_layers: int | None) -> int:
    """``None`` means auto: offload every layer when the runtime supports it, otherwise CPU only."""
    if n_gpu_layers is not None:
        return n_gpu_layers
    return -1 if gpu_offload_supported() else 0


def device_label(gpu_layers: int) -> str:
    return "CPU" if gpu_layers == 0 else f"GPU ({gpu_layers} layers)" if gpu_layers > 0 else "GPU (all layers)"


class LlamaCppBackend:
    """Runs a GGUF model via llama-cpp-python (Metal on macOS when the runtime was built with it)."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 2048,
        threads: int = 4,
        n_batch: int = 256,
        seed: int = 0,
        n_gpu_layers: int | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelNotFoundError(f"model not found: {path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeMissingError(
                "llama-cpp-python is not installed. Install the local runtime with:\n"
                "    pip install -e '.[llm]'\n"
                "On macOS this compiles llama.cpp (a few minutes) and needs the Xcode Command Line Tools:\n"
                "    xcode-select --install"
            ) from exc
        self._n_ctx = n_ctx
        self.model_path = path
        self.gpu_layers = resolve_gpu_layers(n_gpu_layers)
        self._llm: Any = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=threads,
            n_threads_batch=threads,
            n_batch=n_batch,
            n_gpu_layers=self.gpu_layers,
            use_mmap=True,
            use_mlock=False,
            seed=seed,
            verbose=False,
        )

    @property
    def context_size(self) -> int:
        return self._n_ctx

    def stream_chat(self, messages: Sequence[dict[str, str]], params: GenerationParams) -> Iterator[str]:
        stream = self._llm.create_chat_completion(
            messages=list(messages),
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            repeat_penalty=params.repeat_penalty,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content")
            if text:
                yield text

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))


@dataclass
class Turn:
    role: str
    content: str


class ChatSession:
    """Multi-turn conversation that keeps the prompt inside the model's context window."""

    def __init__(self, backend: Backend, system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_turns: int = 8) -> None:
        self.backend = backend
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.history: list[Turn] = []
        self.tokens_generated = 0

    def reset(self) -> None:
        self.history.clear()
        self.tokens_generated = 0

    def _messages(self, user_text: str, budget: int, system_prompt: str | None = None) -> list[dict[str, str]]:
        """Newest turns first; drop the oldest until the prompt fits the context budget."""
        prompt = system_prompt if system_prompt is not None else self.system_prompt
        turns = self.history[-(self.max_turns * 2) :]
        while True:
            messages = [{"role": "system", "content": prompt}]
            messages += [{"role": t.role, "content": t.content} for t in turns]
            messages.append({"role": "user", "content": user_text})
            used = sum(self.backend.count_tokens(m["content"]) for m in messages) + 16 * len(messages)
            if used <= budget or not turns:
                return messages
            turns = turns[2:] if len(turns) >= 2 else []

    def ask(self, user_text: str, params: GenerationParams, *, system_prompt: str | None = None) -> Iterator[str]:
        """Stream the answer. History is updated only when the answer completes.

        ``system_prompt`` overrides the session prompt for this turn only - the
        hook the corpus-grounded mode uses to attach retrieved passages.
        """
        budget = self.backend.context_size - params.max_tokens - 32
        if budget <= 0:
            raise ValueError("max_tokens leaves no room for the prompt; lower --max-tokens or raise --ctx")
        messages = self._messages(user_text, budget, system_prompt)
        pieces: list[str] = []
        for piece in self.backend.stream_chat(messages, params):
            pieces.append(piece)
            yield piece
        answer = "".join(pieces).strip()
        self.history.append(Turn("user", user_text))
        self.history.append(Turn("assistant", answer))
        self.tokens_generated += self.backend.count_tokens(answer)
        if len(self.history) > self.max_turns * 2:
            del self.history[: len(self.history) - self.max_turns * 2]
