"""E15c: the local embedding encoder behind the ``embedding`` baseline (no API, CPU only).

Model: ``sentence-transformers/all-MiniLM-L6-v2`` (Apache-2.0, about 90 MB) through ``transformers``
``AutoTokenizer`` / ``AutoModel``, attention-masked mean pooling, then L2-normalisation, so a dot product is the
cosine. The revision is pinned. The first live use downloads the model once from huggingface.co (nothing else);
the Hugging Face cache lives on D: (``HF_HOME``), never on C:.

Nothing here is imported at module load except the standard library: ``torch`` and ``transformers`` load lazily
inside ``HFEncoder`` so offline runs, tests with a fake encoder and the report never pay for them.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"  # pinned commit on huggingface.co
MODEL_ID = f"{MODEL_NAME}@{MODEL_REVISION}"  # the embeddings-table key, so a revision change never reuses vectors
MODEL_LICENCE = "Apache-2.0"
MAX_TOKENS = 256  # the model's sentence-transformers max_seq_length; longer text is truncated
BATCH_SIZE = 16
HF_HOME_ENV = "HF_HOME"
DEFAULT_HF_SUBDIR = Path(".tmp") / "hf"  # under the repo root, git-ignored; override with HF_HOME


@runtime_checkable
class Encoder(Protocol):
    """Anything that turns texts into L2-normalised vectors. Tests pass a tiny fake."""

    model_id: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hf_home(root: str | Path | None = None) -> Path:
    """The Hugging Face cache folder: ``$HF_HOME`` when set, else ``<root>/.tmp/hf`` (git-ignored)."""
    env = os.environ.get(HF_HOME_ENV)
    if env:
        return Path(env)
    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    return base / DEFAULT_HF_SUBDIR


def l2_normalise(vec: Sequence[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    return [float(x) for x in vec] if norm == 0 else [float(x) / norm for x in vec]


class HFEncoder:
    """The real encoder. CPU only, ``eval()`` mode, no grad, input order preserved, fixed batch size."""

    model_id = MODEL_ID

    def __init__(self, *, cache_dir: str | Path | None = None, local_only: bool = False,
                 batch_size: int = BATCH_SIZE, max_tokens: int = MAX_TOKENS) -> None:
        self._cache = hf_home() if cache_dir is None else Path(cache_dir)
        self._local_only = local_only
        self._batch = batch_size
        self._max_tokens = max_tokens
        self._tok: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # must be set before huggingface_hub / transformers are imported: they read it at import time
        os.environ[HF_HOME_ENV] = str(self._cache)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._cache.mkdir(parents=True, exist_ok=True)
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        kw: dict[str, Any] = {"revision": MODEL_REVISION, "cache_dir": str(self._cache / "hub"),
                              "local_files_only": self._local_only}
        self._tok = AutoTokenizer.from_pretrained(MODEL_NAME, **kw)
        self._model = AutoModel.from_pretrained(MODEL_NAME, **kw).to("cpu").eval()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        import torch

        out: list[list[float]] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self._batch):
                enc = self._tok(list(texts[i:i + self._batch]), padding=True, truncation=True,
                                max_length=self._max_tokens, return_tensors="pt")
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.extend(row.tolist() for row in pooled)
        return out


def model_cached(cache_dir: str | Path | None = None) -> bool:
    """True when the pinned revision is already in the local cache (no network, no model load)."""
    cache = (hf_home() if cache_dir is None else Path(cache_dir)) / "hub"
    snap = cache / ("models--" + MODEL_NAME.replace("/", "--")) / "snapshots" / MODEL_REVISION
    return (snap / "config.json").exists() and any(snap.glob("*.safetensors"))


__all__ = ["BATCH_SIZE", "Encoder", "HFEncoder", "MAX_TOKENS", "MODEL_ID", "MODEL_LICENCE", "MODEL_NAME",
           "MODEL_REVISION", "hf_home", "l2_normalise", "model_cached", "text_sha256"]
