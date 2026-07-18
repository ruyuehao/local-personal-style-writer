import json
import os
import re
import logging
from pathlib import Path
import sys
_reconf = getattr(sys.stdout, "reconfigure", None)
if callable(_reconf):
    _reconf(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

from device_manager import device_manager

logger = logging.getLogger("rag-engine")

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
INDEX_DIR = Path("data/rag_index")
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
TOP_K = 3


class RAGEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embeddings: np.ndarray | None = None
        self.chunks: list[dict] = []
        self._loaded = False

    def load_model(self):
        if self._loaded:
            return
        device = device_manager.pick("embedding")
        bge_path = os.environ.get("BGE_MODEL_PATH", "models/bge_int8")
        logger.info(f"Loading embedding model on {device} from {bge_path}...")
        self.model = OVModelForFeatureExtraction.from_pretrained(
            bge_path, device=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(bge_path)
        self._loaded = True
        logger.info("Embedding model loaded")

    def _embed(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].detach().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-12)

    def _chunk_text(self, text: str, source: str | None = None) -> list[dict]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                search_end = min(end + CHUNK_SIZE, len(text))
                remaining = text[end:search_end]
                match = re.search(r'[。！？\n]', remaining)
                if match:
                    end += match.end()
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "source": source or "", "start": start, "end": end})
            if end >= len(text):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def build_index(self, file_paths: list[str] | None = None, article_dir: str | None = None):
        self.load_model()
        all_chunks = []

        if file_paths:
            for fp in file_paths:
                path = Path(fp)
                if not path.exists():
                    logger.warning(f"File not found: {fp}")
                    continue
                all_chunks.extend(self._load_jsonl(path))

        if article_dir:
            dir_path = Path(article_dir)
            if dir_path.is_dir():
                for f in sorted(dir_path.glob("*.jsonl")):
                    all_chunks.extend(self._load_jsonl(f))

        if not all_chunks:
            logger.warning("No chunks to index")
            return

        texts = [c["text"] for c in all_chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        self.embeddings = self._embed(texts)
        self.chunks = all_chunks
        self._persist()
        logger.info(f"Index built: {len(self.chunks)} chunks, dim {self.embeddings.shape[1]}")

    def _load_jsonl(self, path: Path) -> list[dict]:
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                output = record.get("output") or record.get("response") or ""
                instruction = record.get("instruction") or record.get("prompt") or ""
                topic = ""
                inp = record.get("input") or record.get("query") or {}
                if isinstance(inp, dict):
                    topic = inp.get("topic", "")
                text = f"{instruction}\n{output}" if output else ""
                if text:
                    chunks.extend(self._chunk_text(text, source=str(path.name)))
        return chunks

    def _persist(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(EMBEDDINGS_FILE), self.embeddings)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        logger.info(f"Index saved to {INDEX_DIR}")

    def load_index(self) -> bool:
        if not EMBEDDINGS_FILE.exists() or not CHUNKS_FILE.exists():
            return False
        self.embeddings = np.load(str(EMBEDDINGS_FILE))
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"Index loaded: {len(self.chunks)} chunks")
        return True

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if self.embeddings is None or len(self.chunks) == 0:
            if not self.load_index():
                return []
            if self.embeddings is None:
                return []
        self.load_model()
        query_vec = self._embed([query])
        scores = query_vec @ self.embeddings.T
        top_indices = np.argsort(scores[0])[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": float(scores[0][idx]),
            })
        return results

    def augment_prompt(self, query: str, user_prompt: str, top_k: int = TOP_K) -> str:
        results = self.retrieve(query, top_k)
        if not results:
            return user_prompt
        refs = "\n\n".join(
            f"--- 参考片段 {i+1} ---\n{r['text']}" for i, r in enumerate(results)
        )
        return f"以下是用户历史文章中与当前主题相关的参考片段：\n\n{refs}\n\n{user_prompt}"

    @property
    def status(self) -> dict:
        return {
            "loaded": self.embeddings is not None,
            "chunk_count": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "index_path": str(INDEX_DIR),
        }


engine = RAGEngine()
