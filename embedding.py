"""
手写 Embedding 生成 + 向量相似度检索。

同时支持本地 sentence-transformers + HF Inference API + OpenAI 兼容 API。
"""
import math

import requests
from openai import OpenAI

from eka.config import EMBEDDING_CONFIG


# ═══════════════════════════════════════════════════════════
# 本地 sentence-transformers 封装（国内推荐）
# ═══════════════════════════════════════════════════════════

# 模块级 SentenceTransformer 单例：避免多个 EmbeddingGenerator 重复加载模型到显存
_st_model_cache: dict[str, "SentenceTransformer"] = {}


def _get_or_load_st_model(model: str, device: str) -> "SentenceTransformer":
    """加载或从缓存获取 SentenceTransformer 实例。"""
    cache_key = f"{model}@{device}"
    if cache_key in _st_model_cache:
        return _st_model_cache[cache_key]

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model, device=device, trust_remote_code=True)
    _st_model_cache[cache_key] = st
    return st


def _resolve_model_path(model: str) -> str:
    """尝试从 ModelScope 下载模型，国内可稳定访问。返回本地路径或原 model ID。"""
    try:
        import logging
        logging.getLogger("modelscope").setLevel(logging.WARNING)
        from modelscope import snapshot_download
        return snapshot_download(model, revision="master")
    except Exception:
        return model


class _HFLocalEmbeddingClient:
    """本地 sentence-transformers，优先从 ModelScope 下载模型。"""

    def __init__(self, model: str, device: str, hf_endpoint: str):
        local_path = _resolve_model_path(model)
        self._model = _get_or_load_st_model(local_path, device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


# ═══════════════════════════════════════════════════════════
# HuggingFace Inference API 封装（需要翻墙）
# ═══════════════════════════════════════════════════════════

class _HFInferenceClient:
    """HuggingFace Inference API 的轻量封装，专用于 feature-extraction 模型。"""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.api_url = f"{base_url}/models/{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "inputs": texts,
            "wait_for_model": True,
        }
        resp = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()

        # HF inference API 返回:
        #  - 单条输入 → [float, ...]
        #  - 多条输入 → [[float, ...], ...]
        if isinstance(result[0], float):
            return [result]
        return result


# ═══════════════════════════════════════════════════════════
# Embedding 生成器
# ═══════════════════════════════════════════════════════════

class EmbeddingGenerator:
    """
    Embedding 生成器。

    根据 config 中的 provider 自动切换:
      - huggingface_local → 本地 sentence-transformers（国内推荐）
      - huggingface       → HF Inference API（需翻墙）
      - openai            → OpenAI 兼容 API
    """

    def __init__(self):
        self.provider = EMBEDDING_CONFIG["provider"]
        self.model = EMBEDDING_CONFIG["model"]
        self._cache: dict[str, list[float]] = {}

        if self.provider == "huggingface_local":
            self._client = _HFLocalEmbeddingClient(
                model=self.model,
                device=EMBEDDING_CONFIG["device"],
                hf_endpoint=EMBEDDING_CONFIG.get("hf_endpoint", ""),
            )
            self._backend = self._embed_local
        elif self.provider == "huggingface":
            if not EMBEDDING_CONFIG["api_key"]:
                raise ValueError("HF_TOKEN 未设置，无法使用 HuggingFace Embedding API")
            self._client = _HFInferenceClient(
                model=self.model,
                api_key=EMBEDDING_CONFIG["api_key"],
                base_url=EMBEDDING_CONFIG["base_url"],
            )
            self._backend = self._embed_hf
        else:
            self._client = OpenAI(
                api_key=EMBEDDING_CONFIG["api_key"],
                base_url=EMBEDDING_CONFIG["base_url"],
            )
            self._backend = self._embed_openai

    # ── 后端实现 ───────────────────────────────────────

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts)

    def _embed_hf(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    # ── 公开接口 ───────────────────────────────────────

    def embed(self, text: str, use_cache: bool = True) -> list[float]:
        """单条文本 → 向量"""
        if use_cache and text in self._cache:
            return self._cache[text]

        text = text.replace("\n", " ")
        results = self._backend([text])
        vec = results[0]
        if use_cache:
            self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 Embedding"""
        texts = [t.replace("\n", " ") for t in texts]
        return self._backend(texts)

    @property
    def dimension(self) -> int:
        test_vec = self.embed("test", use_cache=False)
        return len(test_vec)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ═══════════════════════════════════════════════════════════
# 手写余弦相似度
# ═══════════════════════════════════════════════════════════

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    手写余弦相似度。

    公式：cos(θ) = A·B / (|A| × |B|)

    返回值 [-1, 1]，越接近 1 越相似。
    实际中主流 Embedding 模型输出已归一化，|A| ≈ |B| ≈ 1，
    所以退化为点积。
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════
# 手写向量检索器
# ═══════════════════════════════════════════════════════════

class VectorRetriever:
    """
    手写向量检索器。

    原理：
    1. 存储文档 chunk + 对应 embedding
    2. 查询时：计算 query embedding 与每条 chunk embedding 的余弦相似度
    3. 排序返回 top-k

    时间复杂度：O(N·D)，其中 N=chunk 数，D=向量维度。
    N 很大时可用 FAISS/Milvus 做近似最近邻（ANN）。
    """

    def __init__(self, embed_gen: EmbeddingGenerator | None = None):
        self.embed_gen = embed_gen or EmbeddingGenerator()
        self.documents: list[str] = []       # chunk 原文
        self.embeddings: list[list[float]] = []  # 对应向量
        self.metadata: list[dict] = []        # 来源信息

    def add(self, chunks: list[str], metadata: list[dict] | None = None):
        """添加文档 chunk 到索引"""
        if metadata is None:
            metadata = [{}] * len(chunks)
        for chunk, meta in zip(chunks, metadata):
            self.documents.append(chunk)
            self.embeddings.append(self.embed_gen.embed(chunk))
            self.metadata.append(meta)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """
        语义检索 top-k。

        返回 list[dict]，每个包含：
        - content: chunk 原文
        - score: 余弦相似度
        - metadata: 来源信息
        """
        if not self.documents:
            return []

        query_vec = self.embed_gen.embed(query)

        scores = [cosine_similarity(query_vec, doc_vec)
                  for doc_vec in self.embeddings]

        # 排序取 top-k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            results.append({
                "content": self.documents[idx],
                "score": round(score, 4),
                "metadata": self.metadata[idx],
            })
        return results

    @property
    def size(self) -> int:
        return len(self.documents)

    def clear(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []


# ═══════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════

_default_embed_gen: EmbeddingGenerator | None = None
_default_retriever: VectorRetriever | None = None


def get_embedding_generator() -> EmbeddingGenerator:
    global _default_embed_gen
    if _default_embed_gen is None:
        _default_embed_gen = EmbeddingGenerator()
    return _default_embed_gen


def get_vector_retriever() -> VectorRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = VectorRetriever(get_embedding_generator())
    return _default_retriever
