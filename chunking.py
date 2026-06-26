"""
手写三种分块策略 + LangChain 等价实现对比。

每个策略都有"手写版"方法和 LangChain 等价版本的注释说明。
"""
import re
from typing import Callable

from eka.config import CHUNK_SIZE, CHUNK_OVERLAP

# ═══════════════════════════════════════════════════════════
# 策略 1：递归字符切分（Recursive Character Splitter）
# LangChain 等价：langchain_text_splitters.RecursiveCharacterTextSplitter
# ═══════════════════════════════════════════════════════════

SEPARATORS = ["\n\n", "\n", "。", ". ", "? ", "! ", "；", "; ", "，", ", ", " ", ""]


def recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP,
                    separators: list[str] | None = None) -> list[str]:
    """
    手写递归分块。

    原理：按分隔符优先级从高到低递归切分。
    1. 先用最高优先级分隔符（如 \\n\\n）切分
    2. 对每个超过 chunk_size 的片段，降级到下一分隔符
    3. 最终用字符级别切分兜底
    4. 相邻 chunk 保留 overlap

    LangChain 对比：
      from langchain_text_splitters import RecursiveCharacterTextSplitter
      splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size, chunk_overlap=chunk_overlap,
          separators=["\\n\\n", "\\n", "。", ".", " ", ""]
      )
      chunks = splitter.split_text(text)
    """
    if separators is None:
        separators = SEPARATORS
    return _split_recursive(text, separators, 0, chunk_size, chunk_overlap)


def _split_recursive(text: str, seps: list[str], sep_idx: int,
                     chunk_size: int, overlap: int) -> list[str]:
    if sep_idx >= len(seps):
        # 兜底：字符级切分
        return _char_split_with_overlap(text, chunk_size, overlap)

    sep = seps[sep_idx]
    if sep == "":
        return _char_split_with_overlap(text, chunk_size, overlap)

    parts = text.split(sep)
    result = []
    for part in parts:
        if not part:
            continue
        if len(part) <= chunk_size:
            result.append(part)
        else:
            # 递归降级
            result.extend(_split_recursive(part, seps, sep_idx + 1,
                                           chunk_size, overlap))
    return result


def _char_split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ═══════════════════════════════════════════════════════════
# 策略 2：语义切分（Semantic Chunker）
# LangChain 等价：langchain_experimental.text_splitter.SemanticChunker
# ═══════════════════════════════════════════════════════════

def semantic_chunk(text: str, embed_fn: Callable[[str], list[float]],
                   chunk_size: int = CHUNK_SIZE,
                   similarity_threshold: float = 0.5) -> list[str]:
    """
    手写语义切分。

    原理：
    1. 按句子切分全文
    2. 计算相邻句子组的 Embedding 余弦相似度
    3. 相似度低于阈值 → 语义边界 → 此处切分
    4. 合并边界内的句子直到接近 chunk_size

    LangChain 等价：
      from langchain_experimental.text_splitter import SemanticChunker
      splitter = SemanticChunker(embeddings=OpenAIEmbeddings(),
                                 breakpoint_threshold_type="percentile")
      chunks = splitter.split_text(text)
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # 用滑动小窗口计算相邻语义相似度
    window = 3
    breakpoints = []
    for i in range(window, len(sentences) - window):
        left = "".join(sentences[i - window:i])
        right = "".join(sentences[i:i + window])
        try:
            left_emb = embed_fn(left)
            right_emb = embed_fn(right)
            sim = _cosine_sim(left_emb, right_emb)
            if sim < similarity_threshold:
                breakpoints.append(i)
        except Exception:
            pass

    if not breakpoints:
        return _merge_sentences_to_chunks(sentences, chunk_size)

    # 按断点分组
    groups = []
    prev = 0
    for bp in breakpoints:
        groups.append(sentences[prev:bp])
        prev = bp
    groups.append(sentences[prev:])

    # 每组合并到接近 chunk_size
    chunks = []
    for group in groups:
        chunks.extend(_merge_sentences_to_chunks(group, chunk_size))
    return [c for c in chunks if c.strip()]


def _split_sentences(text: str) -> list[str]:
    """按中英文标点切句子"""
    pattern = re.compile(r'(?<=[。！？.!?\n])\s*')
    parts = pattern.split(text)
    return [p for p in parts if p.strip()]


def _merge_sentences_to_chunks(sentences: list[str], max_size: int) -> list[str]:
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_size:
            current += sent
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


# ═══════════════════════════════════════════════════════════
# 策略 3：滑动窗口切分（Sliding Window Chunker）
# ═══════════════════════════════════════════════════════════

def sliding_window_chunk(text: str, window_size: int = CHUNK_SIZE,
                         step_size: int | None = None,
                         overlap: int | None = None) -> list[str]:
    """
    滑动窗口切分。

    step_size 或 overlap 二选一。窗口以固定步长滑动。

    适用场景：
    - 流式文本（实时日志、聊天记录）
    - 不需要语义边界的场景
    - 最简单的实现，但可能切断句子
    """
    if step_size is None:
        step_size = window_size - (overlap or CHUNK_OVERLAP)

    if step_size <= 0:
        step_size = window_size // 2

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step_size
    return chunks


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_report(chunks: list[str]) -> dict:
    """统计分块结果"""
    if not chunks:
        return {"count": 0}
    lengths = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "total_chars": sum(lengths),
        "avg_len": sum(lengths) / len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }