"""
RAG 全链路 — 手写版 + LangChain 版 + 性能对比。

手写版展示底层原理，LangChain 版展示框架运用，性能对比展示工程思维。
"""
import os
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

from eka.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
    EMBEDDING_CONFIG, KNOWLEDGE_DIR,
)
from eka.chunking import recursive_chunk
from eka.embedding import EmbeddingGenerator, VectorRetriever


# ═══════════════════════════════════════════════════════════
# 手写版 RAG 管道
# ═══════════════════════════════════════════════════════════

class HandWrittenRAG:
    """手写 RAG 全链路：chunk → embed → search → context → chat。"""

    def __init__(self):
        self.embed_gen = EmbeddingGenerator()
        self.retriever = VectorRetriever(self.embed_gen)

    def load_and_index(self, directory: str = None, glob: str = "*.txt"):
        """从目录加载文档并建立索引"""
        if directory is None:
            directory = KNOWLEDGE_DIR

        all_chunks = []
        all_meta = []
        file_count = 0

        for fname in os.listdir(directory):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(directory, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = recursive_chunk(text)
            meta = {"source": fname, "filepath": fpath}
            all_chunks.extend(chunks)
            all_meta.extend([meta] * len(chunks))
            file_count += 1

        if all_chunks:
            self.retriever.add(all_chunks, all_meta)

        return {
            "files": file_count,
            "chunks": len(all_chunks),
            "avg_chunk_len": sum(len(c) for c in all_chunks) / len(all_chunks) if all_chunks else 0,
        }

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """语义检索"""
        return self.retriever.search(query, top_k=top_k)

    def generate_context(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
        """构建 LLM 上下文"""
        results = self.search(query, top_k)
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "unknown")
            parts.append(f"[文档 {i} | 来源: {source} | 相关度: {r['score']:.2f}]\n{r['content']}")
        return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# LangChain 版 RAG 管道
# ═══════════════════════════════════════════════════════════

class LangChainRAG:
    """
    LangChain RAG 管道：TextLoader + RecursiveCharacterTextSplitter
                        + OpenAIEmbeddings + Chroma。
    """

    def __init__(self):
        if EMBEDDING_CONFIG["provider"] == "huggingface_local":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_CONFIG["model"],
                model_kwargs={"device": EMBEDDING_CONFIG["device"]},
                encode_kwargs={"normalize_embeddings": True},
            )
        elif EMBEDDING_CONFIG["provider"] == "huggingface":
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=EMBEDDING_CONFIG["api_key"],
                model_name=EMBEDDING_CONFIG["model"],
                api_url=EMBEDDING_CONFIG["base_url"],
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_CONFIG["model"],
                api_key=EMBEDDING_CONFIG["api_key"],
                base_url=EMBEDDING_CONFIG["base_url"],
            )
        self.vectorstore: Chroma | None = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def load_and_index(self, directory: str = None, glob: str = "*.txt"):
        """从目录加载并建立 Chroma 索引"""
        if directory is None:
            directory = KNOWLEDGE_DIR

        from langchain_community.document_loaders import DirectoryLoader, TextLoader

        loader = DirectoryLoader(
            directory, glob=glob, loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        if not docs:
            return {"files": 0, "chunks": 0, "avg_chunk_len": 0}

        chunks = self.text_splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=os.path.join(directory, ".chroma_lc"),
        )

        return {
            "files": len(set(d.metadata.get("source", "") for d in docs)),
            "chunks": len(chunks),
            "avg_chunk_len": sum(len(d.page_content) for d in chunks) / len(chunks),
        }

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """语义检索"""
        if self.vectorstore is None:
            return []
        docs = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            {
                "content": doc.page_content,
                "score": round(score, 4),
                "metadata": doc.metadata,
            }
            for doc, score in docs
        ]

    def generate_context(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
        results = self.search(query, top_k)
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "unknown")
            parts.append(f"[文档 {i} | 来源: {source} | 相关度: {r['score']:.2f}]\n{r['content']}")
        return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# RAG 工厂 — 统一接口
# ═══════════════════════════════════════════════════════════

class RAGManager:
    """
    RAG 管理器：手写版 + LangChain 版双引擎，一键切换。

    使用方式：
      manager = RAGManager(engine="handwritten")
      manager = RAGManager(engine="langchain")
      manager.init()  # 加载索引
      results = manager.search("query")  # 检索
    """

    def __init__(self, engine: str = "handwritten"):
        if engine == "langchain":
            self.engine = LangChainRAG()
        else:
            self.engine = HandWrittenRAG()
        self.engine_name = engine

    def init(self, directory: str = None) -> dict:
        return self.engine.load_and_index(directory)

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        return self.engine.search(query, top_k)

    def generate_context(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
        return self.engine.generate_context(query, top_k)


# ═══════════════════════════════════════════════════════════
# 性能对比工具
# ═══════════════════════════════════════════════════════════

def compare_rag_performance(query: str, directory: str = None) -> dict:
    """手写版 vs LangChain 版性能对比。"""
    results = {}

    # 手写版
    hw = HandWrittenRAG()
    t0 = time.perf_counter()
    hw.load_and_index(directory)
    indexing_hw = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hw_results = hw.search(query)
    search_hw = (time.perf_counter() - t0) * 1000

    results["handwritten"] = {
        "indexing_ms": round(indexing_hw, 1),
        "search_ms": round(search_hw, 1),
        "num_results": len(hw_results),
    }

    # LangChain 版
    lc = LangChainRAG()
    t0 = time.perf_counter()
    lc.load_and_index(directory)
    indexing_lc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    lc_results = lc.search(query)
    search_lc = (time.perf_counter() - t0) * 1000

    results["langchain"] = {
        "indexing_ms": round(indexing_lc, 1),
        "search_ms": round(search_lc, 1),
        "num_results": len(lc_results),
    }

    results["comparison"] = {
        "indexing_speedup": f"{indexing_hw / indexing_lc:.1f}x" if indexing_lc else "N/A",
        "search_speedup": f"{search_hw / search_lc:.1f}x" if search_lc else "N/A",
    }

    return results
