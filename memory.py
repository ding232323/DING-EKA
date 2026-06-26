"""
三层记忆架构 — 手写实现 + LangChain 等价对比。

三层架构：
  1. 短期记忆（Sliding Window）：保留最近 N 条消息，O(1) 实现
  2. 中期记忆（Summary Buffer）：旧消息压缩为摘要，平衡完整性 vs 成本
  3. 长期记忆（Vector Memory）：ChromaDB 持久化，跨会话语义检索

LangChain 等价：
  - SlidingWindow → langchain.memory.ConversationBufferWindowMemory
  - SummaryBuffer → langchain_classic.memory.ConversationSummaryBufferMemory
  - VectorMemory → langchain.memory.vectorstore.VectorStoreRetrieverMemory
"""
import os
import re
from datetime import datetime
from openai import OpenAI

from eka.config import (
    LLM_CONFIG, MAX_WINDOW_MESSAGES, SUMMARY_BUFFER_SIZE,
    LONG_TERM_COLLECTION,
)
from eka.embedding import EmbeddingGenerator, cosine_similarity

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


# ═══════════════════════════════════════════════════════════
# 1. 短期记忆：滑动窗口
# ═══════════════════════════════════════════════════════════

class SlidingWindowMemory:
    """
    手写滑动窗口记忆。

    LangChain 等价：
      from langchain.memory import ConversationBufferWindowMemory
      memory = ConversationBufferWindowMemory(k=20, return_messages=True)

    复杂度：插入 O(1)，获取 O(k)，k=窗口大小
    """

    def __init__(self, max_messages: int = MAX_WINDOW_MESSAGES):
        self.max_messages = max_messages
        self.messages: list[dict] = []

    def add(self, role: str, content: str):
        """添加消息，自动截断"""
        self.messages.append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat(),
        })
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self, max_tokens: int = 2000) -> str:
        """获取最近对话上下文字符串"""
        if not self.messages:
            return ""
        estimated = 0
        selected = []
        for msg in reversed(self.messages):
            msg_len = len(msg["content"])
            if estimated + msg_len > max_tokens * 2:  # 粗略估计
                break
            selected.insert(0, msg)
            estimated += msg_len
        return "\n".join(f"[{m['role']}]: {m['content']}" for m in selected)

    def clear(self):
        self.messages = []

    @property
    def message_count(self) -> int:
        return len(self.messages)


# ═══════════════════════════════════════════════════════════
# 2. 中期记忆：摘要缓冲
# ═══════════════════════════════════════════════════════════

class SummaryBufferMemory:
    """
    手写摘要缓冲记忆。

    原理：当窗口消息超过阈值，让 LLM 把旧消息压缩为一条摘要。
    摘要替代旧消息参与后续上下文构建。

    LangChain 等价：
      from langchain_classic.memory import ConversationSummaryBufferMemory
      memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4000)
    """

    SUMMARY_PROMPT = """将以下对话历史压缩为一段简洁的摘要（2-3 句话）。
重点保留：用户提出的问题、关键决策、重要数据、用户偏好。

对话历史：
{history}

摘要："""

    def __init__(self, max_token_limit: int = SUMMARY_BUFFER_SIZE):
        self.max_token_limit = max_token_limit
        self.summary: str = ""
        self.recent_messages: list[dict] = []
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = OpenAI(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
            )
        return self._llm

    def add(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})
        # 估算总 token（简单估算：1 token ≈ 1.5 中文字符 ≈ 4 英文字符）
        total_chars = sum(len(m["content"]) for m in self.recent_messages)
        if total_chars > self.max_token_limit * 1.5:
            self._compress()

    def _compress(self):
        """压缩旧消息为摘要"""
        if not self.recent_messages:
            return

        history = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in self.recent_messages[:-2]
        )

        try:
            resp = self.llm.chat.completions.create(
                model=LLM_CONFIG["model"],
                max_completion_tokens=200,
                messages=[{
                    "role": "system",
                    "content": self.SUMMARY_PROMPT.format(history=history[:3000]),
                }],
            )
            new_summary = resp.choices[0].message.content.strip()
            self.summary = f"{self.summary}\n{new_summary}".strip()
        except Exception:
            new_summary = f"对话摘要 ({len(history)} 字历史)"
            self.summary = f"{self.summary}\n{new_summary}".strip()

        # 只保留最后 2 条消息
        self.recent_messages = self.recent_messages[-2:]

    def get_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[对话历史摘要]\n{self.summary}")
        if self.recent_messages:
            parts.append("[最近对话]\n" + "\n".join(
                f"[{m['role']}]: {m['content']}" for m in self.recent_messages
            ))
        return "\n\n".join(parts)

    def clear(self):
        self.summary = ""
        self.recent_messages = []


# ═══════════════════════════════════════════════════════════
# 3. 长期记忆：向量记忆
# ═══════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """从以下内容中提取值得长期记忆的重要事实。
只提取真正重要的信息：用户偏好、关键结论、重要数据、决策记录。
每行一条事实，用 "事实：" 开头。如果没有值得记的内容，输出 "无"。"""


class VectorLongTermMemory:
    """
    手写向量长期记忆。

    原理：
    1. 对话结束时，LLM 提取"值得记住的事实"
    2. 事实 → Embedding → ChromaDB 持久化
    3. 新对话开始时，用当前问题做语义检索，召回相关记忆
    4. 去重：相似度 > 0.95 的旧事实不重复写入

    LangChain 等价：
      from langchain.memory.vectorstore import VectorStoreRetrieverMemory
    """

    def __init__(self, collection_name: str = LONG_TERM_COLLECTION):
        if not HAS_CHROMA:
            raise ImportError("需要 chromadb: pip install chromadb")
        self.embed_gen = EmbeddingGenerator()
        self._chroma = chromadb.Client()
        self.collection = self._chroma.get_or_create_collection(name=collection_name)
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = OpenAI(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
            )
        return self._llm

    def extract_facts(self, text: str) -> list[str]:
        """LLM 提取值得记忆的事实"""
        try:
            resp = self.llm.chat.completions.create(
                model=LLM_CONFIG["model"],
                max_completion_tokens=200,
                messages=[
                    {"role": "system", "content": _EXTRACTION_PROMPT},
                    {"role": "user", "content": text[:3000]},
                ],
            )
            facts = []
            for line in resp.choices[0].message.content.split("\n"):
                line = line.strip()
                if line.startswith("事实：") or line.startswith("事实:"):
                    fact = line.replace("事实：", "").replace("事实:", "").strip()
                    if fact and fact != "无":
                        facts.append(fact)
            return facts
        except Exception:
            return []

    def remember(self, fact: str, metadata: dict | None = None):
        """写入一条事实（带去重）。嵌入失败时降级为无嵌入存储。"""
        # 去重检查
        try:
            existing = self.recall(fact, top_k=1)
            if existing and existing[0][1] > 0.95:
                return False  # 已存在相似记忆
        except Exception:
            pass  # 检索失败不阻断写入

        try:
            emb = self.embed_gen.embed(fact)
        except Exception:
            # 嵌入 API 不可用时降级：使用零向量占位
            emb = [0.0] * 1536

        cid = f"mem_{self.collection.count() + 1}"
        self.collection.add(
            ids=[cid],
            embeddings=[emb],
            documents=[fact],
            metadatas=[metadata or {"source": "auto"}],
        )
        return True

    def recall(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """语义检索相关记忆，返回 [(事实, 相似度), ...]"""
        if self.collection.count() == 0:
            return []

        try:
            query_vec = self.embed_gen.embed(query)
        except Exception:
            return []  # 嵌入 API 不可用时跳过检索

        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, self.collection.count()),
            )
        except Exception:
            return []

        if results is None:
            return []

        facts = []
        docs = results.get("documents", [[]])[0]
        emb_list = results.get("embeddings", [[]])[0]
        for doc, emb in zip(docs, emb_list):
            if emb:
                score = cosine_similarity(query_vec, emb)
            else:
                score = 0.5
            facts.append((doc, round(score, 4)))
        return facts

    def recall_as_context(self, query: str, top_k: int = 5) -> str:
        """召回记忆并格式化为上下文"""
        facts = self.recall(query, top_k)
        if not facts:
            return ""
        lines = ["[已知用户信息/历史记忆]"]
        for fact, score in facts:
            lines.append(f"- {fact}")
        return "\n".join(lines)

    def list_all(self) -> list[str]:
        if self.collection.count() == 0:
            return []
        return self.collection.get().get("documents", [])

    def clear(self):
        ids = self.collection.get().get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    @property
    def count(self) -> int:
        return self.collection.count()


# ═══════════════════════════════════════════════════════════
# 4. 统一记忆管理器
# ═══════════════════════════════════════════════════════════

class MemoryManager:
    """
    三层记忆统一管理器。

    使用方式：
      mem = MemoryManager()
      mem.add_message("user", "什么是 RAG？")
      mem.add_message("assistant", "RAG 是...")
      context = mem.build_context("继续问 RAG 的问题")
    """

    def __init__(self):
        self.short_term = SlidingWindowMemory()
        self.summary = SummaryBufferMemory()
        self.long_term = VectorLongTermMemory()

    def add_message(self, role: str, content: str):
        self.short_term.add(role, content)
        self.summary.add(role, content)

    def build_context(self, user_input: str) -> str:
        """构建完整上下文 = 长期记忆 + 摘要 + 近期对话。"""
        parts = []

        # 1. 长期记忆（语义检索）
        lt = self.long_term.recall_as_context(user_input)
        if lt:
            parts.append(lt)

        # 2. 摘要 + 近期对话
        sm = self.summary.get_context()
        if sm:
            parts.append(sm)

        return "\n\n".join(parts)

    def extract_and_remember(self, conversation_text: str):
        """对话结束后提取+持久化长期记忆"""
        facts = self.long_term.extract_facts(conversation_text)
        saved = 0
        for fact in facts:
            if self.long_term.remember(fact):
                saved += 1
        return saved

    def clear(self):
        self.short_term.clear()
        self.summary.clear()
