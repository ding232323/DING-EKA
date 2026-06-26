"""
EKA 集中配置
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────
LLM_CONFIG = {
    "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_API_URL", "https://api.deepseek.com/v1"),
    "temperature": 0.3,
    "max_tokens": 2048,
}

# ── Embedding ─────────────────────────────────────
# provider:
#   huggingface_local → 本地 sentence-transformers（国内推荐，需 HF 镜像下载）
#   huggingface       → HF Inference API（需翻墙 + 模型支持 serverless）
#   openai            → OpenAI 兼容 API
# 自动检测：含 "/" 的模型名默认走 huggingface_local（国内网络 HF API 不可达）
_embedding_model = os.getenv("EMBEDDING_MODEL", "")
_embedding_default = "huggingface_local" if "/" in _embedding_model else "openai"
EMBEDDING_CONFIG = {
    "model": _embedding_model,
    "provider": os.getenv("EMBEDDING_PROVIDER") or _embedding_default,
    "api_key": os.getenv("HF_TOKEN"),
    "base_url": os.getenv("EMBEDDING_API_URL", "https://api-inference.huggingface.co"),
    "device": os.getenv("EMBEDDING_DEVICE", "cpu"),          # cpu / cuda
    "hf_endpoint": os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),  # 国内镜像
}

# ── RAG ───────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K_RETRIEVAL = 4

# ── Memory ────────────────────────────────────────
MAX_WINDOW_MESSAGES = 20        # 滑动窗口大小
SUMMARY_BUFFER_SIZE = 4000      # 摘要缓冲 token 限制
LONG_TERM_COLLECTION = "eka_long_term_memory"

# ── Agent ─────────────────────────────────────────
MAX_AGENT_TURNS = 8
TOOL_TIMEOUT_SECONDS = 10

# ── Safety ────────────────────────────────────────
BLOCKED_TOPICS = ["hack", "crack", "malware", "weapon", "exploit", "illegal"]

# ── Paths ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")

os.makedirs(EXPORTS_DIR, exist_ok=True)