"""
Agent 工具集 — 4 个工具，Agent 自主决策调用哪个。

@tool 装饰器自动生成 OpenAI function schema，按场景可选，并行执行支持。
"""
import os
import json
import math
import time
from datetime import datetime

from langchain_core.tools import tool

from eka.config import EXPORTS_DIR

# ═══════════════════════════════════════════════════════════
# Tool 1：知识库检索（核心工具）
# ═══════════════════════════════════════════════════════════

# 全局 RAG 函数引用（由 main.py 注入）
_rag_search_fn = None


def set_rag_search(fn):
    global _rag_search_fn
    _rag_search_fn = fn


@tool
def search_knowledge_base(query: str) -> str:
    """
    在本地知识库中语义搜索。
    用于检索产品信息、公司政策、技术文档等内部知识。
    返回相关文档片段及其相似度分数。
    """
    if _rag_search_fn is None:
        return "知识库未初始化。"
    results = _rag_search_fn(query)
    if not results:
        return "知识库中未找到相关信息，建议尝试其他关键词或使用 web_search。"

    lines = []
    for i, r in enumerate(results, 1):
        content = r.get("content", str(r))
        score = r.get("score", 0)
        lines.append(f"[结果 {i}] (相关度: {score:.2f})\n{content[:600]}")
    return "\n\n---\n\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Tool 2：计算器
# ═══════════════════════════════════════════════════════════

@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。
    支持：+ - * / ** sqrt() sin() cos() abs() pow() log()。
    用于数据分析、财务报表计算、统计等场景。
    """
    allowed = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "abs": abs, "pow": pow, "log": math.log, "log10": math.log10,
        "pi": math.pi, "e": math.e, "round": round,
        "sum": sum, "max": max, "min": min,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算出错: {e}。请检查表达式 '{expression}' 是否合法。"


# ═══════════════════════════════════════════════════════════
# Tool 3：Web 搜索（补充外部信息）
# ═══════════════════════════════════════════════════════════

# 模拟 Web 搜索知识库
_MOCK_WEB = {
    "transformer": json.dumps({
        "title": "Transformer 架构 (2017)",
        "source": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762",
        "summary": "Transformer 基于 Self-Attention 机制替代 RNN。核心创新：多头注意力 + 位置编码。时间复杂度 O(n²)。优势：并行训练、长距离依赖建模。",
    }, ensure_ascii=False),
    "rag": json.dumps({
        "title": "RAG — 检索增强生成 (2020)",
        "source": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/abs/2005.11401",
        "summary": "RAG 将检索与生成结合：先检索相关文档，再基于文档生成回答。核心组件：Embedding + VectorStore + Retriever + Generator。解决 LLM 幻觉和知识时效性问题。",
    }, ensure_ascii=False),
    "agent": json.dumps({
        "title": "AI Agent 综述 (2024)",
        "source": "LLM Agent 调研报告",
        "summary": "AI Agent = LLM + Tool Calling + Memory + Planning。主流框架：LangChain、LangGraph、AutoGen。关键模式：ReAct、Plan-Execute。",
    }, ensure_ascii=False),
    "mamba": json.dumps({
        "title": "Mamba 状态空间模型 (2023)",
        "source": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "url": "https://arxiv.org/abs/2312.00752",
        "summary": "Mamba 基于选择性状态空间模型，O(n) 线性时间复杂度。在长序列任务上性能接近/超过 Transformer。核心创新：选择性扫描机制 + 硬件感知算法。",
    }, ensure_ascii=False),
    "gpt": json.dumps({
        "title": "GPT-4 技术报告 (2023)",
        "source": "OpenAI GPT-4 Technical Report",
        "summary": "GPT-4 是多模态大模型。在 MMLU 基准上达 86.4%。支持 8192/32768 token 上下文。RLHF 对齐训练。在律师资格考试中超越 90% 人类考生。",
    }, ensure_ascii=False),
}


@tool
def web_search(query: str) -> str:
    """
    搜索互联网获取最新公开信息。
    用于获取知识库中没有的实时信息、行业动态、技术趋势等。
    注意：本工具使用模拟数据，实际部署时替换为真实搜索 API。
    """
    time.sleep(0.3)  # 模拟网络延迟
    for key, content in _MOCK_WEB.items():
        if key in query.lower():
            return content
    return json.dumps({
        "title": f"搜索结果: {query[:60]}",
        "summary": f"关于 '{query}' 的搜索结果：该主题可能在知识库中有更详细的内部文档，建议同时使用 search_knowledge_base。",
    }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# Tool 4：报告导出
# ═══════════════════════════════════════════════════════════

@tool
def export_report(title: str, content: str) -> str:
    """
    将分析报告导出为 Markdown 文件。
    用于保存会议纪要、分析报告、调研结论等重要内容。
    """
    filename = f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(EXPORTS_DIR, filename)
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    full_content = f"# {title}\n\n> 生成时间: {datetime.now().isoformat()}\n\n{content}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)
    return f"报告已导出: {filepath}"


# ═══════════════════════════════════════════════════════════
# 工具注册表
# ═══════════════════════════════════════════════════════════

ALL_TOOLS = [search_knowledge_base, calculator, web_search, export_report]

# 按场景的工具推荐组合
SCENARIO_TOOLS = {
    "full": ALL_TOOLS,                                    # 完整功能
    "qa": [search_knowledge_base],                         # 纯问答
    "research": [search_knowledge_base, web_search, calculator],  # 调研分析
    "report": [search_knowledge_base, web_search, export_report],  # 报告生成
}

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def get_tools_for_scenario(scenario: str = "full") -> list:
    return SCENARIO_TOOLS.get(scenario, ALL_TOOLS)