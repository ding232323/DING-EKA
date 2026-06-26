"""
EKA — Enterprise Knowledge Agent
企业知识库智能问答 Agent

技术栈：OpenAI API + LangChain + ChromaDB + 手写实现全面对比

四大核心模块：
  1. RAG 全链路（手写 + LangChain 双版本 + 性能对比）
  2. 三层记忆架构（滑动窗口 / 摘要缓冲 / 向量长期记忆）
  3. Tool Calling Agent（4 工具 + 并行执行 + Streaming）
  4. LLM-as-Judge 评估（忠实度 + 相关性 + 完整性）
"""

__version__ = "1.0.0"