<p align="center">
  <h1 align="center">EKA — Enterprise Knowledge Agent</h1>
  <p align="center">企业知识库智能问答 Agent · 从零手写 AI Agent 全栈</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license">
</p>

---

## 项目简介

EKA 是一个面向**企业知识库**的智能问答 Agent 系统。它从底层手写实现了现代 LLM Agent 的每一个核心模块，并在旁边标注了 LangChain 等价实现。

---

## 架构总览

```
┌─────────────────────────────────────────────────────┐
│                    用户界面 (CLI)                      │
│              python -m eka.main [--demo|--eval|...]  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 ReAct Agent (手写)                     │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Thought → Action → Observation → ... → Answer  │ │
│  │         │                  │                     │ │
│  │   并行工具执行         流式输出                    │ │
│  │   (ThreadPool)      (Streaming)                  │ │
│  └─────────────────────────────────────────────────┘ │
└──────┬──────────┬───────────┬───────────┬───────────┘
       │          │           │           │
┌──────▼──┐ ┌────▼───┐ ┌─────▼────┐ ┌────▼──────┐
│  RAG    │ │ 记忆层  │ │  安全层   │ │  评估层    │
│ 双引擎   │ │ 三层架构 │ │  护栏     │ │ LLM-Judge │
└─────────┘ └────────┘ └──────────┘ └───────────┘
```

---

## 四大核心模块

### 1. RAG 检索增强生成（双引擎）

| 引擎 | 实现方式 | 用途 |
|------|---------|------|
| `HandWrittenRAG` | 手写分块 + 手写 Embedding + 手写向量检索 | 展示底层原理 |
| `LangChainRAG` | LangChain 全家桶 | 展示框架掌握 |

- 三种手写分块策略：递归字符分割 / 语义分割 / 滑动窗口
- Embedding 支持三种后端：本地 Sentence-Transformers（ModelScope 镜像） / HuggingFace API / OpenAI 兼容 API
- 手写余弦相似度 + 暴力 Top-K 检索
- 内置 `--compare` 模式可做双引擎性能对比

### 2. 三层记忆架构

| 层级 | 实现 | 核心思想 |
|------|------|---------|
| 短期记忆 | `SlidingWindowMemory` | 保留最近 N 条消息，O(1) 插入 |
| 中期记忆 | `SummaryBufferMemory` | 旧消息 → LLM 压缩摘要 |
| 长期记忆 | `VectorLongTermMemory` | ChromaDB 持久化 + 语义检索 |

由 LLM 在对话结束时提取"值得记住的事实"（用户偏好、关键决策、重要数据），写入前去重（相似度 > 0.95 跳过）。

### 3. Tool Calling Agent

- **手写 ReAct 循环**：Thought → Action → Observation → ... → Final Answer
- **并行工具执行**：单轮多工具调用时 `ThreadPoolExecutor` 并发执行
- **流式输出**：自动处理 Windows GBK 编码兼容
- **三层容错**：工具报错返还 LLM → LLM 自我修正 → 连续 3 轮失败即中止
- **四种内置工具**：

| 工具 | 功能 |
|------|------|
| `search_knowledge_base` | RAG 检索内部知识库 |
| `calculator` | 安全数学计算（白名单函数） |
| `web_search` | 外部信息检索（模拟） |
| `export_report` | Markdown 报告导出 |

工具可按场景配置：`full`（全部）/ `qa`（仅 RAG）/ `research`（RAG + 搜索 + 计算）/ `report`（RAG + 搜索 + 导出）。

### 4. LLM-as-Judge 评估

- **Binary Judge**：PASS/FAIL 二元判定
- **Multi-Dimension Judge**：忠实度 / 相关性 / 完整性（1-5 分）
- **Multi-Judge Vote**：三名法官投票共识，减少单模型偏差
- **LatencyStats**：P50 / P95 / P99 延迟统计

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA（可选，用于本地 Embedding 加速）

### 安装依赖

```bash
# 核心依赖
pip install openai python-dotenv chromadb

# LangChain（用于双引擎对比）
pip install langchain-core langchain-openai langchain-community langchain-text-splitters

# 本地 Embedding（国内推荐）
pip install sentence-transformers modelscope
```

### 配置

在项目根目录创建 `.env` 文件：

```env
# LLM 配置（兼容 OpenAI API 的服务皆可）
LLM_MODEL=deepseek-v4-pro
OPENAI_API_KEY=...
OPENAI_API_URL=https://api.deepseek.com/v1

# Embedding 配置
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5   # 带 / 自动识别为本地模型
EMBEDDING_DEVICE=cpu                       # 或 cuda
HF_ENDPOINT=https://hf-mirror.com          # 国内镜像加速
```

### 运行

```bash
# 交互模式（默认）
python -m eka.main

# 演示模式 — 运行 5 个预设场景
python -m eka.main --demo

# 单轮问答
python -m eka.main --question "小米14 Ultra的摄像头参数？"

# 评估模式 — LLM-as-Judge 评测
python -m eka.main --eval

# 性能对比 — 手写 vs LangChain RAG
python -m eka.main --compare
```

### 交互命令

进入交互模式后，可使用以下命令：
- `:help` — 显示帮助
- `:mode <scenario>` — 切换工具场景（`qa` / `research` / `report` / `full`）
- `:clear` — 清除对话记忆
- `:export` — 导出当前对话到 Markdown
- `:quit` — 退出

---

## 项目结构

```
EKA/
├── .env                    # 环境变量（API Key 等）
├── README.md
└── eka/
    ├── __init__.py         # 包入口，版本号
    ├── main.py             # CLI 入口（5 种运行模式）
    ├── agent.py            # ReAct Agent 主循环
    ├── config.py           # 全局配置中心
    ├── rag.py              # RAG 双引擎 + 管理 + 对比
    ├── chunking.py         # 3 种手写分块策略
    ├── embedding.py        # Embedding 生成 + 向量检索
    ├── memory.py           # 三层记忆架构
    ├── tools.py            # Agent 工具集
    ├── guard.py            # 输入/输出安全护栏
    ├── evaluation.py       # LLM-as-Judge 评估
    └── data/
        ├── knowledge/      # 知识库源文件（.txt）
        │   ├── company_policy.txt
        │   └── xiaomi_products.txt
        └── exports/        # 导出报告目录
```

---



## License

MIT