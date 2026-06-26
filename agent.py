"""
Tool Calling Agent 主循环 — ReAct + 并行执行 + Streaming + 错误恢复

Agent 执行流：
  User Input → [Memory Context 注入] → LLM 决策 →
    ├─ 有 tool_calls → 并行执行 → 结果返回 LLM → 继续决策
    └─ 无 tool_calls → Streaming 输出 → 结束
"""
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from eka.config import LLM_CONFIG, MAX_AGENT_TURNS, TOOL_TIMEOUT_SECONDS
from eka.memory import MemoryManager
from eka.tools import ALL_TOOLS, TOOLS_BY_NAME

# ═══════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是企业知识库智能助手（EKA），帮助用户快速获取和分析信息。

## 核心能力
1. **知识检索**：使用 search_knowledge_base 查阅内部知识库
2. **Web 搜索**：使用 web_search 获取最新公开信息
3. **数学计算**：使用 calculator 进行精确计算
4. **报告导出**：使用 export_report 保存分析报告

## 工作原则
- 优先使用 search_knowledge_base 检索内部知识，补充使用 web_search
- 需要数据计算时，必须使用 calculator 工具而非心算
- 引用信息时要注明来源（知识库/Web/计算）
- 无法获取的信息如实告知，不要编造

## 回答格式
- 用 Markdown 格式组织回答
- 关键数据用 **加粗** 强调
- 对比信息用表格呈现"""


# ═══════════════════════════════════════════════════════════
# Agent 类
# ═══════════════════════════════════════════════════════════

class EnterpriseKnowledgeAgent:
    """Tool Calling Agent 主类。"""

    def __init__(self, memory_manager: MemoryManager | None = None,
                 tools: list | None = None, scenario: str = "full"):
        self.client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
        )
        self.model = LLM_CONFIG["model"]
        self.memory = memory_manager or MemoryManager()

        if tools is not None:
            self.tools = tools
        else:
            from eka.tools import get_tools_for_scenario
            self.tools = get_tools_for_scenario(scenario)

        self.tool_map = {t.name: t for t in self.tools}
        self.turn_count = 0
        self.tool_call_count = 0
        self.fail_count = 0

    # ── 单轮 LLM 调用 ──────────────────────────────

    def _call_llm(self, messages: list[dict]) -> dict:
        """调用 LLM，支持 Tool Calling"""
        params = {
            "model": self.model,
            "max_completion_tokens": LLM_CONFIG["max_tokens"],
            "messages": messages,
        }
        if self.tools:
            # 转换 LangChain @tool → OpenAI function schema
            params["tools"] = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.model_json_schema(),
                },
            } for t in self.tools]
        return self.client.chat.completions.create(**params)

    # ── 并行工具执行 ────────────────────────────────

    def _execute_tools(self, tool_calls: list) -> list[dict]:
        """并行执行多个工具调用。"""
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool = self.tool_map.get(tool_name)
                if tool:
                    future = executor.submit(
                        self._safe_tool_call, tool, tool_args
                    )
                    futures[future] = tc

            for future in as_completed(futures):
                tc = futures[future]
                try:
                    result = str(future.result(timeout=TOOL_TIMEOUT_SECONDS))
                except Exception as e:
                    result = f"工具执行失败: {e}。请尝试其他方式获取信息。"

                results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": result,
                })

        # 维持原始顺序
        tc_ids = [tc.id for tc in tool_calls]
        results.sort(key=lambda r: tc_ids.index(r["tool_call_id"]))
        return results

    def _safe_tool_call(self, tool, args: dict) -> str:
        """安全的工具调用，返回字符串结果"""
        try:
            result = tool.invoke(args)
            return str(result)
        except Exception as e:
            return f"工具 '{tool.name}' 执行异常: {e}"

    # ── Streaming 输出 ──────────────────────────────

    def _stream_final_answer(self, messages: list[dict]) -> str:
        """流式输出最终回答（自动处理 Windows GBK 编码问题）"""
        import sys

        stream = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=LLM_CONFIG["max_tokens"],
            messages=messages,
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_answer += token
                # Windows GBK 编码安全输出
                try:
                    print(token, end="", flush=True)
                except UnicodeEncodeError:
                    print(token.encode("gbk", errors="replace").decode("gbk"),
                          end="", flush=True)
        print()
        return full_answer

    # ── 主循环 ──────────────────────────────────────

    def run(self, user_input: str, stream: bool = True) -> dict:
        """Agent 主循环：ReAct 模式。用户问题不确定性高，需要根据工具结果灵活调整。"""
        self.turn_count = 0
        self.tool_call_count = 0
        self.fail_count = 0

        # 构建初始消息
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 注入记忆上下文
        mem_context = self.memory.build_context(user_input)
        if mem_context:
            messages.append({
                "role": "system",
                "content": f"[历史上下文]\n{mem_context}",
            })

        messages.append({"role": "user", "content": user_input})

        tool_results_summary = []

        for turn in range(1, MAX_AGENT_TURNS + 1):
            self.turn_count = turn

            response = self._call_llm(messages)
            msg = response.choices[0].message

            # 无 tool_calls → 最终回答
            if not msg.tool_calls:
                if stream:
                    final = self._stream_final_answer(messages)
                else:
                    final = msg.content
                    try:
                        print(final)
                    except UnicodeEncodeError:
                        print(final.encode("gbk", errors="replace").decode("gbk"))

                # 更新记忆
                self.memory.add_message("user", user_input)
                self.memory.add_message("assistant", final)

                return {
                    "answer": final,
                    "turns": turn,
                    "tool_calls": self.tool_call_count,
                    "tools_used": tool_results_summary,
                }

            # 有 tool_calls → 执行工具
            tool_names = [tc.function.name for tc in msg.tool_calls]
            print(f"\n  [Agent 第 {turn} 轮] 调用工具: {', '.join(tool_names)}")

            # 添加 assistant 消息（含 tool_calls）
            messages.append(msg.model_dump())

            # 并行执行
            tool_results = self._execute_tools(msg.tool_calls)
            self.tool_call_count += len(tool_results)

            for tr, tc in zip(tool_results, msg.tool_calls):
                self.tool_call_count += 1
                content_preview = tr["content"][:80].replace("\n", " ")
                print(f"    → {tc.function.name}: {content_preview}...")
                tool_results_summary.append({
                    "tool": tc.function.name,
                    "args": tc.function.arguments,
                    "result_len": len(tr["content"]),
                })

            # 检查是否全部失败
            failed = sum(1 for tr in tool_results
                        if "失败" in tr["content"] or "出错" in tr["content"])
            if failed == len(tool_results):
                self.fail_count += 1
                if self.fail_count >= 3:
                    return {
                        "answer": "抱歉，连续多次工具调用失败，暂时无法完成您的请求。",
                        "turns": turn,
                        "tool_calls": self.tool_call_count,
                        "tools_used": tool_results_summary,
                        "error": "连续工具调用失败",
                    }
            else:
                self.fail_count = 0

            messages.extend(tool_results)

        # 达到最大轮数
        return {
            "answer": "抱歉，处理您的请求超过了最大推理步数。请尝试简化问题。",
            "turns": MAX_AGENT_TURNS,
            "tool_calls": self.tool_call_count,
            "tools_used": tool_results_summary,
            "error": "达到最大轮数",
        }

    def run_non_streaming(self, user_input: str) -> dict:
        """非流式运行（用于评估）"""
        return self.run(user_input, stream=False)

    @property
    def stats(self) -> dict:
        return {
            "turns": self.turn_count,
            "tool_calls": self.tool_call_count,
            "fail_count": self.fail_count,
        }
