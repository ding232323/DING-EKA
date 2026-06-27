"""
EKA — Enterprise Knowledge Agent CLI

用法:
    python -m eka.main                           # 交互模式（默认 LangChain）
    python -m eka.main --engine handwritten      # 交互模式（手写引擎）
    python -m eka.main --demo                    # 演示模式
    python -m eka.main --eval                    # 评估模式
    python -m eka.main --compare                 # 手写 vs LangChain RAG 性能对比
    python -m eka.main --question "问题"         # 单次问答
"""
import sys
import json

# Windows 终端 UTF-8 支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from eka.config import KNOWLEDGE_DIR
from eka.rag import RAGManager, compare_rag_performance
from eka.memory import MemoryManager
from eka.tools import set_rag_search
from eka.agent import EnterpriseKnowledgeAgent
from eka.guard import InputGuard, OutputGuard, AuditLogger
from eka.evaluation import run_evaluation


def init_system(rag_engine: str = "langchain"):
    """初始化：加载 RAG 索引 + 注入工具（RAG 失败不阻断）"""
    print(f"  RAG 引擎: {rag_engine}")
    rag = RAGManager(engine=rag_engine)
    try:
        stats = rag.init()
        print(f"  知识库: {stats.get('files', 0)} 文件, "
              f"{stats.get('chunks', 0)} 分块, "
              f"平均 {stats.get('avg_chunk_len', 0):.0f} 字符/块")
        set_rag_search(rag.search)
    except Exception as e:
        print(f"  知识库: 索引失败 ({e})，知识库检索不可用")
        print(f"  其他工具 (calculator / web_search / export_report) 仍可正常使用")

    memory = MemoryManager()
    agent = EnterpriseKnowledgeAgent(memory_manager=memory)
    return rag, memory, agent


# ── 交互模式 ──────────────────────────────────────

def interactive_mode(engine: str = "langchain"):
    print("\n" + "=" * 55)
    print("  EKA — 企业知识库智能助手 v1.0")
    print("  输入 'quit' 退出 | 'clear' 清除记忆 | 'stats' 查看状态")
    print("        'engine' 切换 RAG 引擎 | 'compare' 性能对比")
    print("=" * 55)

    rag, mem, agent = init_system(engine)
    guard = InputGuard()
    audit = AuditLogger()
    out_guard = OutputGuard()

    while True:
        try:
            user_input = input("\n提问: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break
        if user_input.lower() == "clear":
            mem.clear()
            print("记忆已清除")
            continue
        if user_input.lower() == "stats":
            print(f"短期记忆: {mem.short_term.message_count} 条")
            print(f"长期记忆: {mem.long_term.count} 条")
            print(f"RAG 索引: {rag.engine.retriever.size if hasattr(rag.engine, 'retriever') else 'N/A'} 条")
            continue
        if user_input.lower() == "engine":
            new_engine = "handwritten" if rag.engine_name == "langchain" else "langchain"
            print(f"\n切换 RAG 引擎: {rag.engine_name} → {new_engine}")
            rag, mem, agent = init_system(new_engine)
            continue
        if user_input.lower() == "compare":
            from eka.config import KNOWLEDGE_DIR as kd
            result = compare_rag_performance("小米14 Ultra 摄像头配置", kd)
            print(f"\n{'引擎':<15} {'索引耗时':<12} {'检索耗时':<12} {'结果数'}")
            print("-" * 55)
            for eng, data in result.items():
                if eng == "comparison":
                    continue
                print(f"{eng:<15} {data['indexing_ms']:<8.1f}ms   "
                      f"{data['search_ms']:<8.1f}ms   {data['num_results']}")
            comp = result["comparison"]
            print(f"\n加速比: 索引 {comp['indexing_speedup']}, 检索 {comp['search_speedup']}")
            continue

        # 安全检查
        check = guard.check(user_input)
        audit.log("input_check", check)
        if not check["safe"]:
            print(f"\n[!] 输入被拦截 (可疑度: {check['suspicion']})")
            if check["blocked_topic"]:
                print(f"  禁止话题: {check['blocked_topic']}")
            continue

        # 执行
        print()
        result = agent.run(user_input, stream=True)

        # 输出脱敏 + 记忆提取
        answer = result.get("answer", "")
        sanitized = out_guard.sanitize(answer)
        audit.log("output_sanitize", {"len": len(answer)})

        # 提取长期记忆
        saved = mem.extract_and_remember(f"问: {user_input}\n答: {answer[:2000]}")
        if saved:
            print(f"\n[已保存 {saved} 条长期记忆]")

        # 工具使用统计
        if result.get("tool_calls"):
            tools_used = result.get("tools_used", [])
            tool_names = set(t["tool"] for t in tools_used)
            print(f"[共调用 {result['tool_calls']} 次工具: {', '.join(tool_names)}]")


# ── 演示模式 ──────────────────────────────────────

def demo_mode(engine: str = "langchain"):
    """演示：4 个典型场景"""
    rag, mem, agent = init_system(engine)

    demos = [
        {
            "title": "场景 1：产品信息查询",
            "question": "小米14 Ultra 的摄像头配置和价格是多少？不同配置的价格差异大吗？",
        },
        {
            "title": "场景 2：数据分析 + 计算",
            "question": "小米 SU7 三个版本的价格分别是 215900、245900、299900 元。帮我算一下三个版本的平均价格，以及最高配比最低配贵多少百分比。",
        },
        {
            "title": "场景 3：政策查询",
            "question": "公司年假是怎么算的？如果我刚入职满 3 年，每年有多少天年假？",
        },
        {
            "title": "场景 4：知识综合分析",
            "question": "对比一下小米手环9和Redmi Watch 4的区别，推荐给运动场景的用户。",
        },
        {
            "title": "场景 5：外部知识补充",
            "question": "什么是 Transformer 架构？和最近流行的 Mamba 有什么区别？",
        },
    ]

    print(f"\n{'=' * 55}")
    print("  EKA 演示 — 5 个典型企业知识库场景")
    print(f"{'=' * 55}")

    for i, demo in enumerate(demos, 1):
        print(f"\n{'─' * 55}")
        print(f"\n【{demo['title']}】")
        print(f"问: {demo['question']}")

        t0 = __import__("time").perf_counter()
        result = agent.run(demo["question"], stream=True)
        elapsed = (__import__("time").perf_counter() - t0) * 1000

        print(f"\n[{elapsed:.0f}ms] 工具调用: {result.get('tool_calls', 0)} 次 | "
              f"轮次: {result.get('turns', 0)}")

        # 提取记忆
        saved = mem.extract_and_remember(
            f"问: {demo['question']}\n答: {result.get('answer', '')[:2000]}"
        )
        if saved:
            print(f"[Memory] 已保存 {saved} 条长期记忆")

    # 显示长期记忆
    print(f"\n{'─' * 55}")
    print(f"\n长期记忆库 ({mem.long_term.count} 条):")
    for fact in mem.long_term.list_all():
        print(f"  - {fact}")


# ── 评估模式 ──────────────────────────────────────

def eval_mode(engine: str = "langchain"):
    print("EKA 评估模式 — LLM-as-Judge\n")
    rag, mem, agent = init_system(engine)

    test_cases = [
        {
            "question": "小米14 Ultra 的电池容量和充电速度是多少？",
            "criterion": "回答应准确包含电池容量(5300mAh)和充电规格(90W有线/80W无线)",
        },
        {
            "question": "公司年假满 3 年有多少天？",
            "criterion": "回答应准确说明满 3 年享 10 天带薪年假",
        },
        {
            "question": "小米 SU7 Max 版的续航和加速性能？",
            "criterion": "回答应包含 Max 版续航 810km 和 0-100km/h 2.78 秒",
        },
    ]

    results = run_evaluation(agent.run_non_streaming, test_cases)

    print(f"\n{'=' * 50}")
    print("评估总结")
    print(f"{'=' * 50}")
    print(f"通过率: {results['pass_rate']}")
    print(f"平均分: 忠实度={results['avg_scores']['faithfulness']} "
          f"相关性={results['avg_scores']['relevance']} "
          f"完整性={results['avg_scores']['completeness']}")
    lat = results['latency']
    print(f"延迟: 平均 {lat.get('avg_ms', 0)}ms, "
          f"P50={lat.get('p50_ms', 0)}ms, P95={lat.get('p95_ms', 0)}ms")


# ── 性能对比模式 ──────────────────────────────────

def compare_mode():
    print("EKA RAG 性能对比 — 手写版 vs LangChain 版\n")
    print("索引中...")
    result = compare_rag_performance("小米14 Ultra 摄像头配置", KNOWLEDGE_DIR)

    print(f"\n{'引擎':<15} {'索引耗时':<12} {'检索耗时':<12} {'结果数'}")
    print("-" * 55)
    for engine, data in result.items():
        if engine == "comparison":
            continue
        print(f"{engine:<15} {data['indexing_ms']:<8.1f}ms   "
              f"{data['search_ms']:<8.1f}ms   {data['num_results']}")

    comp = result["comparison"]
    print(f"\n加速比: 索引 {comp['indexing_speedup']}, 检索 {comp['search_speedup']}")

    print("手写版通常更快（无框架开销），LangChain 版在开发效率和可维护性上更优。")


# ── 单次问答模式 ──────────────────────────────────

def single_question(question: str, engine: str = "langchain"):
    rag, mem, agent = init_system(engine)
    result = agent.run(question, stream=True)
    print(f"\n工具调用: {result.get('tool_calls', 0)} 次, "
          f"轮次: {result.get('turns', 0)}")


# ── 入口 ──────────────────────────────────────────

def _parse_engine(default: str = "langchain") -> str:
    """从命令行参数中提取 --engine / -e 的值"""
    for flag in ("--engine", "-e"):
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                val = sys.argv[idx + 1].lower()
                if val in ("langchain", "handwritten"):
                    return val
                print(f"无效引擎 '{val}'，有效值: langchain, handwritten。使用默认值。")
    return default

def main():
    engine = _parse_engine()
    if "--demo" in sys.argv:
        demo_mode(engine)
    elif "--eval" in sys.argv:
        eval_mode(engine)
    elif "--compare" in sys.argv:
        compare_mode()
    elif "--question" in sys.argv:
        idx = sys.argv.index("--question")
        if idx + 1 < len(sys.argv):
            single_question(sys.argv[idx + 1], engine)
        else:
            print("用法: python -m eka.main --question \"你的问题\"")
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
