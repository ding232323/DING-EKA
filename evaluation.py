"""
LLM-as-Judge 评估 + 运行时指标收集。

包含三维评估、多裁判投票、延迟分位数统计。
"""
import json
import time
from openai import OpenAI

from eka.config import LLM_CONFIG

JUDGE_CLIENT = OpenAI(
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
)

# ═══════════════════════════════════════════════════════════
# 单维评审
# ═══════════════════════════════════════════════════════════

BINARY_PROMPT = """你是严格的事实核查员。判断回答是否满足标准。

标准：{criterion}
问题：{question}
回答：{answer}

只回答 PASS 或 FAIL。"""


def binary_judge(question: str, answer: str, criterion: str) -> str:
    """二元评审"""
    try:
        resp = JUDGE_CLIENT.chat.completions.create(
            model=LLM_CONFIG["model"],
            max_completion_tokens=50,
            messages=[{
                "role": "system",
                "content": BINARY_PROMPT.format(
                    criterion=criterion, question=question, answer=answer[:3000],
                ),
            }],
        )
        text = resp.choices[0].message.content.upper().strip()
        return "PASS" if "PASS" in text else "FAIL"
    except Exception as e:
        return f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════
# 三维评审
# ═══════════════════════════════════════════════════════════

MULTI_DIM_PROMPT = """你是评估专家。从以下三个维度给回答打分（1-5 分）：

1. **忠实度 (Faithfulness)**：回答内容是否基于提供的上下文？有无编造（幻觉）？
2. **相关性 (Relevance)**：回答是否切中问题核心？有无答非所问？
3. **完整性 (Completeness)**：是否覆盖问题的所有关键方面？

问题：{question}
回答：{answer}

输出 JSON：{{"faithfulness": X, "relevance": X, "completeness": X}}"""


def multi_dimension_judge(question: str, answer: str) -> dict:
    """三维评审"""
    try:
        resp = JUDGE_CLIENT.chat.completions.create(
            model=LLM_CONFIG["model"],
            max_completion_tokens=200,
            messages=[{
                "role": "system",
                "content": MULTI_DIM_PROMPT.format(
                    question=question, answer=answer[:3000],
                ),
            }],
        )
        text = resp.choices[0].message.content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception:
        return {"faithfulness": 3, "relevance": 3, "completeness": 3}


# ═══════════════════════════════════════════════════════════
# 多裁判投票
# ═══════════════════════════════════════════════════════════

def multi_judge_vote(question: str, answer: str, criterion: str,
                     n_judges: int = 3) -> dict:
    """多裁判投票降低偏差"""
    votes = []
    for _ in range(n_judges):
        result = binary_judge(question, answer, criterion)
        votes.append(result)

    pass_count = sum(1 for v in votes if v == "PASS")
    fail_count = n_judges - pass_count
    return {
        "votes": votes,
        "pass": pass_count,
        "fail": fail_count,
        "consensus": "PASS" if pass_count > fail_count else "FAIL",
    }


# ═══════════════════════════════════════════════════════════
# 延迟统计
# ═══════════════════════════════════════════════════════════

class LatencyStats:
    """运行时延迟统计"""

    def __init__(self):
        self.samples: list[float] = []

    def record(self, duration_ms: float):
        self.samples.append(duration_ms)

    @property
    def summary(self) -> dict:
        if not self.samples:
            return {"count": 0}
        sorted_lat = sorted(self.samples)
        n = len(sorted_lat)
        return {
            "count": n,
            "avg_ms": round(sum(sorted_lat) / n, 1),
            "p50_ms": sorted_lat[int(n * 0.50)],
            "p95_ms": sorted_lat[int(n * 0.95)] if n > 20 else sorted_lat[-1],
            "p99_ms": sorted_lat[int(n * 0.99)] if n > 100 else sorted_lat[-1],
            "min_ms": sorted_lat[0],
            "max_ms": sorted_lat[-1],
        }


# ═══════════════════════════════════════════════════════════
# 评估运行器
# ═══════════════════════════════════════════════════════════

def run_evaluation(agent_fn, test_cases: list[dict]) -> dict:
    """
    完整评估流程：延迟统计 + 多维评审 + 用例通过率

    test_cases 格式：
      [{"question": "...", "criterion": "..."}, ...]
    """
    lat_stats = LatencyStats()
    multi_scores = []
    binary_results = []

    for case in test_cases:
        question = case["question"]
        criterion = case.get("criterion", "回答应准确、相关、完整")

        print(f"\n{'='*50}")
        print(f"测试: {question}")

        # 执行
        t0 = time.perf_counter()
        result = agent_fn(question)
        elapsed = (time.perf_counter() - t0) * 1000
        lat_stats.record(elapsed)

        answer = result.get("answer", str(result))

        # 二元评审
        binary = binary_judge(question, answer, criterion)
        binary_results.append({"question": question, "result": binary})

        # 多维评审
        multi = multi_dimension_judge(question, answer)
        multi_scores.append({"question": question, **multi})

        # 打印结果
        print(f"  二元: {binary}")
        print(f"  多维: 忠实度={multi['faithfulness']} "
              f"相关性={multi['relevance']} "
              f"完整性={multi['completeness']}")
        print(f"  延迟: {elapsed:.0f}ms")

    # 汇总
    pass_count = sum(1 for r in binary_results if r["result"] == "PASS")
    avg_multi = {
        dim: round(sum(m[dim] for m in multi_scores) / len(multi_scores), 2)
        for dim in ["faithfulness", "relevance", "completeness"]
    }

    return {
        "pass_rate": f"{pass_count}/{len(binary_results)}",
        "avg_scores": avg_multi,
        "latency": lat_stats.summary,
        "details": {
            "binary": binary_results,
            "multi_dimension": multi_scores,
        },
    }
