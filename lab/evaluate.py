"""evaluate.py — 评测层：把"检索到底准不准"变成可对比的数字。

面试高频：你怎么确认 RAG 的召回是准的 / 检索效果怎么量化？
标准答案就是本模块做的事：准备一批带 golden 标注的问题，跑检索，用可复现的指标打分。
  recall@k  前 k 个结果里是否命中了应该召回的内容(命中率视角)。
  MRR       第一个命中结果排在第几名的倒数均值——不仅要召回到，还要排得靠前。
  延迟       检索快不快，做策略取舍时的另一个坐标轴。

命中判定(hit)：一个召回的 chunk 算命中 golden，当且仅当
  它属于 golden_doc，且它的正文里出现了 golden_keywords 中的任意一个。
这样判定既认"文档对了"、又认"具体段落对了"，比只看文档 id 更严格、更接近真实体验。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .chunking import Chunk


@dataclass
class QueryResult:
    question: str
    hit_rank: int              # 第一个命中 chunk 的名次(1 起算)，整个结果都没命中记 0
    latency_ms: float
    retrieved_ids: list[str] = field(default_factory=list)


def is_relevant(chunk: Chunk, golden_doc: str, golden_keywords: list[str]) -> bool:
    """单个 chunk 是否命中 golden：文档匹配 且 命中任一关键词(大小写不敏感)。"""
    if chunk.doc_id != golden_doc:
        return False
    text = chunk.text.lower()
    return any(kw.lower() in text for kw in golden_keywords)


def first_hit_rank(retrieved: list[tuple[Chunk, float]],
                   golden_doc: str, golden_keywords: list[str]) -> int:
    """返回第一个命中 chunk 的名次(1 起算)；一个都没命中返回 0。"""
    for rank, (chunk, _score) in enumerate(retrieved, start=1):
        if is_relevant(chunk, golden_doc, golden_keywords):
            return rank
    return 0


def recall_at_k(results: list[QueryResult], k: int) -> float:
    """recall@k：命中发生在前 k 名以内的问题占比。
    注：本评测每题只有一个 golden 目标，所以这里的 recall@k 等价于'命中率 hit@k'——
    面试里说清这个前提即可，是很自然的单答案检索评测口径。"""
    if not results:
        return 0.0
    hit = sum(1 for r in results if 0 < r.hit_rank <= k)
    return hit / len(results)


def mrr(results: list[QueryResult]) -> float:
    """MRR(Mean Reciprocal Rank)：命中名次倒数的均值，没命中记 0。
    它对'排在第 1 名'和'排在第 5 名'区别对待——衡量的是排序质量，不只是召回与否。"""
    if not results:
        return 0.0
    return sum((1.0 / r.hit_rank if r.hit_rank else 0.0) for r in results) / len(results)


def latency_stats(results: list[QueryResult]) -> dict[str, float]:
    """延迟统计：平均 / p50 / p95(毫秒)。p95 比平均更能暴露长尾抖动。"""
    xs = sorted(r.latency_ms for r in results)
    if not xs:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    pick = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"avg": sum(xs) / len(xs), "p50": pick(0.5), "p95": pick(0.95)}


def evaluate(search_fn: Callable[[str], list], eval_set: list[dict],
             k: int = 3) -> tuple[list[QueryResult], dict]:
    """对整个评测集跑一遍检索并汇总指标。

    search_fn(question) -> [(chunk, score), ...]  是被测的"某一套配置"的检索函数，
    调用方(run_matrix.py)通过闭包把"切分策略 × 检索策略 × 是否 rerank"都固化进去，
    评测层不关心内部细节，只管计时、判命中、算指标——职责单一，好复用也好测。
    """
    results: list[QueryResult] = []
    for item in eval_set:
        t0 = time.perf_counter()
        retrieved = search_fn(item["question"])
        latency = (time.perf_counter() - t0) * 1000
        rank = first_hit_rank(retrieved, item["golden_doc"], item["golden_keywords"])
        results.append(QueryResult(
            question=item["question"], hit_rank=rank, latency_ms=latency,
            retrieved_ids=[c.chunk_id for c, _ in retrieved],
        ))
    metrics = {
        "recall@1": recall_at_k(results, 1),
        f"recall@{k}": recall_at_k(results, k),
        "mrr": mrr(results),
        "latency": latency_stats(results),
    }
    return results, metrics
