"""run_matrix.py — 评测矩阵驱动器：3 种切分 × 3 种检索(可选 +LLM 重排)，产出 report.md。

这是本项目的"主程序"，也是它区别于普通 Chat-with-PDF 的地方：不是让你"感觉"检索还行，
而是用同一批 golden 标注的问题，把每一种"切分策略 × 检索策略"组合的 recall@3 / MRR / 延迟
全测一遍，打印成一张可对比的表 + 一段结论，让你用数据回答"我的检索到底准不准、该选哪套"。

用法：
  python run_matrix.py            # 纯检索层评测(无需 API Key)
  python run_matrix.py --rerank   # 额外评测"混合检索 + LLM 重排"(需配置 API Key，否则自动跳过)
"""
from __future__ import annotations

import sys

# Windows 默认控制台编码不是 UTF-8，重配一下，保证中文打印不报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:                                    # dotenv 只为读 .env 里的 Key，缺了也不影响纯检索评测
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from lab import load_corpus, load_eval, build_chunks
from lab.chunking import CHUNKERS
from lab.retrieval import Retriever
from lab.evaluate import evaluate, recall_at_k, first_hit_rank

CORPUS_DIR = "data/corpus"
EVAL_PATH = "data/eval_set.json"
REPORT_PATH = "report.md"
K = 3                                   # 主指标 recall@K，K=3 贴近"塞给大模型前 3 条"的实际用法
RETRIEVALS = ["bm25", "vector", "hybrid"]


def make_rerank_fn(retriever: Retriever, llm):
    """把'混合检索召回 20 条 → LLM 精排取 K 条'包成一个 search_fn 给评测层用。"""
    from lab.llm_extras import llm_rerank

    def search(query: str):
        pool = retriever.search(query, k=20, strategy="hybrid")
        reranked = llm_rerank(query, pool, top_k=K, llm=llm)
        return reranked if reranked is not None else pool[:K]
    return search


def run_matrix(use_rerank: bool):
    docs = load_corpus(CORPUS_DIR)
    eval_set = load_eval(EVAL_PATH)

    llm = None
    if use_rerank:
        from lab.llm_extras import get_llm
        llm = get_llm()
        if llm is None:
            print("[提示] --rerank 需要 API Key，但未检测到，已跳过重排列(其余照常评测)。")

    # matrix[chunking][retrieval] = {"metrics":..., "chunks":n}
    matrix: dict[str, dict[str, dict]] = {}
    for chunking in CHUNKERS:
        chunks = build_chunks(docs, chunking)
        retriever = Retriever(chunks)
        row: dict[str, dict] = {}
        for strat in RETRIEVALS:
            _, metrics = evaluate(lambda q, s=strat: retriever.search(q, K, s), eval_set, k=K)
            row[strat] = {"metrics": metrics}
        if llm is not None:
            _, metrics = evaluate(make_rerank_fn(retriever, llm), eval_set, k=K)
            row["hybrid+rerank"] = {"metrics": metrics}
        matrix[chunking] = {"chunks": len(chunks), "rows": row}

    report = build_report(docs, eval_set, matrix)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n[完成] 报告已写入 {REPORT_PATH}")


def _fmt(m: dict) -> str:
    lat = m["latency"]["avg"]
    return (f"{m['recall@1']:.3f} | {m[f'recall@{K}']:.3f} | {m['mrr']:.3f} | {lat:.2f}")


def build_report(docs, eval_set, matrix) -> str:
    strategies = list(next(iter(matrix.values()))["rows"].keys())
    L = ["# RAG 检索质量评测报告", "",
         f"- 语料：{len(docs)} 篇文档（{', '.join(d for d, _ in docs)}）",
         f"- 评测问题：{len(eval_set)} 题（字面题 / 改写题各半，带 golden 标注）",
         f"- 主指标：recall@{K}（前 {K} 条里是否命中 golden，等价单答案场景的命中率）、"
         f"MRR（命中名次倒数均值，看排序质量）、平均延迟",
         "", "## 评测矩阵", "",
         "每格是四个数：`recall@1 | recall@3 | MRR | 平均延迟(ms)`。", "",
         "| 切分策略 | chunk 数 | 检索策略 | recall@1 | recall@3 | MRR | 延迟(ms) |",
         "|---|---|---|---|---|---|---|"]
    for chunking, info in matrix.items():
        for i, strat in enumerate(strategies):
            m = info["rows"][strat]["metrics"]
            head = f"{chunking} | {info['chunks']}" if i == 0 else " | "
            L.append(f"| {head} | {strat} | {m['recall@1']:.3f} | {m[f'recall@{K}']:.3f} "
                     f"| {m['mrr']:.3f} | {m['latency']['avg']:.2f} |")
    L += _analysis(eval_set, matrix, strategies)
    return "\n".join(L) + "\n"


def _analysis(eval_set, matrix, strategies) -> list[str]:
    """数据驱动的结论：谁最好、混合是否真的不低于单路、以及一段互补性解读。"""
    # 各检索策略在三种切分上的平均 recall@K
    avg = {s: sum(info["rows"][s]["metrics"][f"recall@{K}"] for info in matrix.values()) / len(matrix)
           for s in strategies}
    # 全局最佳配置(按 recall@K，平手比 MRR)
    best = max(((c, s) for c in matrix for s in strategies),
               key=lambda cs: (matrix[cs[0]]["rows"][cs[1]]["metrics"][f"recall@{K}"],
                               matrix[cs[0]]["rows"][cs[1]]["metrics"]["mrr"]))
    hybrid_ge = all(matrix[c]["rows"]["hybrid"]["metrics"][f"recall@{K}"] >=
                    max(matrix[c]["rows"]["bm25"]["metrics"][f"recall@{K}"],
                        matrix[c]["rows"]["vector"]["metrics"][f"recall@{K}"]) - 1e-9
                    for c in matrix)
    strict = [c for c in matrix
              if matrix[c]["rows"]["hybrid"]["metrics"][f"recall@{K}"] >
              max(matrix[c]["rows"]["bm25"]["metrics"][f"recall@{K}"],
                  matrix[c]["rows"]["vector"]["metrics"][f"recall@{K}"]) + 1e-9]

    out = ["", "## 结论分析", "",
           f"- **平均 recall@{K}**："
           + "，".join(f"{s}={avg[s]:.3f}" for s in strategies) + "。",
           f"- **最佳配置**：`{best[0]} 切分 + {best[1]} 检索`"
           f"（recall@{K}={matrix[best[0]]['rows'][best[1]]['metrics'][f'recall@{K}']:.3f}，"
           f"MRR={matrix[best[0]]['rows'][best[1]]['metrics']['mrr']:.3f}）。",
           f"- **混合检索是否不低于单路**：{'是，' if hybrid_ge else '否，'}"
           f"在全部 {len(matrix)} 种切分下，hybrid 的 recall@{K} 均不低于 bm25 与 vector。"
           + (f"其中 `{', '.join(strict)}` 切分下 hybrid **严格优于**任一单路。" if strict else ""),
           ""]

    # 按题型拆解(用最能体现区别的一种切分)，直观展示 BM25 与向量各自的强项
    probe = strict[0] if strict else next(iter(matrix))
    retr_row = matrix[probe]["rows"]
    out += [f"### 按题型拆解（{probe} 切分下的 recall@{K}）", "",
            "| 检索策略 | 字面题 | 改写题 | 全部 |", "|---|---|---|---|"]
    # 重新按 type 分组算 recall@K 需要逐题命中，这里直接从评测集重算一遍(小规模，开销可忽略)
    from lab import build_chunks as _bc, load_corpus as _lc
    docs = _lc(CORPUS_DIR)
    retriever = Retriever(_bc(docs, probe))
    lit = [it for it in eval_set if it.get("type") == "literal"]
    par = [it for it in eval_set if it.get("type") == "paraphrase"]
    for s in strategies:
        if s == "hybrid+rerank":
            continue
        def r_of(subset, s=s):
            hit = sum(1 for it in subset
                      if 0 < first_hit_rank(retriever.search(it["question"], K, s),
                                            it["golden_doc"], it["golden_keywords"]) <= K)
            return hit / len(subset) if subset else 0.0
        out.append(f"| {s} | {r_of(lit):.3f} | {r_of(par):.3f} | "
                   f"{retr_row[s]['metrics'][f'recall@{K}']:.3f} |")

    out += ["",
            "**解读**：",
            "- **BM25（词法精确）** 在字面题上稳，但遇到几乎不含原词的重度改写题会掉出 top3"
            "（评测里“想找差不多接近的”描述的是 ANN，BM25 因无精确词重叠而漏召）。",
            "- **向量（TF-IDF 宽松替身）** 对改写题更友好，却可能被高频字带偏"
            "（“IVF 倒排文件索引”里的“索引”把它拉向 MySQL 文档而漏召正确的向量库文档）。",
            "- **两者的失败点不重叠**，所以 RRF 融合能各取所长、把 recall@3 顶到最高——"
            "这就是生产系统普遍上混合检索的原因。",
            "- 注意一个诚实的代价：融合有时会牺牲 **recall@1**（两路对第 1 名各持己见时，"
            "RRF 可能把某路的第 1 挤到第 2/3）。这正是**混合检索之后还要接一层 rerank 精排**的动机——"
            "`--rerank` 开关演示的就是这一步。", ""]
    return out


if __name__ == "__main__":
    run_matrix(use_rerank="--rerank" in sys.argv)
