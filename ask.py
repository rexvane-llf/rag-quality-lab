"""ask.py — 交互式问答 CLI：把检索层真正"用起来"。

无 API Key：混合检索出 top3，列出处(文档 / 小节 / 片段)——一个纯本地、零依赖的语义搜索。
有 API Key：查询改写 → 混合检索 → (可选 LLM 重排) → 生成带 [编号] 引用的答案，答案每句可溯源。
这条链路就是一个最小但完整的 RAG，麻雀虽小五脏俱全。

用法：
  python ask.py                 # 交互模式
  python ask.py "什么是覆盖索引"  # 单次提问
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from lab import load_corpus, build_chunks
from lab.retrieval import Retriever
from lab.llm_extras import get_llm, query_rewrite, llm_rerank, answer_with_citations

CORPUS_DIR = "data/corpus"
TOP_K = 3


def _print_sources(retrieved) -> None:
    """列出检索到的出处，编号与答案里的 [编号] 引用一一对应。"""
    print("\n参考出处：")
    for i, (chunk, score) in enumerate(retrieved, start=1):
        loc = f"{chunk.doc_id}" + (f" › {chunk.heading}" if chunk.heading else "")
        snippet = chunk.text.replace("\n", " ")[:70]
        print(f"  [{i}] {loc}（score={score:.3f}）\n      {snippet}…")


def answer(retriever: Retriever, llm, question: str) -> None:
    if llm is None:
        # —— 无 Key 的优雅退化路径：纯检索，只列出处 ——
        retrieved = retriever.search(question, k=TOP_K, strategy="hybrid")
        if not retrieved:
            print("（没有检索到相关内容）")
            return
        print("（未配置 API Key，仅做混合检索并列出出处；配置后可生成带引用的答案）")
        _print_sources(retrieved)
        return

    # —— 有 Key 的完整 RAG 路径 ——
    rewritten = query_rewrite(question, llm) or question
    if rewritten != question:
        print(f"（查询改写：{rewritten}）")
    pool = retriever.search(rewritten, k=20, strategy="hybrid")   # 先宽召回一批
    reranked = llm_rerank(question, pool, top_k=TOP_K, llm=llm)   # 再精排取前 K
    top = reranked if reranked is not None else pool[:TOP_K]
    reply = answer_with_citations(question, top, llm)
    print("\n" + (reply or "（生成失败，降级为列出处）"))
    _print_sources(top)


def main() -> None:
    docs = load_corpus(CORPUS_DIR)
    retriever = Retriever(build_chunks(docs, "structure"))        # 默认用结构切分 + 混合检索
    llm = get_llm()
    mode = "完整 RAG（改写→检索→重排→带引用生成）" if llm else "纯检索（无 Key，列出处）"
    print(f"rag-quality-lab 问答  |  模式：{mode}  |  语料 {len(docs)} 篇")

    if len(sys.argv) > 1:
        answer(retriever, llm, " ".join(sys.argv[1:]))
        return
    print("输入问题开始（exit 退出）")
    while True:
        try:
            q = input("\n问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q == "exit":
            break
        answer(retriever, llm, q)


if __name__ == "__main__":
    main()
