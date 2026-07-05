"""test_lab.py — 纯检索层单测(无需 API Key，不发任何网络请求)。

覆盖：三种切分的正确性、中文分词、BM25 对字面题的召回、向量对改写题的召回、
RRF 融合在"互补子集"上严格优于单路、以及 recall@k / MRR 指标计算的正确性。
这些是本项目最该被测的"骨头"——评测结论的可信度全靠它们不出错。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab import build_chunks, load_corpus, load_eval
from lab.chunking import chunk_fixed, chunk_sentence, chunk_structure
from lab.index import tokenize, BM25Index, TfidfIndex
from lab.retrieval import Retriever
from lab.evaluate import (QueryResult, first_hit_rank, recall_at_k, mrr, is_relevant)


# ----------------------------------------------------------------------------- 切分
def test_chunk_fixed_has_overlap():
    """固定切分：相邻两个 chunk 应有 overlap 个字符的重叠(补边界漏召的关键设计)。"""
    text = "0123456789abcdefghij"          # 20 字符，无空白避免 strip 干扰
    chunks = chunk_fixed("d", text, size=10, overlap=4)
    assert len(chunks) == 3                  # step=6：[0:10] [6:16] [12:20]
    assert chunks[0].text[-4:] == chunks[1].text[:4]     # 尾 4 字符 == 下一块头 4 字符
    assert all(c.doc_id == "d" and c.chunk_id for c in chunks)


def test_chunk_sentence_keeps_whole_sentences():
    """句子切分：每个 chunk 由完整句子组成，不会把一句话拦腰切断。"""
    text = "第一句。第二句！第三句？"
    chunks = chunk_sentence("d", text, max_chars=4)      # 每句 4 字符，恰好各成一块
    assert [c.text for c in chunks] == ["第一句。", "第二句！", "第三句？"]
    assert all(c.text[-1] in "。！？" for c in chunks)    # 每块都以句末标点收尾


def test_chunk_structure_tracks_heading():
    """结构切分：以标题为边界，每个 chunk 记录自己所属的小节标题。"""
    text = "# 标题A\n内容一。\n## 标题B\n内容二。"
    chunks = chunk_structure("d", text)
    by_head = {c.heading: c.text for c in chunks}
    assert "标题A" in by_head and "标题B" in by_head
    assert "内容一" in by_head["标题A"]
    assert "内容二" in by_head["标题B"]                   # 内容二被正确归到标题B 名下


# ----------------------------------------------------------------------------- 分词
def test_tokenize_chinese_bigram_and_english_word():
    """中文按 bigram 切、英文/数字按词切并小写归一。"""
    assert tokenize("消息队列") == ["消息", "息队", "队列"]
    assert tokenize("BM25 检索") == ["bm25", "检索"]


# ----------------------------------------------------------------------------- 检索
def _corpus_chunks(strategy="structure"):
    return build_chunks(load_corpus(str(ROOT / "data" / "corpus")), strategy)


def test_bm25_recalls_literal_query():
    """BM25 对字面题(含精确术语)应把 golden 文档召回到 top3。"""
    bm25 = BM25Index(_corpus_chunks())
    hits = bm25.search("什么是覆盖索引和回表", k=3)
    assert first_hit_rank(hits, "mysql-index", ["覆盖索引", "回表"]) > 0


def test_vector_recalls_paraphrase_where_bm25_fails():
    """向量(宽松 TF-IDF)能召回重度改写题，而同一题 BM25 会掉出 top3——两路互补的实证。"""
    chunks = _corpus_chunks()
    q = "想找差不多接近的、不是一模一样的，还得够快，用什么办法？"   # 口语描述 ANN
    gold_doc, gold_kw = "vector-search", ["ann", "最近邻"]
    vec_hit = first_hit_rank(TfidfIndex(chunks).search(q, k=3), gold_doc, gold_kw)
    bm_hit = first_hit_rank(BM25Index(chunks).search(q, k=3), gold_doc, gold_kw)
    assert vec_hit > 0                       # 向量命中
    assert bm_hit == 0                       # BM25 在 top3 内漏召(它擅长精确词，不擅长改写)


def test_rrf_hybrid_beats_each_single_route():
    """RRF 融合在'互补子集'上严格优于任一单路(评测集子集断言)。

    子集刻意选两道单路各有盲区的题：
      - 'IVF 倒排文件索引'：向量被高频字'索引'带偏而漏召，BM25 命中；
      - '想找差不多接近的'：BM25 因无精确词重叠而漏召，向量命中。
    单路各只能答对一半，RRF 因为把两路排名相加、能各取所长，两题全中。
    """
    r = Retriever(_corpus_chunks("structure"))
    subset = [
        {"question": "IVF 倒排文件索引是怎么工作的？",
         "golden_doc": "vector-search", "golden_keywords": ["倒排", "ivf"]},
        {"question": "想找差不多接近的、不是一模一样的，还得够快，用什么办法？",
         "golden_doc": "vector-search", "golden_keywords": ["ann", "最近邻"]},
    ]

    def recall3(strategy):
        hits = sum(1 for it in subset
                   if 0 < first_hit_rank(r.search(it["question"], 3, strategy),
                                         it["golden_doc"], it["golden_keywords"]) <= 3)
        return hits / len(subset)

    rb, rv, rh = recall3("bm25"), recall3("vector"), recall3("hybrid")
    assert rb == 0.5 and rv == 0.5           # 单路各答对一半
    assert rh == 1.0                         # 融合两题全中
    assert rh > rb and rh > rv               # 严格优于两条单路


def test_hybrid_not_below_single_on_full_set():
    """全评测集上，hybrid 的 recall@3 不低于任一单路(项目的硬验收)。"""
    eval_set = load_eval(str(ROOT / "data" / "eval_set.json"))
    r = Retriever(_corpus_chunks("structure"))

    def recall3(strategy):
        hits = sum(1 for it in eval_set
                   if 0 < first_hit_rank(r.search(it["question"], 3, strategy),
                                         it["golden_doc"], it["golden_keywords"]) <= 3)
        return hits / len(eval_set)

    assert recall3("hybrid") >= recall3("bm25")
    assert recall3("hybrid") >= recall3("vector")


# ----------------------------------------------------------------------------- 指标
def test_recall_and_mrr_computation():
    """recall@k 与 MRR 的算法正确性(用构造好的命中名次断言，脱离检索单独验证)。"""
    results = [QueryResult("a", 1, 0.0), QueryResult("b", 3, 0.0),
               QueryResult("c", 0, 0.0), QueryResult("d", 5, 0.0)]   # 命中名次 1/3/未命中/5
    assert recall_at_k(results, 1) == 0.25                # 只有名次 1 落在 top1
    assert recall_at_k(results, 3) == 0.5                 # 名次 1、3 落在 top3
    assert recall_at_k(results, 5) == 0.75                # 名次 1、3、5 落在 top5
    expected_mrr = (1 / 1 + 1 / 3 + 0 + 1 / 5) / 4
    assert abs(mrr(results) - expected_mrr) < 1e-9


def test_is_relevant_needs_doc_and_keyword():
    """命中判定：文档 id 不对、或关键词一个都不含，都不算命中。"""
    from lab.chunking import Chunk
    c = Chunk("mysql-index", "mysql-index#0", "覆盖索引避免回表", "回表与覆盖索引")
    assert is_relevant(c, "mysql-index", ["回表"]) is True
    assert is_relevant(c, "http-caching", ["回表"]) is False       # 文档不匹配
    assert is_relevant(c, "mysql-index", ["削峰"]) is False        # 关键词不出现
