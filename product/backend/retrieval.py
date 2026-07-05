"""retrieval.py — 检索策略：bm25 / vector / hybrid(RRF 融合)。

RAG 里"检索"这一步决定了生成质量的上限：喂给大模型的资料本身不相关，
再强的模型也只能基于错的资料编答案(garbage in, garbage out)。所以本项目把检索
单独拎出来量化。三种策略对应面试里"你们检索用什么、为什么"这个必答题：
  bm25    纯关键词/词法检索，对精确术语、编号、专名最稳，是不可替代的基线。
  vector  语义检索(本项目用 TF-IDF 替身)，对同义改写、口语化提问更友好。
  hybrid  两路都跑再融合，取长补短——这是当下生产系统的主流做法。
"""
from __future__ import annotations

from .chunking import Chunk
from .index import BM25Index, TfidfIndex

# RRF 的平滑常数。60 是原论文(Cormack 2009)的经验值，被 Elasticsearch 等直接沿用。
# 它的作用：削弱头部名次的绝对优势，让第 1 名和第 2 名的贡献差距不至于悬殊。
RRF_K = 60


class Retriever:
    """把两种索引封装成三种检索策略，对外统一 search(query, k, strategy) 接口。
    一次构建、多策略复用——run_matrix.py 靠它对同一批 chunk 跑三种检索做对比。"""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.vector = TfidfIndex(chunks)

    def search(self, query: str, k: int = 5, strategy: str = "hybrid",
               pool: int = 20) -> list[tuple[Chunk, float]]:
        if strategy == "bm25":
            return self.bm25.search(query, k)
        if strategy == "vector":
            return self.vector.search(query, k)
        if strategy == "hybrid":
            return self._rrf(query, k, pool)
        raise ValueError(f"未知检索策略: {strategy}(可选 bm25/vector/hybrid)")

    def _rrf(self, query: str, k: int, pool: int) -> list[tuple[Chunk, float]]:
        """RRF(Reciprocal Rank Fusion，倒数排名融合)融合 BM25 与向量两路。

        score(d) = Σ_每一路  1 / (RRF_K + rank_该路(d))   (rank 从 1 起算)

        为什么 RRF 用"名次"而不是"原始分数"融合(这是面试高频追问)：
          BM25 的分可能是几十上百，余弦相似度却在 0~1，两者量纲天差地别。
          直接加权求和 = 要人肉调一个权重系数、还随语料分布漂移，非常脆。
          RRF 干脆丢掉分数、只用名次：不管哪一路，排第 1 就贡献 1/(K+1)。
          于是天然免调参、对离群分数鲁棒——某一路给出一个虚高分也翻不了盘。
        一个文档只要在任一路里排得靠前就能得分，被两路都认可则得分叠加冲到最前，
        这正是"取长补短"在数学上的体现。
        """
        fused: dict[str, list] = {}   # chunk_id -> [chunk, 累计RRF分]
        for ranked in (self.bm25.search(query, pool), self.vector.search(query, pool)):
            for rank, (chunk, _score) in enumerate(ranked, start=1):
                slot = fused.setdefault(chunk.chunk_id, [chunk, 0.0])
                slot[1] += 1.0 / (RRF_K + rank)
        ordered = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        return [(chunk, score) for chunk, score in ordered[:k]]
