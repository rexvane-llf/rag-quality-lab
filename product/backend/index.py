"""index.py — 索引层：中文友好分词 + BM25 + TF-IDF 向量(纯 Python，零依赖)。

重要的诚实声明(面试被追问时照实讲)：
  这里的 TF-IDF 向量是**语义 embedding 的教学替身**。它和真 embedding 的本质区别：
  TF-IDF 只会"词的加权重叠"——查询和文档必须出现相同的词才算相关；
  真 embedding(bge / OpenAI text-embedding)理解语义——"番茄"和"西红柿"没有共同字
  也能算相近。所以生产环境用 bge/OpenAI embedding + 向量库(Chroma/pgvector/Milvus)，
  本项目用纯 Python 是为了"无 API Key 也能完整跑评测"这一教学目标。
  好在 BM25 与 TF-IDF 都是经得起考的传统 IR 基石，把它们讲透本身就是加分项。
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .chunking import Chunk

# 分词正则：一段连续的英文/数字，或一段连续的中文(CJK 基本区 U+4E00–U+9FFF)
_TOKEN_UNIT = re.compile(r"[a-z0-9]+|[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """中文友好分词(教学版)：
      - 英文/数字：按词整体切，小写归一(Cache-Control -> cache, control)
      - 中文：按 bigram(相邻两字)切，'消息队列' -> 消息/息队/队列

    为什么中文用 bigram 而不是单字？单字('的''是''据')区分度太低；
    真正的中文分词要靠词典(jieba)或模型，但 bigram 无需任何词典，就能让
    '缓存''索引''队列'这类二字术语在检索时精确对齐，是纯 Python 场景的经典折中。
    (真 embedding 不需要分词这一步，这也是它和传统 IR 的一个区别)
    """
    tokens: list[str] = []
    for m in _TOKEN_UNIT.finditer(text.lower()):
        s = m.group()
        if s[0].isascii():
            tokens.append(s)                       # 英文/数字词整体作为一个 token
        elif len(s) == 1:
            tokens.append(s)                       # 孤立单字(如"锁")
        else:
            tokens.extend(s[i:i + 2] for i in range(len(s) - 1))  # 中文串做 bigram
    return tokens


def tokenize_loose(text: str) -> list[str]:
    """更"宽松"的分词：中文同时产出**字级 unigram + 词级 bigram**。给 TF-IDF 向量route用。

    为什么两路要用不同粒度的分词(本项目的一个关键设计，面试可当亮点讲)：
      真实的混合检索是"稀疏词法检索(BM25) + 稠密语义向量(embedding)"两种**不同表示**的互补——
      BM25 精确但死板，向量宽松但能跨表述召回，二者错误互补，融合才有意义。
      但本项目为了离线零依赖，向量route只能用 TF-IDF 词法替身；若它和 BM25 用**完全相同**的
      bigram 分词，两路排序会几乎一致，RRF 融合就退化成"和自己融合"，毫无增益。
      于是让向量route多产出字级 unigram：它能靠更细的字符重叠，召回一些改写得面目全非、
      连 bigram 都对不上的口语化问题(实测见 tests)。
      必须诚实承认：这仍是**词法**匹配、不是语义——它靠的是共享字符，而非理解含义；
      真 embedding 能匹配"零字符重叠的同义句"，那才是它不可替代的地方。这里只是用最小的代价，
      在纯离线环境里复现出"两路互补、RRF 取长补短"这一现象，好让评测能测出融合的价值。
    """
    tokens: list[str] = []
    for m in _TOKEN_UNIT.finditer(text.lower()):
        s = m.group()
        if s[0].isascii():
            tokens.append(s)
        else:
            tokens.extend(s)                                     # 字级 unigram(宽松召回)
            tokens.extend(s[i:i + 2] for i in range(len(s) - 1))  # 词级 bigram(保留精度)
    return tokens


class BM25Index:
    """BM25 —— 传统检索的黄金基线(面试必考：BM25 vs 向量、BM25 打分怎么来的)。

    打分直觉，三个力的乘积/加权：
      TF  词在该文档出现越多越相关，但**边际递减**(出现 10 次不该是 1 次的 10 倍) —— k1 控制饱和;
      IDF 词在整个语料越稀有越值钱('缓存'比'的'有区分度) —— log 平滑;
      长度归一 长文档天然词多，要打折，别让长文档白占便宜 —— b 控制归一强度。

    完整公式(对查询里每个词 t 求和)：
      score(q,d) = Σ_t  IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
      IDF(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
    其中 f(t,d)=词频, |d|=文档长度, avgdl=平均长度, N=文档数, df=含该词的文档数。
    经验默认 k1=1.5, b=0.75。
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0
        self.tf = [Counter(d) for d in self.docs]           # 每篇的词频表
        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())                        # 每个词出现在多少篇里
        self.N = len(self.docs)
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored: list[tuple[int, float]] = []
        for i, counts in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = counts.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], sc) for i, sc in scored[:k]]


class TfidfIndex:
    """TF-IDF 向量 + 余弦相似度 —— 语义 embedding 的教学替身(见模块顶部声明)。

    把每个 chunk 表示成稀疏向量(维度=词表，某维的值 = 该词的 tf·idf)，查询也向量化，
    再算查询向量与各文档向量夹角的**余弦相似度**(只看方向、不看长度，范围 0~1)。
    它捕捉的是"加权词重叠"，捕捉不了没有共同词的语义相关——那正是真 embedding 的用武之地。

    注意：本类用 tokenize_loose(字级 unigram + 词级 bigram)，比 BM25 的纯 bigram 更宽松，
    好和 BM25 形成互补的两路(为什么这么设计见 tokenize_loose 的注释)。
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs = [tokenize_loose(c.text) for c in chunks]
        self.N = len(docs) or 1
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        # 平滑 idf：log((1+N)/(1+df)) + 1，避免除零、并保证 idf 恒为正
        self.idf = {t: math.log((1 + self.N) / (1 + n)) + 1 for t, n in df.items()}
        self.vecs = [self._vectorize(d) for d in docs]
        self.norms = [math.sqrt(sum(v * v for v in vec.values())) or 1e-9 for vec in self.vecs]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        """词序列 -> {词: tf·idf}。tf 用"词频/文档长度"做归一，避免长文档分量整体偏大。"""
        if not tokens:
            return {}
        counts = Counter(tokens)
        n = len(tokens)
        return {t: (c / n) * self.idf.get(t, 0.0) for t, c in counts.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        qv = self._vectorize(tokenize_loose(query))
        qnorm = math.sqrt(sum(v * v for v in qv.values())) or 1e-9
        scored: list[tuple[int, float]] = []
        for i, vec in enumerate(self.vecs):
            # 点积只需遍历较短的那个向量的词(稀疏向量的常规优化)
            small, big = (qv, vec) if len(qv) <= len(vec) else (vec, qv)
            dot = sum(val * big[t] for t, val in small.items() if t in big)
            if dot <= 0:
                continue
            scored.append((i, dot / (qnorm * self.norms[i])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], sc) for i, sc in scored[:k]]
