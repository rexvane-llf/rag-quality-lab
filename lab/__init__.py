"""rag-quality-lab — 能量化测量自身检索质量的教学级 RAG 评测实验台。

四个核心模块，对应 RAG 的检索侧全链路：
  chunking.py   切分：把长文档切成可检索的 chunk(fixed / sentence / structure)
  index.py      索引：中文友好分词 + BM25 + TF-IDF 向量(纯 Python 零依赖)
  retrieval.py  检索：bm25 / vector / hybrid(RRF 融合)
  evaluate.py   评测：recall@k / MRR / 命中判定 / 延迟
llm_extras.py 是可选增强层(查询改写 / LLM 重排 / 带引用生成)，无 API Key 时优雅退化。

下面几个 load_/build_ 小工具是给 run_matrix.py、ask.py、tests 复用的数据装配逻辑，
放在包入口避免三处重复。"""
from __future__ import annotations

import glob
import json
import os

from .chunking import Chunk, chunk_document, CHUNKERS
from .index import tokenize, BM25Index, TfidfIndex
from .retrieval import Retriever
from . import evaluate


def load_corpus(corpus_dir: str) -> list[tuple[str, str]]:
    """读取语料目录下所有 .md，返回 [(doc_id, 全文)]。doc_id 取文件名(不含扩展名)。"""
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            docs.append((doc_id, f.read()))
    return docs


def load_eval(path: str) -> list[dict]:
    """读取评测集 JSON，返回问题列表。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def build_chunks(docs: list[tuple[str, str]], strategy: str) -> list[Chunk]:
    """用指定切分策略把整批文档切成一个扁平的 chunk 列表(供建索引)。"""
    chunks: list[Chunk] = []
    for doc_id, text in docs:
        chunks.extend(chunk_document(doc_id, text, strategy))
    return chunks
