"""chunking.py — 切分策略：把长文档切成可检索的 chunk。

为什么切分是 RAG 的第一个质量分水岭(面试高频：RAG 里 chunk 怎么切)？
  - 切太大：一个 chunk 里混了多个主题，检索命中率下降，还浪费上下文、稀释相关信号。
  - 切太小：一句话被切碎，答案跨 chunk，单个 chunk 语义不完整，召回到了也没用。
  - 切错边界：把"问题"和"答案"劈到两个 chunk，等于人为制造了检索失败。
所以切分不是"按长度暴力截断"这么简单，而要尽量对齐语义/结构边界。本模块提供三档策略，
run_matrix.py 会把它们各跑一遍，用数据告诉你哪种更适合你的语料——这正是本项目的价值。

每个 chunk 记录 (doc_id, chunk_id, text, heading)：heading 是它所属的小节标题，
既能在生成答案时做更精确的引用溯源，也让"结构感知切分"的优势可被观测。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str        # 来源文档 id
    chunk_id: str      # 全局唯一 id(doc_id#策略序号)，评测/融合时用它去重
    text: str          # chunk 正文
    heading: str = ""  # 所属 markdown 小节标题(结构切分能拿到，固定切分只能事后回查)


_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$")
_HEADING_ANY = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
# 句子边界：中英文句末标点 + 分号 + 换行。切句时保留标点(用 lookbehind 零宽切分)
_SENT_BOUNDARY = re.compile(r"(?<=[。！？；!?;\n])")


def _segments(text: str) -> list[tuple[str, str]]:
    """把文档按 markdown 标题切成 [(heading, 该标题下的正文块)]。
    标题文字本身也并入正文，好让检索能命中标题里的关键词。"""
    segs: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in text.splitlines():
        m = _HEADING_LINE.match(line)
        if m:
            if buf:
                segs.append((heading, "\n".join(buf)))
                buf = []
            heading = m.group(2).strip()
            buf.append(heading)          # 标题词并入正文
        else:
            buf.append(line)
    if buf:
        segs.append((heading, "\n".join(buf)))
    return segs


def _sentences(block: str) -> list[str]:
    """把一段文本切成句子(保留句末标点)，并丢掉纯空白。"""
    return [s.strip() for s in _SENT_BOUNDARY.split(block) if s.strip()]


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    """把句子按顺序打包成不超过 max_chars 的块(贪心)：
    宁可让一个块略小，也不把一句话拦腰切断——这是句子级切分优于固定切分的关键。"""
    out: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) > max_chars:
            out.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        out.append(buf)
    return out


def _heading_at(text: str, pos: int) -> str:
    """回查 text[pos] 之前最近的一个标题。固定窗口切分不认识结构，
    只能这样事后定位自己落在哪个小节下——也解释了它为何可能把一节从中间劈开。"""
    heading = ""
    for m in _HEADING_ANY.finditer(text):
        if m.start() > pos:
            break
        heading = m.group(2).strip()
    return heading


def chunk_fixed(doc_id: str, text: str, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """固定窗口切分：按字符数滑动切，相邻窗口重叠 overlap 个字符。

    最朴素的基线。重叠是它唯一的"补救"：万一答案正好落在切割线上，
    重叠区能让相邻两个 chunk 都各含一部分，降低边界处漏召回的概率。
    缺点很明显——它完全不认识句子和小节边界，会把语义单元切碎(面试可对比讲)。
    """
    text = text.strip()
    step = max(1, size - overlap)     # 步长=窗口-重叠；防 overlap>=size 时死循环
    chunks: list[Chunk] = []
    i = n = 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append(Chunk(doc_id, f"{doc_id}#fixed{n}", piece, _heading_at(text, i)))
            n += 1
        if i + size >= len(text):     # 本窗口已覆盖到结尾，停——否则会多出被上一块完全包含的冗余尾块
            break
        i += step
    return chunks


def chunk_sentence(doc_id: str, text: str, max_chars: int = 220) -> list[Chunk]:
    """句子级切分：先按句末标点切句，再贪心打包到 max_chars 以内，且不跨小节。
    保证每个 chunk 都由完整句子组成，语义比固定切分干净。"""
    chunks: list[Chunk] = []
    n = 0
    for heading, block in _segments(text):
        for piece in _pack(_sentences(block), max_chars):
            chunks.append(Chunk(doc_id, f"{doc_id}#sent{n}", piece.strip(), heading))
            n += 1
    return chunks


def chunk_structure(doc_id: str, text: str, max_chars: int = 400) -> list[Chunk]:
    """结构感知切分：以 markdown 标题为天然边界，一个小节尽量作为一个 chunk。
    小节过长(> max_chars)时，再在小节内部按句子二次切分，但绝不跨标题——
    每个 chunk 语义完整、且明确知道自己属于哪个小节(heading 精确)。
    对结构良好的技术文档，这通常是召回质量最好的策略。"""
    chunks: list[Chunk] = []
    n = 0
    for heading, block in _segments(text):
        block = block.strip()
        if not block:
            continue
        pieces = [block] if len(block) <= max_chars else _pack(_sentences(block), max_chars)
        for piece in pieces:
            chunks.append(Chunk(doc_id, f"{doc_id}#struct{n}", piece.strip(), heading))
            n += 1
    return chunks


# 策略注册表：run_matrix.py 遍历它跑切分矩阵，名字即报告里的列名
CHUNKERS = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "structure": chunk_structure,
}


def chunk_document(doc_id: str, text: str, strategy: str = "structure") -> list[Chunk]:
    """按名字选切分策略切一篇文档。"""
    if strategy not in CHUNKERS:
        raise ValueError(f"未知切分策略: {strategy}(可选 {'/'.join(CHUNKERS)})")
    return CHUNKERS[strategy](doc_id, text)
