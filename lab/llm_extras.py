"""llm_extras.py — 可选的 LLM 增强层：查询改写 / LLM 重排 / 带引用生成。

设计原则：**能力可选、退化优雅**。检索层(chunking/index/retrieval/evaluate)是纯 Python、
零依赖、无 Key 也能完整跑评测；本模块是"锦上添花"——配了 API Key 才启用，没配就每个函数
返回 None，调用方(ask.py)据此回退到"只列检索出处"。这样项目对没有 Key 的同学依然 100% 可用。

双通道 Provider 沿用 agent-kernel 的思路(业务不感知厂商，切模型只改环境变量)：
  - anthropic：Claude，用 claude-opus-4-8 + 自适应思考(thinking adaptive)
  - openai_compat：DeepSeek/Qwen/GLM/Kimi 等国产模型走 OpenAI 兼容协议

三个函数分别对应三道 RAG 面试题：
  query_rewrite       —— 查询改写：口语问题 → 检索友好查询(为什么要改写)
  llm_rerank          —— 召回后重排：粗筛之后为什么还要精排
  answer_with_citations —— 带 [编号] 引用的生成：RAG 怎么溯源、怎么治幻觉
"""
from __future__ import annotations

import os
import re


class _AnthropicLLM:
    """Anthropic Claude 通道。用 claude-opus-4-8，开自适应思考让模型自己决定想多深。"""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()                       # 读 ANTHROPIC_API_KEY
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class _OpenAICompatLLM:
    """OpenAI 兼容通道：DeepSeek/Qwen/GLM/Kimi 等国产模型。"""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = os.getenv("OPENAI_MODEL", "deepseek-chat")

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm():
    """按环境变量选一个可用的 LLM，选不出(没配 Key)就返回 None —— 这是优雅退化的开关。

    显式 LLM_PROVIDER 优先；未显式指定时，谁有 Key 用谁(方便本地随手配一个就能用)。
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return _AnthropicLLM()
    if provider in ("openai_compat", "openai", "deepseek") and os.getenv("OPENAI_API_KEY"):
        return _OpenAICompatLLM()
    if provider in ("auto", ""):
        if os.getenv("ANTHROPIC_API_KEY"):
            return _AnthropicLLM()
        if os.getenv("OPENAI_API_KEY"):
            return _OpenAICompatLLM()
    return None


def query_rewrite(query: str, llm=None) -> str | None:
    """查询改写：把口语化问题改写成检索友好的查询(补同义词/专业术语)。

    面试点(为什么要查询改写 / query rewriting)：用户问"网页第二次打开为什么快"，
    字面里根本没有"缓存"这个关键词，纯词法检索抓瞎；先让 LLM 把它改写成
    "HTTP 缓存 强缓存 协商缓存 304 复用本地副本"，召回率立刻上一个台阶。
    这是弥补"用户表述"和"文档术语"词汇鸿沟的最轻量手段。无 LLM 时返回 None(用原查询)。
    """
    llm = llm or get_llm()
    if llm is None:
        return None
    system = ("你是检索查询改写器。把用户的口语化问题改写成用于关键词检索的查询："
              "补充同义词与专业术语，保留原意。只输出改写后的查询词，不要解释、不要标点堆砌。")
    try:
        out = llm.generate(system, f"用户问题：{query}\n改写为检索查询：")
        return out or None
    except Exception:
        return None                                              # 网络/额度等异常一律退化，不让增强层拖垮主流程


def llm_rerank(query: str, candidates: list, top_k: int = 5, llm=None) -> list | None:
    """LLM 重排：对召回的候选(通常 top~20)让 LLM 按与问题的相关性重排，取 top_k。

    面试点(召回之后为什么还要 rerank)：
      召回(BM25/向量)是"粗筛"，追求快和高召回，排序不够精，还常被高频词带偏(见评测里的 IVF 例子)；
      rerank 是"精排"，让强模型逐条判断语义相关性，把最该在前的顶上来。
      工业界多用 cross-encoder(如 bge-reranker)；这里用通用 LLM 打分演示同一思想。
      这也正好补上混合检索的短板：RRF 融合提升了 recall@k(覆盖)，却可能牺牲 top1 精度，
      rerank 恰好把 top1 的精度再拉回来 —— "融合扩召回、重排提精度"是标准组合拳。

    candidates: list[(chunk, score)]。无 LLM 时返回 None(调用方沿用原顺序)。
    """
    llm = llm or get_llm()
    if llm is None or not candidates:
        return None
    listing = "\n".join(f"[{i}] {c.text[:180]}" for i, (c, _) in enumerate(candidates))
    system = ("你是相关性排序器。根据【问题】从候选段落里挑出最相关的若干条，"
              "按相关性从高到低输出它们的编号，用英文逗号分隔，如 3,0,5。只输出编号。")
    try:
        out = llm.generate(system, f"【问题】{query}\n【候选】\n{listing}\n最相关的 {top_k} 个编号：")
        order = [int(x) for x in re.findall(r"\d+", out)]
        picked, seen = [], set()
        for idx in order:                                        # 先按模型给的顺序取
            if 0 <= idx < len(candidates) and idx not in seen:
                picked.append(candidates[idx])
                seen.add(idx)
            if len(picked) >= top_k:
                break
        for i, cand in enumerate(candidates):                    # 模型漏选的按原顺序补齐，保证不丢候选
            if len(picked) >= top_k:
                break
            if i not in seen:
                picked.append(cand)
        return picked
    except Exception:
        return None


def answer_with_citations(query: str, retrieved: list, llm=None) -> str | None:
    """带 [编号] 引用的答案生成 —— RAG 里的 "G"(Generation)。

    面试点(RAG 怎么治幻觉 + 怎么做溯源)：
      把 top-k chunk 编号后塞进 prompt，要求模型**只依据给定资料作答**、并在每个论断后标 [编号]；
      资料没提到的就答"资料未提及"，不许编。溯源 = 答案里每句话都能追回到某个 chunk，用户可核对原文。
      这就是 RAG 相比"纯大模型直接答"最大的工程价值：可控、可查、可更新知识而不用重训模型。

    无 LLM 时返回 None(调用方回退到直接列出处)。
    """
    llm = llm or get_llm()
    if llm is None or not retrieved:
        return None
    context = "\n\n".join(f"[{i + 1}] {c.text}" for i, (c, _) in enumerate(retrieved))
    system = ("你是严谨的知识助手。只依据【资料】回答【问题】，在每个关键论断后用 [编号] 标注来源；"
              "资料没提到的绝不编造，应回答'资料未提及'。")
    user = f"【资料】\n{context}\n\n【问题】{query}\n\n请给出带 [编号] 引用的回答："
    try:
        return llm.generate(system, user, max_tokens=1200) or None
    except Exception:
        return None
