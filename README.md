# rag-quality-lab — 能量化测量自身检索质量的教学级 RAG 实验台

> 这不是"又一个 Chat-with-PDF"。它的卖点是**用数据回答"我的检索到底准不准"**：
> 同一批文档、同一组带 golden 标注的问题，一键对比
> **切分策略(3) × 检索策略(3) × 可选 LLM 重排** 的 recall@3 / MRR / 延迟，产出 markdown 对比报告。
>
> 检索层**纯 Python、零第三方依赖**，无任何 API Key 也能完整跑评测；配了 Key(Anthropic 或
> DeepSeek 等国产模型)则解锁查询改写、LLM 重排、带 `[1][2]` 引用溯源的答案生成。
> 面试官问"你们 RAG 怎么保证召回质量"，你可以打开这张报告，而不是空谈。

## 为什么做这个

大多数 RAG demo 只演示"能答上来"，答不上来时你根本不知道是**切分切碎了**、**检索选错策略**、
还是**问题本身表述和文档对不上**。本项目把 RAG 的检索侧拆成四段可替换、可度量的模块，
让每一步的选择都有数字支撑——这正是工业界做 RAG 的真实工作方式，也是简历上"做过 RAG"和
"做过**能评测的** RAG"的区别。

## 架构

```
                          data/corpus/*.md            data/eval_set.json
                          四篇面试八股文档              12 题(golden 标注)
                                 │                            │
   ┌─────────────────────────────▼────────────────────────────▼──────────────┐
   │  lab/chunking.py   切分   fixed / sentence / structure                    │
   │  lab/index.py      索引   BM25(词级bigram)  +  TF-IDF向量(字级+词级, 更宽松) │
   │  lab/retrieval.py  检索   bm25 / vector / hybrid(RRF 融合)                 │
   │  lab/evaluate.py   评测   recall@k · MRR · 命中判定 · 延迟                  │
   └───────────────────────────────────┬──────────────────────────────────────┘
                                        │
        run_matrix.py ── 跑 3×3 矩阵 ──►  report.md(对比表 + 结论分析)
        ask.py ─────── 交互问答 ──────►  无Key:列出处 / 有Key:改写→检索→重排→带引用生成
                                        │
              lab/llm_extras.py(可选增强): query_rewrite · llm_rerank · answer_with_citations
```

## 快速开始

```bash
# 1) 零依赖跑评测(无需任何 API Key)
python run_matrix.py            # 生成 report.md：3 切分 × 3 检索 的 recall/MRR/延迟对比
python ask.py "什么是覆盖索引"    # 混合检索并列出处

# 2) (可选)解锁 LLM 增强
pip install -r requirements.txt
cp .env.example .env            # 填 Anthropic 或 DeepSeek 等国产模型的 Key(二选一)
python run_matrix.py --rerank   # 额外评测"混合检索 + LLM 重排"
python ask.py "网页第二次打开为什么快"   # 查询改写 → 检索 → 重排 → 带 [编号] 引用的答案

# 3) 单元测试(纯检索层，无需 Key)
PYTHONIOENCODING=utf-8 python -m pytest tests -q
```

## 评测矩阵怎么看

`run_matrix.py` 对每个"切分 × 检索"组合，用 12 道 golden 标注问题算三个指标：

| 指标 | 含义 | 面试口径 |
|------|------|----------|
| **recall@3** | 前 3 条结果里是否命中 golden | 本项目每题一个正确目标，等价"命中率 hit@3" |
| **MRR** | 命中名次倒数的均值 | 不只看召回与否，还看**排得靠不靠前** |
| **延迟** | 单次检索耗时 | 策略取舍的另一个坐标轴 |

**命中判定**：一个召回的 chunk 算命中，当且仅当它属于 golden 文档、且正文里出现了该题的
golden 关键词之一——既认"文档对了"、也认"具体段落对了"。

### 本仓库跑出来的关键结论（节选自 report.md）

平均 recall@3：**bm25 = 0.917，vector = 0.972，hybrid = 1.000**。按题型拆解（structure 切分）：

| 检索策略 | 字面题 recall@3 | 改写题 recall@3 |
|---|---|---|
| bm25（词法精确） | **1.000** | 0.833 |
| vector（TF-IDF 宽松） | 0.833 | **1.000** |
| **hybrid（RRF 融合）** | **1.000** | **1.000** |

一句话讲清价值：**BM25 擅长字面题、向量擅长改写题，两者的失败点不重叠，RRF 融合各取所长、
把 recall@3 顶到最高**。评测集里特意埋了两道"互补英雄题"制造这个对比：

- `IVF 倒排文件索引是怎么工作的？`——向量route被高频字"索引"带偏、错召到 MySQL 文档；BM25 靠精确术语命中。
- `想找差不多接近的、不是一模一样的，还得够快？`（在描述 ANN）——BM25 无精确词重叠而漏召；向量靠字符重叠命中。

## 面试怎么讲（每个模块对应的高频考题）

| 模块 / 概念 | 对应面试题 | 你能讲的点 |
|------|-----------|-----------|
| 整体 | RAG 的完整流程？ | 文档→切分→向量化/建索引→检索→(重排)→把资料塞进 prompt 生成。本项目每一步都是独立可换的模块 |
| `chunking.py` | chunk 怎么切？切分策略有哪些？ | 固定窗口(带重叠补边界)/句子边界/结构感知；切太大稀释信号、切太小语义不全、切错边界把问答劈两半 |
| `index.py` BM25 | BM25 打分怎么来的？BM25 vs 向量？ | TF 饱和(k1)+IDF 稀有度+长度归一(b)三个力；词法精确但不懂同义 |
| `index.py` 向量 | 为什么用向量检索/embedding？ | 把"文本相似"变成"向量距离"，能跨表述召回；本项目用 TF-IDF 替身(见"已知边界") |
| `retrieval.py` RRF | 混合检索怎么融合两路？为什么用 RRF？ | `score=Σ 1/(K+rank)`，只用**名次**不用分数：BM25 分和余弦分量纲天差地别，RRF 免调权重、对离群分数鲁棒 |
| `llm_extras.query_rewrite` | 为什么要查询改写？ | 用户口语("第二次打开为什么快")没有"缓存"这个词，先补同义词/术语再检索，弥补词汇鸿沟 |
| `llm_extras.llm_rerank` | 召回之后为什么还要 Re-rank？ | 召回是粗筛(快、高召回、排序糙)，rerank 是精排(强模型逐条判相关性)；正好补上 RRF 提召回却牺牲 top1 精度的短板 |
| `llm_extras.answer_with_citations` | RAG 怎么治幻觉 / 怎么溯源？ | 只依据给定资料作答、每句标 `[编号]`、没提到就答"资料未提及"；可控可查可更新，不用重训模型 |
| `evaluate.py` | **怎么确认你的召回是准的？** | 就是本项目在做的事：golden 标注 + recall@k/MRR，可复现地量化，而不是靠感觉 |
| `llm.py`(双通道) | 政府/合规项目不让用国外模型怎么办？ | Provider 抽象，业务不感知厂商，切 Anthropic↔DeepSeek 只改环境变量 |

## 项目结构

```
lab/
├── chunking.py     切分：fixed / sentence / structure，每个 chunk 记录 (doc_id, chunk_id, text, heading)
├── index.py        索引：中文友好分词(bigram) + BM25 + TF-IDF 向量(字级+词级, 更宽松)
├── retrieval.py    检索：bm25 / vector / hybrid(RRF, K=60)
├── evaluate.py     评测：recall@k / MRR / 命中判定 / 延迟统计
└── llm_extras.py   可选增强：query_rewrite / llm_rerank / answer_with_citations(无 Key 优雅退化)
data/
├── corpus/         四篇技术文档(HTTP缓存 / MySQL索引 / 消息队列 / 向量检索)——本身就是面试八股
└── eval_set.json   12 道 golden 标注问题(字面题 / 改写题各半)
run_matrix.py       跑评测矩阵，产出 report.md
ask.py              交互式问答 CLI
tests/test_lab.py   纯检索层单测(10 个，无需 Key)
```

## 已知边界（面试被追问时的诚实回答）

- **TF-IDF 是语义 embedding 的教学替身**。它匹配的是"词/字的重叠"，不是"语义"——
  "番茄"和"西红柿"没有共同字它就认不出。生产环境应换成 **bge / OpenAI text-embedding** 等
  真 embedding + **Chroma / pgvector / Milvus** 等向量库(ANN 索引)。本项目用纯 Python 是为了
  "无 Key 也能完整跑评测"这一教学目标。
- **两路用了不同粒度的分词**(BM25 用词级 bigram、向量route额外加字级 unigram)，是为了在
  没有真 embedding 的离线环境里，人为造出"词法精确 vs 语义宽松"的互补性，好让评测能测出
  RRF 融合的价值。这是**刻意的教学近似**：真 embedding 的互补性来自语义理解，远比字符重叠强大。
- **语料只有 4 篇**，指标偏高且波动大；真实评测需要几百上千题、更大更杂的语料，
  并区分 recall / precision / nDCG 等更细的指标。
- **LLM 重排与生成的质量依赖所选模型**，本项目只演示链路与提示词结构，不做提示词精调。
