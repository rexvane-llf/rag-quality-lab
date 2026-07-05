# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.0] - 2026-07-05

### Added
- 可替换的检索侧四模块：切分（`lab/chunking.py`，fixed/sentence/structure）、索引（`lab/index.py`，BM25 + TF-IDF）、检索（`lab/retrieval.py`，bm25/vector/hybrid-RRF）、评测（`lab/evaluate.py`，recall@k · MRR · 延迟）
- `run_matrix.py`：一键跑 3×3 策略矩阵，产出 `report.md` 对比报告——纯 Python 标准库，零依赖、无需 API Key
- `ask.py` 交互问答：无 Key 列出处；有 Key 走改写 → 检索 → 重排 → 带 `[1][2]` 引用生成
- 可选 LLM 增强（`lab/llm_extras.py`）：query_rewrite / llm_rerank / answer_with_citations，双通道（Anthropic + OpenAI 兼容协议）
- 评测数据集：4 篇语料 + 12 题 golden 标注（`data/`）
- 纯检索层单元测试（无需 API Key）
