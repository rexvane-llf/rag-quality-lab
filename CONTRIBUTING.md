# 贡献指南

感谢关注 rag-quality-lab！欢迎提 Issue 和 PR。

## 开发环境

检索与评测层是纯 Python 标准库，克隆即可跑：

```bash
git clone https://github.com/RexVane/rag-quality-lab.git
cd rag-quality-lab
python run_matrix.py              # 零依赖跑完整评测矩阵

pip install -r requirements.txt   # 仅在需要 LLM 增强 / 跑测试时安装
```

## 运行测试

测试只覆盖纯检索层，**不需要任何 API Key**：

```bash
python -m pytest tests/ -q
```

提交 PR 前请确保测试全部通过；新增切分/检索策略请附带对应测试与评测结果。

## 提交规范

提交信息采用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 风格：

```
feat(retrieval): 新增 ColBERT 风格 late-interaction 检索
fix(evaluate): 修复 MRR 在零命中时的除零
docs(report): 更新 3×3 矩阵基线数据
```

## PR 流程

1. Fork 本仓库并创建特性分支（`feat/xxx` 或 `fix/xxx`）
2. 完成修改并保证 `pytest` 通过
3. 提交 PR，描述清楚动机与改动点
