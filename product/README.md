# 产品版 - 个人知识库 AI 助手

基于 rag-quality-lab 的核心检索引擎，扩展为完整的个人知识库产品。

## 功能特性

- ✅ 完整的 REST API（10 个接口）
- ✅ SQLite 数据持久化
- ✅ 文档管理（上传/删除/列表）
- ✅ 智能检索（BM25 + 向量混合）
- ✅ 对话历史保存
- ✅ 本地优先，隐私安全

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
python test_api.py

# 启动服务
python main.py
```

访问 http://localhost:8000/docs 查看完整 API 文档

## 目录说明

- `main.py` - FastAPI 主应用
- `database.py` - 数据库模型
- `document_processor.py` - 文档处理器
- `test_api.py` - 功能测试

核心检索模块（复用教学版）：
- `chunking.py` - 文档切分
- `index.py` - BM25/向量索引
- `retrieval.py` - 混合检索引擎

## API 使用

### 上传文档
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.pdf"
```

### 检索查询
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"question":"Python装饰器是什么","top_k":5}'
```

### 对话问答
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是索引"}'
```

## 下一步

- [ ] React 前端界面
- [ ] LLM 集成（生成式回答）
- [ ] Electron 桌面应用
- [ ] 文件夹监听
- [ ] OCR 支持
