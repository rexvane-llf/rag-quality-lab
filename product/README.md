# 产品版 - 个人知识库 AI 助手

基于 rag-quality-lab 的核心检索引擎，扩展为完整的个人知识库产品。

## 功能特性

### 后端
- ✅ 完整的 REST API（10 个接口）
- ✅ SQLite 数据持久化
- ✅ 文档管理（上传/删除/列表/扫描）
- ✅ 智能检索（BM25 + 向量混合）
- ✅ 对话历史保存
- ✅ 本地优先，隐私安全

### 前端（新增）
- ✅ React 18 + Vite + TailwindCSS
- ✅ 三个主要页面：聊天/文档管理/对话历史
- ✅ 文档拖拽上传
- ✅ 实时问答界面
- ✅ 响应式设计

## 快速开始

### 1. 启动后端

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 运行测试
python test_api.py

# 启动服务
python main.py
```

访问 http://localhost:8000/docs 查看完整 API 文档

### 2. 启动前端

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

访问 http://localhost:3000 使用 Web 界面

## 项目结构

```
product/
├── backend/
│   ├── main.py              # FastAPI 主应用
│   ├── database.py          # 数据库模型
│   ├── document_processor.py # 文档处理器
│   ├── chunking.py          # 文档切分
│   ├── index.py             # BM25/向量索引
│   ├── retrieval.py         # 混合检索引擎
│   ├── requirements.txt     # Python 依赖
│   └── test_api.py          # 测试脚本
└── frontend/
    ├── src/
    │   ├── App.jsx          # 主组件
    │   ├── main.jsx         # 入口
    │   └── index.css        # 样式
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── tailwind.config.js
```

## 技术栈

### 后端
- FastAPI - Web 框架
- SQLAlchemy - ORM
- SQLite - 数据库
- PyMuPDF - PDF 解析
- 复用教学版核心：chunking, index, retrieval

### 前端
- React 18 - UI 框架
- Vite - 构建工具
- TailwindCSS - 样式框架
- Axios - HTTP 客户端

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传文档并索引 |
| `/api/scan` | POST | 扫描文件夹批量索引 |
| `/api/documents` | GET | 获取文档列表 |
| `/api/documents/{id}` | DELETE | 删除文档 |
| `/api/search` | POST | 检索知识库 |
| `/api/chat` | POST | 对话问答 |
| `/api/conversations` | GET | 查看对话历史 |
| `/api/rebuild-index` | POST | 重建检索索引 |
| `/api/health` | GET | 健康检查 |

## 使用示例

### 通过 API

```bash
# 上传文档
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.pdf"

# 查询知识库
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"question":"Python装饰器是什么","top_k":5}'

# 对话问答
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是索引"}'
```

### 通过 Web UI

1. 打开 http://localhost:3000
2. 在"文档管理"页面上传文档
3. 在"聊天"页面提问
4. 在"对话历史"查看记录

## 截图

### 聊天界面
- 类似 ChatGPT 的对话体验
- 显示答案来源和引用

### 文档管理
- 文档列表展示
- 一键上传
- 删除操作

### 对话历史
- 历史记录查看
- 时间排序

## 下一步

- [ ] LLM 集成（生成式回答）
- [ ] Electron 桌面应用
- [ ] 文件夹监听
- [ ] OCR 支持
- [ ] 用户认证
- [ ] 多知识库管理
