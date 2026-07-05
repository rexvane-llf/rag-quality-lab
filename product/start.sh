#!/bin/bash

# 个人知识库 AI 助手 - 启动脚本

echo "🚀 启动个人知识库 AI 助手"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python 3，请先安装"
    exit 1
fi

# 检查依赖
if [ ! -d "backend/venv" ]; then
    echo "📦 创建虚拟环境..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
fi

# 启动后端
echo "🔧 启动后端服务..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

echo "✅ 后端服务已启动 (PID: $BACKEND_PID)"
echo "📚 API 文档: http://localhost:8000/docs"
echo ""

# 检查 Node.js
if command -v npm &> /dev/null; then
    echo "🎨 检查前端..."
    cd frontend

    if [ ! -d "node_modules" ]; then
        echo "📦 安装前端依赖..."
        npm install
    fi

    echo "🚀 启动前端服务..."
    npm run dev &
    FRONTEND_PID=$!
    cd ..

    echo "✅ 前端服务已启动 (PID: $FRONTEND_PID)"
    echo "🌐 Web 界面: http://localhost:3000"
else
    echo "⚠️  未找到 Node.js，跳过前端启动"
    echo "   只使用 API: http://localhost:8000/docs"
fi

echo ""
echo "按 Ctrl+C 停止服务"

# 等待
wait
