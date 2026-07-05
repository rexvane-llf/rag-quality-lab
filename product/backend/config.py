"""config.py - 配置管理"""
import os
from pathlib import Path

# 基础路径
BASE_DIR = Path.home() / ".personal-kb"
BASE_DIR.mkdir(exist_ok=True)

# 数据库配置
DATABASE_URL = f"sqlite:///{BASE_DIR}/knowledge.db"

# 文件上传配置
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB

# 支持的文件类型
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

# 检索配置
DEFAULT_TOP_K = 5
DEFAULT_STRATEGY = "hybrid"  # bm25, vector, hybrid

# 分块配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# API 配置
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

# LLM 配置（可选）
LLM_PROVIDER = os.getenv("LLM_PROVIDER")  # anthropic, openai, deepseek
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-sonnet-20240229")
