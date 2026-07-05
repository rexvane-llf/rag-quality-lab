"""database.py - 数据库模型和配置"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# SQLite 数据库路径
DB_PATH = os.path.expanduser("~/.personal-kb/knowledge.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# 创建数据库引擎
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    """文档表"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # pdf, txt, md, docx
    size = Column(Integer)
    hash = Column(String)  # 文件哈希，用于检测变化
    folder = Column(String)
    tags = Column(Text)  # JSON 格式的标签
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_indexed = Column(DateTime)

    # 关系
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """文档切片表"""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    page_number = Column(Integer)  # PDF 页码
    heading = Column(String)  # 章节标题

    # 关系
    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    """对话历史表"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(Text)  # JSON 格式的来源信息
    created_at = Column(DateTime, default=datetime.utcnow)


class WatchedFolder(Base):
    """监听文件夹表"""
    __tablename__ = "watched_folders"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String, unique=True, nullable=False)
    enabled = Column(Boolean, default=True)
    last_scan = Column(DateTime)


# 创建所有表
Base.metadata.create_all(bind=engine)


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
