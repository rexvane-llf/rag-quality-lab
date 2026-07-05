"""document_processor.py - 文档处理和索引"""
import os
import hashlib
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF
from datetime import datetime

from database import Document, Chunk, SessionLocal
from chunking import Chunker
from index import BM25Index, TfidfIndex
from retrieval import Retriever


class DocumentProcessor:
    """文档处理器：扫描、解析、切分、索引"""

    def __init__(self, db_session=None):
        self.db = db_session or SessionLocal()
        self.chunker = Chunker()
        self.retriever = None  # 延迟初始化

    def calculate_hash(self, file_path: str) -> str:
        """计算文件 MD5 哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def extract_text_from_pdf(self, file_path: str) -> List[tuple]:
        """从 PDF 提取文本，返回 [(page_num, text), ...]"""
        doc = fitz.open(file_path)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((page_num, text))
        doc.close()
        return pages

    def extract_text_from_txt(self, file_path: str) -> str:
        """从 TXT 文件提取文本"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract_text_from_md(self, file_path: str) -> str:
        """从 Markdown 文件提取文本"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def process_document(self, file_path: str) -> Optional[Document]:
        """处理单个文档：提取文本、切分、存储"""
        file_path = os.path.abspath(file_path)

        # 检查文件是否已存在
        existing = self.db.query(Document).filter(Document.path == file_path).first()
        file_hash = self.calculate_hash(file_path)

        if existing and existing.hash == file_hash:
            print(f"文档未变化，跳过: {file_path}")
            return existing

        # 提取文本
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            pages = self.extract_text_from_pdf(file_path)
            full_text = "\n\n".join([text for _, text in pages])
        elif file_ext in ['.txt', '.md']:
            full_text = self.extract_text_from_txt(file_path)
            pages = [(1, full_text)]
        else:
            print(f"不支持的文件类型: {file_ext}")
            return None

        # 切分文档
        chunks = self.chunker.chunk_text(
            full_text,
            strategy='structure',  # 使用结构感知切分
            doc_id=file_path
        )

        # 保存或更新文档
        if existing:
            doc = existing
            doc.hash = file_hash
            doc.updated_at = datetime.utcnow()
            doc.last_indexed = datetime.utcnow()
            # 删除旧的 chunks
            self.db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        else:
            doc = Document(
                path=file_path,
                name=Path(file_path).name,
                type=file_ext[1:],
                size=os.path.getsize(file_path),
                hash=file_hash,
                folder=str(Path(file_path).parent),
                last_indexed=datetime.utcnow()
            )
            self.db.add(doc)
            self.db.flush()  # 获取 doc.id

        # 保存 chunks
        for idx, chunk in enumerate(chunks):
            db_chunk = Chunk(
                document_id=doc.id,
                content=chunk.text,
                chunk_index=idx,
                page_number=chunk.metadata.get('page_number'),
                heading=chunk.heading
            )
            self.db.add(db_chunk)

        self.db.commit()
        print(f"✓ 已索引文档: {file_path} ({len(chunks)} 个切片)")
        return doc

    def scan_folder(self, folder_path: str) -> List[Document]:
        """扫描文件夹并处理所有文档"""
        folder_path = os.path.abspath(folder_path)
        supported_exts = {'.pdf', '.txt', '.md'}
        docs = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                if Path(file).suffix.lower() in supported_exts:
                    file_path = os.path.join(root, file)
                    try:
                        doc = self.process_document(file_path)
                        if doc:
                            docs.append(doc)
                    except Exception as e:
                        print(f"✗ 处理文档失败 {file_path}: {e}")

        return docs

    def build_retriever(self) -> Retriever:
        """构建检索器"""
        # 从数据库加载所有 chunks
        db_chunks = self.db.query(Chunk).all()

        # 转换为 Retriever 需要的格式
        from chunking import Chunk as ChunkModel
        chunks = []
        for db_chunk in db_chunks:
            chunk = ChunkModel(
                doc_id=str(db_chunk.document_id),
                chunk_id=f"{db_chunk.document_id}_{db_chunk.chunk_index}",
                text=db_chunk.content,
                heading=db_chunk.heading or ""
            )
            chunks.append(chunk)

        if not chunks:
            print("⚠️ 没有可用的文档切片")
            return None

        self.retriever = Retriever(chunks)
        print(f"✓ 检索器已构建，包含 {len(chunks)} 个切片")
        return self.retriever

    def search(self, query: str, top_k: int = 5, strategy: str = 'hybrid'):
        """检索文档"""
        if not self.retriever:
            self.build_retriever()

        if not self.retriever:
            return []

        results = self.retriever.search(query, k=top_k, strategy=strategy)

        # 补充文档信息
        enriched_results = []
        for chunk, score in results:
            doc_id = int(chunk.doc_id)
            doc = self.db.query(Document).filter(Document.id == doc_id).first()
            enriched_results.append({
                'content': chunk.text,
                'score': score,
                'document': doc.name if doc else 'Unknown',
                'heading': chunk.heading
            })

        return enriched_results
