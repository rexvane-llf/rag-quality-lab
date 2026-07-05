from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
import os
import shutil

from database import get_db, Document as DBDocument, Conversation
from document_processor import DocumentProcessor

app = FastAPI(
    title="Personal Knowledge Base API",
    description="个人知识库 AI 助手",
    version="0.1.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局文档处理器
processor = DocumentProcessor()


# Pydantic 模型
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    strategy: str = 'hybrid'


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class DocumentResponse(BaseModel):
    id: int
    name: str
    type: str
    size: int
    path: str
    created_at: str
    chunk_count: int


@app.get("/")
async def root():
    return {
        "message": "Personal Knowledge Base API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """上传并索引文档"""
    # 保存到临时目录
    upload_dir = os.path.expanduser("~/.personal-kb/uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 处理文档
        doc = processor.process_document(file_path)

        if not doc:
            raise HTTPException(status_code=400, detail="不支持的文件类型")

        return {
            "message": "文档上传成功",
            "document_id": doc.id,
            "filename": file.filename,
            "status": "indexed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scan")
async def scan_folder(
    folder_path: str,
    db: Session = Depends(get_db)
):
    """扫描文件夹并索引所有文档"""
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="文件夹不存在")

    try:
        docs = processor.scan_folder(folder_path)
        return {
            "message": f"已扫描 {len(docs)} 个文档",
            "documents": len(docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)):
    """获取所有文档列表"""
    docs = db.query(DBDocument).all()

    result = []
    for doc in docs:
        result.append(DocumentResponse(
            id=doc.id,
            name=doc.name,
            type=doc.type,
            size=doc.size,
            path=doc.path,
            created_at=doc.created_at.isoformat(),
            chunk_count=len(doc.chunks)
        ))

    return result


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """删除文档"""
    doc = db.query(DBDocument).filter(DBDocument.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    db.delete(doc)
    db.commit()

    # 重新构建检索器
    processor.build_retriever()

    return {"message": "文档已删除", "document_id": document_id}


@app.post("/api/search")
async def search_documents(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """检索文档"""
    try:
        results = processor.search(
            request.question,
            top_k=request.top_k,
            strategy=request.strategy
        )

        return {
            "query": request.question,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_with_knowledge(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """基于知识库回答问题（简单版本，无 LLM）"""
    try:
        # 检索相关文档
        results = processor.search(
            request.question,
            top_k=request.top_k,
            strategy=request.strategy
        )

        if not results:
            answer = "抱歉，我在知识库中没有找到相关信息。"
            sources = []
        else:
            # 简单拼接答案（生产环境应该调用 LLM）
            answer = "根据你的文档，以下是相关信息：\n\n"
            sources = []

            for i, result in enumerate(results[:3], 1):
                answer += f"[{i}] {result['content'][:200]}...\n\n"
                sources.append({
                    "index": i,
                    "document": result['document'],
                    "heading": result['heading'],
                    "score": result['score']
                })

        # 保存对话历史
        conversation = Conversation(
            question=request.question,
            answer=answer,
            sources=str(sources)
        )
        db.add(conversation)
        db.commit()

        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations")
async def get_conversations(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """获取对话历史"""
    conversations = db.query(Conversation)\
        .order_by(Conversation.created_at.desc())\
        .limit(limit)\
        .all()

    return [{
        "id": conv.id,
        "question": conv.question,
        "answer": conv.answer,
        "created_at": conv.created_at.isoformat()
    } for conv in conversations]


@app.post("/api/rebuild-index")
async def rebuild_index(db: Session = Depends(get_db)):
    """重建检索索引"""
    try:
        processor.build_retriever()
        return {"message": "索引重建成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("🚀 启动个人知识库 API 服务...")
    print("📚 API 文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
