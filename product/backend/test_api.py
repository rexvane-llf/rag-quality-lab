"""test_api.py - 快速测试 API 功能"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from database import SessionLocal, Document, Chunk
from document_processor import DocumentProcessor


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试个人知识库核心功能 ===\n")

    # 1. 测试数据库连接
    print("1. 测试数据库连接...")
    db = SessionLocal()
    doc_count = db.query(Document).count()
    chunk_count = db.query(Chunk).count()
    print(f"   ✓ 数据库连接成功")
    print(f"   ✓ 当前文档数: {doc_count}")
    print(f"   ✓ 当前切片数: {chunk_count}\n")

    # 2. 测试文档处理器
    print("2. 测试文档处理器...")
    processor = DocumentProcessor(db)

    # 创建测试文档
    test_dir = os.path.expanduser("~/.personal-kb/test_docs")
    os.makedirs(test_dir, exist_ok=True)

    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
# 测试文档

这是一个测试文档，用于验证个人知识库的核心功能。

## Python 装饰器

装饰器是 Python 中的一种设计模式，可以在不修改函数定义的情况下，
增加函数的功能。装饰器本质上是一个接受函数作为参数的函数。

## 常见用法

1. 日志记录
2. 性能测试
3. 权限验证
4. 缓存
        """)

    print(f"   ✓ 创建测试文档: {test_file}")

    # 3. 处理文档
    print("\n3. 处理和索引文档...")
    doc = processor.process_document(test_file)
    if doc:
        print(f"   ✓ 文档已索引")
        print(f"   ✓ 文档 ID: {doc.id}")
        print(f"   ✓ 切片数量: {len(doc.chunks)}")
    else:
        print("   ✗ 文档处理失败")
        return False

    # 4. 测试检索
    print("\n4. 测试检索功能...")
    processor.build_retriever()

    query = "Python 装饰器是什么"
    results = processor.search(query, top_k=3)

    print(f"   ✓ 查询: {query}")
    print(f"   ✓ 找到 {len(results)} 条结果\n")

    for i, result in enumerate(results, 1):
        print(f"   [{i}] {result['document']} (评分: {result['score']:.3f})")
        print(f"       {result['content'][:100]}...\n")

    db.close()
    print("=== 所有测试通过 ✓ ===")
    return True


if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
