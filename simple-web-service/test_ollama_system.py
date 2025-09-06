#!/usr/bin/env python3
"""
测试Ollama RAG系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_embedder import OllamaEmbedder
from gpt35_turbo_rag import Gpt35TurboRAG
from vector_db import FaissVectorDB
import document_loader

def test_ollama_connection():
    """测试Ollama连接"""
    print("🔍 测试Ollama连接...")
    
    try:
        import requests
        url = "http://localhost:11434/api/tags"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            required_models = ["text-embedding-ada-002:latest", "gpt-3.5-turbo:latest"]
            missing_models = [m for m in required_models if m not in model_names]
            
            if not missing_models:
                print(f"✅ 成功连接到Ollama，所有必需模型已安装")
                print(f"📊 可用模型: {model_names}")
            else:
                print(f"⚠️  缺少模型: {missing_models}")
                print(f"📊 已安装模型: {model_names}")
                
            return True
        else:
            print(f"❌ Ollama连接失败: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("请确保:")
        print("1. Ollama正在运行: ollama serve")
        print("2. 已安装所需模型:")
        print("   ollama pull text-embedding-ada-002:latest")
        print("   ollama pull gpt-3.5-turbo:latest")
        return False
    
    return False

def test_embedding():
    """测试嵌入功能"""
    print("\n🔍 测试嵌入功能...")
    
    try:
        embedder = OllamaEmbedder()
        test_text = "这是一个测试文本"
        
        # 测试单个查询嵌入
        query_embedding = embedder.embed_query(test_text)
        print(f"✅ 查询嵌入成功，维度: {len(query_embedding)}")
        
        # 测试文档嵌入
        docs = ["测试文档1", "测试文档2", "测试文档3"]
        doc_embeddings = embedder.embed_documents(docs)
        print(f"✅ 文档嵌入成功，数量: {len(doc_embeddings)}, 维度: {len(doc_embeddings[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 嵌入测试失败: {e}")
        return False

def test_vector_db():
    """测试向量数据库"""
    print("\n🔍 测试向量数据库...")
    
    try:
        # 创建新的向量数据库（使用新的文件名避免冲突）
        vector_db = FaissVectorDB(
            index_path="./vector_store/test_faiss_index.bin",
            metadata_path="./vector_store/test_metadata.pkl"
        )
        
        # 创建嵌入器
        embedder = OllamaEmbedder()
        
        # 测试文档
        test_docs = [
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器。",
            "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习并做出预测的算法。",
            "深度学习是机器学习的一个分支，使用神经网络来处理复杂的模式识别任务。"
        ]
        
        # 生成嵌入
        embeddings = embedder.embed_documents(test_docs)
        
        # 添加文档到数据库
        metadatas = [{"source": f"test_doc_{i}"} for i in range(len(test_docs))]
        vector_db.add_documents(embeddings, test_docs, metadatas)
        
        # 测试搜索
        query = "什么是人工智能"
        query_embedding = embedder.embed_query(query)
        results = vector_db.search(query_embedding, k=2)
        
        print(f"✅ 向量数据库测试成功")
        print(f"📊 文档数量: {len(test_docs)}")
        print(f"🔍 搜索结果数量: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"   {i+1}. 分数: {result['score']:.3f}, 内容: {result['document'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 向量数据库测试失败: {e}")
        return False

def test_rag_system():
    """测试完整的RAG系统"""
    print("\n🔍 测试RAG系统...")
    
    try:
        # 创建向量数据库
        vector_db = FaissVectorDB(
            index_path="./vector_store/test_faiss_index.bin",
            metadata_path="./vector_store/test_metadata.pkl"
        )
        
        # 创建RAG系统
        rag = Gpt35TurboRAG(
            model_name="gpt-3.5-turbo:latest"
        )
        
        # 测试查询
        test_query = "什么是机器学习"
        
        # 模拟一些上下文文档
        context_docs = [
            {
                'document': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器。',
                'metadata': {'source': 'test_doc_1.txt'}
            },
            {
                'document': '机器学习是人工智能的一个子领域，专注于开发能够从数据中学习并做出预测的算法。',
                'metadata': {'source': 'test_doc_2.txt'}
            }
        ]
        
        answer, references = rag.generate_response(test_query, context_docs)
        
        print(f"✅ RAG系统测试成功")
        print(f"🤖 回答: {answer[:100]}...")
        print(f"📚 来源数量: {len(references)}")
        return True
            
    except Exception as e:
        print(f"❌ RAG系统测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Ollama RAG系统测试开始")
    print("=" * 50)
    
    # 测试步骤
    tests = [
        ("Ollama连接", test_ollama_connection),
        ("嵌入功能", test_embedding),
        ("向量数据库", test_vector_db),
        ("RAG系统", test_rag_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
        
    # 总结
    print(f"\n{'='*50}")
    print("📊 测试结果总结:")
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n📈 通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！可以启动Streamlit应用了")
        print("启动命令: streamlit run app_ollama.py")
    else:
        print("⚠️  部分测试失败，请检查上面的错误信息")

if __name__ == "__main__":
    main()