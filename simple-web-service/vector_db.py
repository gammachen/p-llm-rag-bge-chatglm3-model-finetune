import faiss
import numpy as np
import os
import pickle
import logging
import time
from datetime import datetime


class FaissVectorDB:
    def __init__(self, index_path="vector_store/faiss_index.bin", metadata_path="vector_store/metadata.pkl"):
        # 设置日志
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        
        self.logger.info(f"🚀 初始化FaissVectorDB")
        self.logger.info(f"📁 索引路径: {index_path}")
        self.logger.info(f"📁 元数据路径: {metadata_path}")
        
        start_time = time.time()
        
        # 加载现有索引或创建新索引
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.logger.info("📂 检测到现有索引文件，开始加载...")
            self.load_index()
            self.logger.info(f"✅ 索引加载完成，共{len(self.metadata)}条记录")
        else:
            self.logger.info("🆕 未检测到索引文件，创建新索引...")
            self.create_index()
            self.logger.info("✅ 新索引创建完成")
            
        elapsed_time = time.time() - start_time
        self.logger.info(f"⏱️  初始化耗时: {elapsed_time:.2f}秒")
    
    def create_index(self, dim=1536):
        """创建新的FAISS索引"""
        start_time = time.time()
        self.logger.info(f"🔧 开始创建FAISS索引，维度: {dim}")
        
        self.index = faiss.IndexFlatIP(dim)  # 使用内积相似度
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"✅ FAISS索引创建完成，耗时: {elapsed_time:.4f}秒")
    
    def load_index(self):
        """加载现有索引"""
        start_time = time.time()
        self.logger.info("📥 开始加载FAISS索引...")
        
        try:
            self.index = faiss.read_index(self.index_path)
            self.logger.info(f"📊 索引维度: {self.index.d}")
            self.logger.info(f"📈 索引中向量数量: {self.index.ntotal}")
            
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
                
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ 索引和元数据加载完成，耗时: {elapsed_time:.4f}秒")
            
        except Exception as e:
            self.logger.error(f"❌ 加载索引时出错: {e}")
            raise
    
    def save_index(self):
        """保存索引到文件"""
        start_time = time.time()
        self.logger.info("💾 开始保存索引到文件...")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # 保存FAISS索引
            faiss.write_index(self.index, self.index_path)
            self.logger.info(f"📊 已保存索引，包含 {self.index.ntotal} 个向量")
            
            # 保存元数据
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            self.logger.info(f"📋 已保存元数据，共 {len(self.metadata)} 条记录")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ 索引保存完成，耗时: {elapsed_time:.4f}秒")
            
        except Exception as e:
            self.logger.error(f"❌ 保存索引时出错: {e}")
            raise
    
    def add_documents(self, vectors, documents, metadatas):
        """添加文档向量到索引"""
        start_time = time.time()
        self.logger.info(f"📥 开始添加文档到索引...")
        
        try:
            vectors = np.array(vectors).astype('float32')
            original_count = self.index.ntotal
            
            # 检查维度并重新创建索引（如果需要）
            if self.index.ntotal == 0 and len(vectors) > 0:
                vector_dim = vectors.shape[1]
                if vector_dim != self.index.d:
                    self.logger.info(f"🔄 检测到维度变化，重新创建索引: {self.index.d} -> {vector_dim}")
                    self.create_index(dim=vector_dim)
            
            self.logger.info(f"📊 原始向量数量: {original_count}")
            self.logger.info(f"📊 待添加向量数量: {len(vectors)}")
            self.logger.info(f"📊 向量维度: {vectors.shape[1]}")
            
            # 添加向量到索引
            self.index.add(vectors)
            added_count = self.index.ntotal - original_count
            self.logger.info(f"✅ 成功添加 {added_count} 个向量")
            
            # 更新元数据
            start_idx = len(self.metadata)
            self.logger.info(f"📝 更新元数据，起始ID: {start_idx}")
            
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
                self.logger.debug(f"📄 添加文档 {start_idx + i}: {doc_preview}")
                
                self.metadata.append({
                    "id": start_idx + i,
                    "document": doc,
                    "metadata": meta
                })
            
            # 保存索引
            self.save_index()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ 文档添加完成，总耗时: {elapsed_time:.4f}秒")
            
        except Exception as e:
            self.logger.error(f"❌ 添加文档时出错: {e}")
            raise
    
    def search(self, query_vector, k=5):
        """相似性搜索"""
        start_time = time.time()
        self.logger.info(f"🔍 开始相似性搜索，k={k}")
        
        try:
            query_vector = np.array([query_vector]).astype('float32')
            
            if self.index.ntotal == 0:
                self.logger.warning("⚠️  索引为空，无搜索结果")
                return []
            
            self.logger.info(f"📊 索引中总向量数: {self.index.ntotal}")
            
            # 执行搜索
            distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            # 获取相关文档和元数据
            results = []
            self.logger.info("📋 搜索结果:")
            
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx >= 0 and idx < len(self.metadata):
                    result = {
                        **self.metadata[idx],
                        "score": float(dist)
                    }
                    results.append(result)
                    
                    doc_preview = result["document"][:50] + "..." if len(result["document"]) > 50 else result["document"]
                    self.logger.info(f"   {i+1}. ID={idx}, 分数={dist:.4f}, 内容={doc_preview}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ 搜索完成，找到 {len(results)} 个结果，耗时: {elapsed_time:.4f}秒")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 搜索时出错: {e}")
            raise
    
    def clear(self):
        """清空数据库，删除所有向量和元数据"""
        self.logger.info("🗑️ 开始清空数据库...")
        
        try:
            # 重新创建空索引
            if self.index:
                dim = self.index.d
                self.create_index(dim=dim)
            else:
                self.create_index()
            
            # 清空元数据
            self.metadata = []
            
            # 删除索引文件
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
                self.logger.info(f"✅ 已删除索引文件: {self.index_path}")
            
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
                self.logger.info(f"✅ 已删除元数据文件: {self.metadata_path}")
            
            self.logger.info("✅ 数据库已清空")
            
        except Exception as e:
            self.logger.error(f"❌ 清空数据库时出错: {e}")
            raise

    def get_stats(self):
        """获取数据库统计信息"""
        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.metadata),
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
            "index_exists": os.path.exists(self.index_path),
            "metadata_exists": os.path.exists(self.metadata_path)
        }
        
        self.logger.info("📊 数据库统计信息:")
        for key, value in stats.items():
            self.logger.info(f"   {key}: {value}")
        
        return stats