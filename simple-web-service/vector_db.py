import faiss
import numpy as np
import os
import pickle


class FaissVectorDB:
    def __init__(self, index_path="vector_store/faiss_index.bin", metadata_path="vector_store/metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        
        # 加载现有索引或创建新索引
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load_index()
        else:
            self.create_index()
    
    def create_index(self, dim=1024):
        """创建新的FAISS索引"""
        self.index = faiss.IndexFlatIP(dim)  # 使用内积相似度
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def load_index(self):
        """加载现有索引"""
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
    
    def save_index(self):
        """保存索引到文件"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
    
    def add_documents(self, vectors, documents, metadatas):
        """添加文档向量到索引"""
        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)
        
        # 更新元数据
        start_idx = len(self.metadata)
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            self.metadata.append({
                "id": start_idx + i,
                "document": doc,
                "metadata": meta
            })
        self.save_index()
    
    def search(self, query_vector, k=5):
        """相似性搜索"""
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        # 获取相关文档和元数据
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    **self.metadata[idx],
                    "score": float(dist)
                })
        return results