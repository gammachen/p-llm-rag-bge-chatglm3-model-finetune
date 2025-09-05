from sentence_transformers import SentenceTransformer
import numpy as np


class BGEM3Embedder:
    def __init__(self):
        # self.model_id = "BAAI/bge-m3"
        self.model_id = "BAAI/bge-small-zh-v1.5"
        self.model = SentenceTransformer(self.model_id)
    
    def embed_texts(self, texts):
        """向量化文本列表"""
        embeddings = self.model.encode(texts)
        return embeddings
    
    def embed_query(self, query):
        """向量化查询"""
        return self.embed_texts([query])[0]
