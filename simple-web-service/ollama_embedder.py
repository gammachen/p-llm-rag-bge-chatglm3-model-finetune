import requests
import json
import numpy as np
from typing import List, Union
import os


class OllamaEmbedder:
    """使用Ollama的text-embedding-ada-002:latest模型的嵌入器"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_dim = 1536  # text-embedding-ada-002的维度
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入向量"""
        return self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> List[float]:
        """调用Ollama API获取单个文本的嵌入"""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            # 返回零向量作为后备
            return [0.0] * self.embedding_dim
    
    def __call__(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """使类实例可调用"""
        if isinstance(texts, str):
            return self.embed_query(texts)
        else:
            return self.embed_documents(texts)