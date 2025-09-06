import requests
import json
import os
from typing import List, Dict, Any


class Gpt35TurboRAG:
    """使用Ollama的gpt-3.5-turbo:latest模型的RAG系统"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.mock_mode = False
        
        # 测试连接
        self._test_connection()
    
    def _test_connection(self):
        """测试与Ollama的连接"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                if self.model_name in model_names:
                    print(f"✅ 成功连接到Ollama，模型 {self.model_name} 可用")
                else:
                    print(f"⚠️ 模型 {self.model_name} 未找到，可用模型: {model_names}")
                    self.mock_mode = True
            else:
                print(f"❌ 无法连接到Ollama: {response.status_code}")
                self.mock_mode = True
        except Exception as e:
            print(f"❌ 连接Ollama失败: {e}")
            print("启用模拟模式...")
            self.mock_mode = True
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """基于检索内容生成回答"""
        if self.mock_mode:
            # 模拟回答
            response = f"这是一个模拟回答，基于您的问题'{query}'和提供的{len(context_docs)}个文档。"
            references = [{"text": doc['document'][:100] + "...", "source": doc['metadata']['source']} 
                          for doc in context_docs]
            return response, references

        # 构建提示
        context = "\n\n".join([doc['document'] for doc in context_docs])
        
        # 限制上下文长度
        max_context_length = 3000  # 为GPT-3.5-turbo预留空间
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的问答助手。请基于提供的上下文信息准确、简洁地回答用户的问题。如果信息不足以回答问题，请明确说明。"
            },
            {
                "role": "user",
                "content": f"基于以下信息回答问题:\n\n{context}\n\n问题: {query}"
            }
        ]
        
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 512
                }
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("message", {}).get("content", "无法生成回答")
            
            # 添加引用信息
            references = [{"text": doc['document'], "source": doc['metadata']['source']} 
                          for doc in context_docs]
            
            return answer, references
            
        except Exception as e:
            print(f"生成回答时出错: {e}")
            response = f"抱歉，生成回答时出现错误: {str(e)[:100]}..."
            references = [{"text": doc['document'], "source": doc['metadata']['source']} 
                          for doc in context_docs]
            return response, references
    
    def generate_streaming_response(self, query: str, context_docs: List[Dict[str, Any]]):
        """生成流式响应（用于Web界面）"""
        if self.mock_mode:
            yield f"这是一个模拟流式回答，基于您的问题'{query}'..."
            return

        context = "\n\n".join([doc['document'] for doc in context_docs])
        max_context_length = 3000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的问答助手。请基于提供的上下文信息准确、简洁地回答用户的问题。"
            },
            {
                "role": "user",
                "content": f"基于以下信息回答问题:\n\n{context}\n\n问题: {query}"
            }
        ]
        
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 512
                }
            }
            
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if 'message' in data and 'content' in data['message']:
                            yield data['message']['content']
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"生成回答时出错: {str(e)[:100]}..."