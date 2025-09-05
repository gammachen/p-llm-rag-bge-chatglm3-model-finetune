import torch
import os
from transformers import AutoModel, AutoTokenizer


class ChatGLM3RAG:
    
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        self.model_name = model_name
        self.mock_mode = False
        
        # 检测CUDA是否可用，自动选择设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        try:
            # 尝试加载本地模型
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            local_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
            
            if os.path.exists(local_path):
                print(f"使用本地模型: {local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_path, trust_remote_code=True, local_files_only=True
                )
                self.model = AutoModel.from_pretrained(
                    local_path, trust_remote_code=True, local_files_only=True
                )
            else:
                print(f"使用在线模型: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True
                )
            
            if self.device == "cuda":
                self.model = self.model.half().cuda().eval()  # GPU模式
            else:
                self.model = self.model.float().eval()  # CPU模式
                
            print(f"成功加载模型: {model_name}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("启用模拟模式...")
            self.mock_mode = True
    
    def generate_response(self, query, context_docs):
        """基于检索内容生成回答"""
        if self.mock_mode:
            # 模拟回答
            response = f"这是一个模拟回答，基于您的问题'{query}'和提供的{len(context_docs)}个文档。"
            references = [{"text": doc['document'][:100] + "...", "source": doc['metadata']['source']} 
                          for doc in context_docs]
            return response, references
        
        # 构建提示
        context = "\n\n".join([doc['document'] for doc in context_docs])
        prompt = f"基于以下信息回答问题:\n\n{context}\n\n问题: {query}\n回答:"
        
        # 生成回答
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            temperature=0.7,
            max_length=2048
        )
        
        # 添加引用信息
        references = [{"text": doc['document'], "source": doc['metadata']['source']} 
                      for doc in context_docs]
        return response, references