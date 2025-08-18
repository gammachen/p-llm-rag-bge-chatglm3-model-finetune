from transformers import AutoModel, AutoTokenizer


class ChatGLM3RAG:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).half().cuda().eval()  # 使用GPU加速
    
    def generate_response(self, query, context_docs):
        """基于检索内容生成回答"""
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