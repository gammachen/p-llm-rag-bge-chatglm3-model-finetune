import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置MPS设备回退到CPU的环境变量
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class ChatGLM3RAG:
    
    def __init__(self, model_name="Qwen/Qwen2-1.5B", use_small_model=True):
        self.model_name = model_name
        self.mock_mode = False
        
        # 如果内存不足，使用更小的模型
        if use_small_model:
            self.model_name = "microsoft/DialoGPT-small"
        
        # 检测设备：优先CUDA，其次MPS（Apple Silicon），最后CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        # elif torch.backends.mps.is_available():
        #     self.device = "mps"
        else:
            self.device = "cpu"
        print(f"使用设备: {self.device}")
        print(f"使用模型: {self.model_name}")
        
        try:
            # 尝试加载本地模型
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            local_path = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
            local_path = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{self.model_name.replace('/', '--')}"
            )
            if os.path.exists(local_path):
                snapshot_path = os.path.join(local_path, "snapshots")
                if os.path.exists(snapshot_path):
                    snapshot_dirs = os.listdir(snapshot_path)
                    if snapshot_dirs:
                        local_path = os.path.join(snapshot_path, snapshot_dirs[0])
                    else:
                        local_path = None
                else:
                    local_path = None
            else:
                local_path = None
            
            # 临时设置，让模型在线加载到本地的先
            local_path = ""
            
            if local_path and os.path.exists(local_path):
                print(f"使用本地模型: {local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_path, trust_remote_code=True, local_files_only=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_path, 
                    trust_remote_code=True, 
                    local_files_only=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                print(f"使用在线模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            
            if self.device == "cuda":
                self.model = self.model.half().cuda().eval()  # GPU模式
            else:
                self.model = self.model.float().eval()  # CPU模式
                
            print(f"成功加载模型: {self.model_name}")
            
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
        
        try:
            # 更精确地限制上下文长度，为生成留出足够空间
            max_input_length = 700  # 减少输入长度，为生成留出空间
            
            # 使用tokenizer进行更精确的截断
            encoded_prompt = self.tokenizer.encode(prompt, max_length=max_input_length, truncation=True)
            prompt = self.tokenizer.decode(encoded_prompt, skip_special_tokens=True)
            
            # 使用generate方法生成回答 - 修复MPS设备兼容性
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=max_input_length, truncation=True)
            
            # 确保输入在正确的设备上
            inputs = inputs.to(self.model.device)
            
            # 创建注意力掩码，解决MPS设备问题
            attention_mask = torch.ones_like(inputs)
            
            try:
                with torch.no_grad():
                    # 使用max_new_tokens而不是max_length，避免冲突
                    max_new_tokens = min(256, 1024 - inputs.shape[1])
                    if max_new_tokens < 20:
                        max_new_tokens = 50  # 确保至少有最小生成长度
                    
                    # 在MPS设备上，某些操作可能不支持，尝试使用CPU回退
                    try:
                        outputs = self.model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,  # 只限制新生成的token数量
                            min_new_tokens=20,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    except RuntimeError as e:
                        if "MPS" in str(e) or "not currently implemented" in str(e):
                            print(f"MPS操作不支持，回退到CPU: {e}")
                            # 回退到CPU
                            inputs_cpu = inputs.cpu()
                            attention_mask_cpu = attention_mask.cpu()
                            model_cpu = self.model.cpu()
                            
                            outputs = model_cpu.generate(
                                inputs_cpu,
                                attention_mask=attention_mask_cpu,
                                max_new_tokens=max_new_tokens,
                                min_new_tokens=20,
                                temperature=0.7,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id
                            )
                        else:
                            raise e
                            
            except Exception as e:
                print(f"生成回答时出错: {e}")
                response = f"抱歉，生成回答时出现错误: {str(e)[:100]}..."
                references = [{"text": doc['document'], "source": doc['metadata']['source']} 
                              for doc in context_docs]
                return response, references
            
            # 解码生成的文本
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 只保留回答部分
            if "回答:" in response:
                response = response.split("回答:")[-1].strip()
            else:
                response = response[len(prompt):].strip()
                
            # 确保响应不为空
            if not response or len(response.strip()) < 5:
                response = "基于提供的文档信息，我无法生成具体的回答。"
                
        except Exception as e:
            print(f"生成回答时出错: {e}")
            response = f"抱歉，生成回答时出现错误: {str(e)[:100]}..."
        
        # 添加引用信息
        references = [{"text": doc['document'], "source": doc['metadata']['source']} 
                      for doc in context_docs]
        return response, references