import fitz  # PyMuPDF
import os


def load_pdf(file_path):
    """加载PDF文档并提取文本"""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"Error loading PDF: {e}")
    return text


def load_txt(file_path):
    """加载TXT文档"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading TXT: {e}")
        return ""


def chunk_text(text, chunk_size=512, overlap=50):
    """文本分块处理 - 优化版
    
    参数:
        text: 输入文本
        chunk_size: 每个块的最大长度
        overlap: 相邻块之间的重叠字符数
    
    返回:
        chunks: 文本块列表
    """
    # 参数验证
    if not text or not text.strip():
        print("⚠️ 输入文本为空")
        return []
    
    if chunk_size <= 0:
        print("⚠️ chunk_size必须大于0")
        return [text] if text else []
    
    if overlap < 0:
        overlap = 0
    
    # 修复潜在的无限循环问题
    if overlap >= chunk_size:
        print(f"⚠️ overlap({overlap}) >= chunk_size({chunk_size})，将overlap设为chunk_size的一半")
        overlap = max(1, chunk_size // 4)  # 使用1/4重叠作为默认值
    
    chunks = []
    text_length = len(text)
    start = 0
    chunk_count = 0
    
    # 确保至少处理一次
    if text_length > 0:
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()
            
            # 确保不添加空块
            if chunk:
                chunks.append(chunk)
                chunk_count += 1
            
            # 关键修复：确保向前推进
            new_start = end - overlap
            if new_start <= start:  # 防止无限循环
                new_start = end
            
            start = new_start
            
            # 安全机制：如果无法推进，强制推进
            if start >= text_length and end < text_length:
                start = text_length
    
    # 安全的统计信息
    if chunks:
        avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
        print(f"✅ 分块完成: {len(chunks)}个块，平均长度: {avg_length:.1f}字符")
    else:
        print("⚠️ 没有生成有效的文本块")
    
    return chunks