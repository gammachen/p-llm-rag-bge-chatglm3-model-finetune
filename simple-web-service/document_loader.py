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
    """文本分块处理"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks