import streamlit as st
import os
import sys
import time
from pathlib import Path
import logging
from typing import List, Dict, Any

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_embedder import OllamaEmbedder
from gpt35_turbo_rag import Gpt35TurboRAG
from document_loader import load_pdf, load_txt, chunk_text
from vector_db import FaissVectorDB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="Ollama RAG 问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_db = None
    st.session_state.rag_engine = None
    st.session_state.embedder = None

# 缓存初始化函数
@st.cache_resource
def initialize_system():
    """初始化RAG系统"""
    try:
        logger.info("🚀 开始初始化Ollama RAG系统...")
        
        # 初始化嵌入器
        embedder = OllamaEmbedder()
        
        # 初始化RAG引擎
        rag_engine = Gpt35TurboRAG()
        
        # 初始化向量数据库
        vector_db = FaissVectorDB(
            index_path="vector_store/faiss_index_ollama.bin",
            metadata_path="vector_store/metadata_ollama.pkl"
        )
        
        logger.info("✅ Ollama RAG系统初始化完成")
        return embedder, rag_engine, vector_db
        
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        st.error(f"系统初始化失败: {str(e)}")
        return None, None, None

def process_documents(uploaded_files):
    """处理上传的文档"""
    try:
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_chunks = []
        all_metadata = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((idx + 1) / total_files)
            status_text.text(f"正在处理: {uploaded_file.name}")
            
            # 保存临时文件
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 根据文件类型加载文档
            if uploaded_file.name.endswith('.pdf'):
                text = load_pdf(temp_path)
            else:
                text = load_txt(temp_path)
            
            if text:
                # 文本分块
                chunks = chunk_text(text, chunk_size=512, overlap=50)
                
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'source': uploaded_file.name,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks)
                    })
            
            # 清理临时文件
            os.remove(temp_path)
        
        if all_chunks:
            # 生成嵌入
            status_text.text("正在生成嵌入向量...")
            embeddings = st.session_state.embedder.embed_documents(all_chunks)
            
            # 添加到向量数据库
            st.session_state.vector_db.add_documents(embeddings, all_chunks, all_metadata)
            
            progress_bar.progress(1.0)
            status_text.text("✅ 文档处理完成")
            
            st.success(f"成功处理了 {len(uploaded_files)} 个文档，共 {len(all_chunks)} 个文本块")
            
    except Exception as e:
        st.error(f"处理文档时出错: {str(e)}")
        logger.error(f"文档处理错误: {e}")

# 侧边栏配置
with st.sidebar:
    st.title("🤖 Ollama RAG系统")
    st.markdown("---")
    
    # 系统状态
    if not st.session_state.initialized:
        with st.spinner("正在初始化系统..."):
            embedder, rag_engine, vector_db = initialize_system()
            if all([embedder, rag_engine, vector_db]):
                st.session_state.embedder = embedder
                st.session_state.rag_engine = rag_engine
                st.session_state.vector_db = vector_db
                st.session_state.initialized = True
                st.success("✅ 系统初始化完成")
            else:
                st.error("❌ 系统初始化失败")
                st.stop()
    
    # 文档上传区域
    st.markdown("### 📄 文档管理")
    uploaded_files = st.file_uploader(
        "上传文档",
        type=['txt', 'pdf', 'docx', 'md'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files and st.button("📤 处理文档", key="process_docs"):
        process_documents(uploaded_files)
    
    # 系统信息
    st.markdown("---")
    st.markdown("### ℹ️ 系统信息")
    st.info(f"嵌入模型: text-embedding-ada-002:latest")
    st.info(f"LLM模型: gpt-3.5-turbo:latest")
    
    if st.session_state.vector_db:
        doc_count = len(st.session_state.vector_db.metadata)
        st.info(f"已索引文档: {doc_count} 条")


def main():
    """主应用界面"""
    st.title("🤖 Ollama RAG 问答系统")
    st.markdown("基于Ollama的text-embedding-ada-002和gpt-3.5-turbo的智能问答系统")
    
    if not st.session_state.initialized:
        st.warning("请等待系统初始化...")
        return
    
    # 主要功能区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 智能问答")
        
        # 查询输入
        query = st.text_area(
            "输入您的问题：",
            placeholder="例如：天龙八部中乔峰的身世是什么？",
            height=100,
            key="query_input"
        )
        
        # 搜索参数
        search_k = st.slider("检索相关文档数量", 1, 5, 3)
        
        if st.button("🔍 提问", key="ask_button") and query.strip():
            with st.spinner("正在生成回答..."):
                try:
                    # 生成查询嵌入
                    query_embedding = st.session_state.embedder.embed_query(query)
                    
                    # 检索相关文档
                    docs = st.session_state.vector_db.search(
                        query_embedding, k=search_k
                    )
                    
                    if docs:
                        # 生成回答
                        answer, references = st.session_state.rag_engine.generate_response(
                            query, docs
                        )
                        
                        # 显示结果
                        st.markdown("### 🎯 回答")
                        st.markdown(answer)
                        
                        # 显示引用
                        st.markdown("### 📚 参考文档")
                        for i, ref in enumerate(references, 1):
                            with st.expander(f"📄 {ref['source']} - 片段 {i}"):
                                st.text(ref['text'][:500] + "..." if len(ref['text']) > 500 else ref['text'])
                    else:
                        st.info("未找到相关文档，请上传更多文档")
                        
                except Exception as e:
                    st.error(f"生成回答时出错: {str(e)}")
                    logger.error(f"问答错误: {e}")
    
    with col2:
        st.markdown("### 📊 系统统计")
        
        if st.session_state.vector_db:
            doc_count = len(st.session_state.vector_db.metadata)
            st.metric("已索引文档", doc_count)
            
            if doc_count > 0:
                sources = set()
                for meta in st.session_state.vector_db.metadata:
                    sources.add(meta.get('source', '未知'))
                st.metric("文档来源", len(sources))
        
        # 快速操作
        st.markdown("### ⚡ 快速操作")
        
        if st.button("🗑️ 清空数据库", key="clear_db"):
            st.session_state.vector_db.clear()
            st.success("数据库已清空")
            st.rerun()
        
        if st.button("📈 查看统计", key="show_stats"):
            if st.session_state.vector_db:
                stats = st.session_state.vector_db.get_stats()
                st.json(stats)

if __name__ == "__main__":
    main()