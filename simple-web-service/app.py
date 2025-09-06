import streamlit as st
import os
import time
import logging
import psutil
import gc
import numpy as np
from document_loader import load_pdf, load_txt, chunk_text
from embedding_utils import BGEM3Embedder
from vector_db import FaissVectorDB
from rag_engine import ChatGLM3RAG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# 内存监控工具
def log_memory_usage(operation=""):
    """记录内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"💾 [{operation}] 内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    return memory_info.rss / 1024 / 1024

# 初始化全局变量
@st.cache_resource
def init_components():
    logger.info("🚀 开始初始化系统...")
    start_time = time.time()
    
    log_memory_usage("初始化前")
    
    embedder = BGEM3Embedder()
    vector_db = FaissVectorDB()
    rag_engine = ChatGLM3RAG()
    
    elapsed_time = time.time() - start_time
    memory_used = log_memory_usage("初始化后")
    
    logger.info(f"✅ 系统初始化完成，耗时: {elapsed_time:.2f}秒，内存使用: {memory_used:.2f}MB")
    return embedder, vector_db, rag_engine

# 主应用
def main():
    st.title("智能文档问答系统 (RAG)")
    st.markdown("上传文档后，即可基于文档内容进行问答")
    
    # 初始化组件
    embedder, vector_db, rag_engine = init_components()
    
    # 文档上传与处理
    with st.sidebar:
        st.header("文档管理")
        uploaded_files = st.file_uploader(
            "上传PDF或TXT文档",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("处理文档"):
            if uploaded_files:
                with st.spinner("处理文档中..."):
                    total_files = len(uploaded_files)
                    logger.info(f"📁 开始处理 {total_files} 个文件")
                    
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, file in enumerate(uploaded_files):
                        logger.info(f"📄 处理文件 {idx+1}/{total_files}: {file.name}")
                        file_start_time = time.time()
                        
                        # 内存监控 - 文件处理前
                        mem_before_file = log_memory_usage(f"处理文件 {file.name} 前")
                        
                        # 保存文件
                        file_path = f"data/{file.name}"
                        os.makedirs("data", exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        logger.info(f"💾 文件已保存: {file_path}")
                        
                        try:
                            # 加载文档内容
                            logger.info("📖 开始加载文档内容...")
                            load_start = time.time()
                            if file.name.endswith(".pdf"):
                                text = load_pdf(file_path)
                            else:
                                text = load_txt(file_path)
                            load_time = time.time() - load_start
                            logger.info(f"✅ 文档加载完成，耗时: {load_time:.2f}秒，文本长度: {len(text)}字符")
                            
                            # 分块处理
                            logger.info("✂️ 开始文本分块...")
                            chunk_start = time.time()
                            chunks = chunk_text(text, chunk_size=512, overlap=50)
                            chunk_time = time.time() - chunk_start
                            logger.info(f"✅ 分块完成，耗时: {chunk_time:.2f}秒，共{len(chunks)}个块")
                            
                            # 内存监控 - 分块后
                            mem_after_chunk = log_memory_usage(f"分块后 ({file.name})")
                            
                            # 生成向量
                            logger.info("🧮 开始生成向量...")
                            vector_start = time.time()
                            vectors = embedder.embed_texts(chunks)
                            vector_time = time.time() - vector_start
                            logger.info(f"✅ 向量生成完成，耗时: {vector_time:.2f}秒，向量形状: {np.array(vectors).shape}")
                            
                            # 内存监控 - 向量生成后
                            mem_after_vector = log_memory_usage(f"向量生成后 ({file.name})")
                            
                            # 添加到向量数据库
                            logger.info("💾 开始添加到向量数据库...")
                            db_start = time.time()
                            metadatas = [{"source": file.name}] * len(chunks)
                            vector_db.add_documents(vectors, chunks, metadatas)
                            db_time = time.time() - db_start
                            logger.info(f"✅ 向量数据库添加完成，耗时: {db_time:.2f}秒")
                            
                            # 垃圾回收
                            gc.collect()
                            log_memory_usage(f"垃圾回收后 ({file.name})")
                            
                            file_total_time = time.time() - file_start_time
                            logger.info(f"🎉 文件 {file.name} 处理完成，总耗时: {file_total_time:.2f}秒")
                            
                        except Exception as e:
                            logger.error(f"❌ 处理文件 {file.name} 时出错: {e}")
                            raise
                        
                        # 进度更新
                        progress_bar.progress((idx + 1) / total_files)
                        status_text.text(f"处理完成: {file.name}")
                    
                    # 最终内存统计
                    log_memory_usage("所有文件处理完成")
                    logger.info(f"✅ 所有文件处理完成，共处理 {total_files} 个文件")
                
                st.success(f"成功处理 {len(uploaded_files)} 个文档!")
            else:
                st.warning("请先上传文档")
    
    # 问答界面
    st.header("文档问答")
    query = st.text_input("输入您的问题:")
    
    if st.button("提问") and query:
        logger.info(f"🤔 用户提问: {query}")
        
        with st.spinner("思考中..."):
            # 内存监控 - 提问前
            mem_before_query = log_memory_usage("提问前")
            
            # 向量化问题
            logger.info("🧮 开始向量化问题...")
            query_vector = embedder.embed_query(query)
            logger.info("✅ 问题向量化完成")
            
            # 检索相关文档（减少数量以控制上下文长度）
            logger.info("🔍 开始检索相关文档...")
            search_start = time.time()
            context_docs = vector_db.search(query_vector, k=2)  # 减少到2个文档
            search_time = time.time() - search_start
            logger.info(f"🔎 检索完成，耗时: {search_time:.2f}秒，找到 {len(context_docs)} 个相关文档")
            
            # 内存监控 - 检索后
            mem_after_search = log_memory_usage("检索后")
            
            # 生成回答
            logger.info("🤖 开始生成回答...")
            response_start = time.time()
            response, references = rag_engine.generate_response(query, context_docs)
            response_time = time.time() - response_start
            response_length = len(response) if response else 0
            logger.info(f"✅ 回答生成完成，耗时: {response_time:.2f}秒，长度: {response_length}字符")
            
            # 内存监控 - 回答生成后
            mem_after_response = log_memory_usage("回答生成后")
            
            # 显示结果
            st.subheader("回答:")
            st.write(response)
            
            # 显示引用
            st.subheader("参考来源:")
            for ref in references:
                with st.expander(f"文档: {ref['source']} (相关度: {ref.get('score', 0):.2f})"):
                    st.write(ref['text'])
            
            # 垃圾回收
            gc.collect()
            log_memory_usage("垃圾回收后")

if __name__ == "__main__":
    main()