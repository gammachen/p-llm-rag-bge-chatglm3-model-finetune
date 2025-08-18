import streamlit as st
import os
import time
from document_loader import load_pdf, load_txt, chunk_text
from embedding_utils import BGEM3Embedder
from vector_db import FaissVectorDB
from rag_engine import ChatGLM3RAG

# 初始化组件
@st.cache_resource
def init_components():
    embedder = BGEM3Embedder()
    vector_db = FaissVectorDB()
    rag_engine = ChatGLM3RAG()
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
                    for file in uploaded_files:
                        # 保存文件
                        file_path = f"data/{file.name}"
                        os.makedirs("data", exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # 加载文档内容
                        if file.name.endswith(".pdf"):
                            text = load_pdf(file_path)
                        else:
                            text = load_txt(file_path)
                        
                        # 分块处理
                        chunks = chunk_text(text)
                        
                        # 生成向量
                        vectors = embedder.embed_texts(chunks)
                        
                        # 添加到向量数据库
                        metadatas = [{"source": file.name}] * len(chunks)
                        vector_db.add_documents(vectors, chunks, metadatas)
                
                st.success(f"成功处理 {len(uploaded_files)} 个文档!")
            else:
                st.warning("请先上传文档")
    
    # 问答界面
    st.header("文档问答")
    query = st.text_input("输入您的问题:")
    
    if st.button("提问") and query:
        with st.spinner("思考中..."):
            # 向量化问题
            query_vector = embedder.embed_query(query)
            
            # 检索相关文档
            context_docs = vector_db.search(query_vector, k=3)
            
            # 生成回答
            response, references = rag_engine.generate_response(query, context_docs)
            
            # 显示结果
            st.subheader("回答:")
            st.write(response)
            
            # 显示引用
            st.subheader("参考来源:")
            for ref in references:
                with st.expander(f"文档: {ref['source']} (相关度: {ref.get('score', 0):.2f})"):
                    st.write(ref['text'])

if __name__ == "__main__":
    main()