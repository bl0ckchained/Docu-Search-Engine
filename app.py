# The SQLite fix must be the absolute first thing in the file
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import traceback
import os

st.title("Enterprise IT Asset Assistant")

try:
    # Late imports prevent the server from crashing before boot
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    
    # Manually bridge the API key from Streamlit to the OS environment
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "GOOGLE_API_KEY" not in os.environ:
        st.error("API Key not found in Streamlit Secrets. Please check your advanced settings.")
        st.stop()

    @st.cache_resource
    def load_and_process_data():
        loader = PyPDFDirectoryLoader("data")
        docs = loader.load()
        if not docs:
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vectorstore

    with st.spinner("Loading ITAM Knowledge Base... This may take a minute."):
        vectorstore = load_and_process_data()
        
    if vectorstore is None:
        st.error("No PDFs found in the data folder. Check your GitHub repository.")
        st.stop()

    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    system_prompt = (
        "You are an IT Asset Management assistant. "
        "Use the provided context from the uploaded manuals to answer the question. "
        "If you do not know the answer based on the context, say you do not know. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    user_input = st.chat_input("Ask a question about IT Asset Management:")

    if user_input:
        st.chat_message("user").write(user_input)
        response = rag_chain.invoke({"input": user_input})
        st.chat_message("assistant").write(response["answer"])

except Exception as e:
    st.error("An error occurred during startup. Please see the details below.")
    st.code(traceback.format_exc())
