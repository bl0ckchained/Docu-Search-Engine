__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

st.title("Enterprise IT Asset Assistant")

# Cache the database creation so it only runs once
@st.cache_resource
def load_and_process_data():
    # 1. Load the PDFs from your data folder
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    
    # 2. Chunk the text into readable pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Create the ChromaDB vector database using Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

with st.spinner("Loading ITAM Knowledge Base..."):
    vectorstore = load_and_process_data()

# 4. Set up the Retriever and the Gemini Brain
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 5. Give the AI its instructions
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

# 6. Build the chain that connects the database to the chat
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 7. Create the Streamlit Chat Interface
user_input = st.chat_input("Ask a question about IT Asset Management:")

if user_input:
    st.chat_message("user").write(user_input)
    response = rag_chain.invoke({"input": user_input})
    st.chat_message("assistant").write(response["answer"])
