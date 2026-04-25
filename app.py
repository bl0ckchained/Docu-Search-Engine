import streamlit as st
import traceback
import os

st.title("Enterprise IT Asset Assistant")

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    
    # The Fix: We now import from langchain_classic instead of langchain
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    
    from langchain_core.prompts import ChatPromptTemplate
    
    # Securely bridge the API key
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "GOOGLE_API_KEY" not in os.environ:
        st.error("API Key not found in Streamlit Secrets.")
        st.stop()

    @st.cache_resource
    def load_and_process_data():
        # Prevent crash if the data folder is completely missing
        if not os.path.exists("data"):
            os.makedirs("data")
            return None
            
        loader = PyPDFDirectoryLoader("data")
        docs = loader.load()
        if not docs:
            return None
            
        # Reduced chunk size to protect cloud memory limits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore

    # Control the boot sequence
    if "db_ready" not in st.session_state:
        st.session_state.db_ready = False

    # The UI boots FIRST, then waits for your command
    if not st.session_state.db_ready:
        st.info("System Online. Standing by to compile IT manuals into the vector database.")
        if st.button("Initialize Knowledge Base"):
            with st.spinner("Processing dense PDFs... this will take a moment."):
                vectorstore = load_and_process_data()
                if vectorstore is not None:
                    st.session_state.db_ready = True
                    st.rerun()
                else:
                    st.error("No PDFs found in the 'data' folder on GitHub.")

    # Only load the chat interface AFTER the database is built
    if st.session_state.db_ready:
        st.success("Knowledge Base is fully operational.")
        vectorstore = load_and_process_data()
        
        # Limit the AI to retrieving the top 3 chunks to save RAM during the chat
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
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
    st.error("A system error occurred. Please see the technical details below.")
    st.code(traceback.format_exc())
