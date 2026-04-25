import streamlit as st
import traceback
import os
import hashlib
import datetime
import json

st.title("Enterprise IT Asset Assistant")

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    
    # Securely bridge the API key
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "GOOGLE_API_KEY" not in os.environ:
        st.error("API Key not found in Streamlit Secrets.")
        st.stop()

    # --- FEATURE 3: Cryptographic Audit Ledger ---
    def log_interaction(user_query, ai_response):
        log_file = "audit_ledger.json"
        
        # Create the data payload
        entry_data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": user_query,
            "response_snippet": ai_response[:150] + "..." # Save space by logging the snippet
        }
        
        # Load existing logs to get the previous hash (Blockchain style)
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try:
                    logs = json.load(f)
                    prev_hash = logs[-1]["hash"] if logs else "00000000"
                except:
                    logs, prev_hash = [], "00000000"
        else:
            logs, prev_hash = [], "00000000"

        # Cryptographically hash the current entry + previous hash
        data_string = json.dumps(entry_data, sort_keys=True) + prev_hash
        entry_hash = hashlib.sha256(data_string.encode()).hexdigest()
        
        # Finalize entry
        entry_data["prev_hash"] = prev_hash
        entry_data["hash"] = entry_hash
        
        logs.append(entry_data)
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)

    @st.cache_resource
    def load_and_process_data():
        if not os.path.exists("data"):
            os.makedirs("data")
            return None
            
        loader = PyPDFDirectoryLoader("data")
        docs = loader.load()
        if not docs:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
        splits = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore

    # Control the boot sequence
    if "db_ready" not in st.session_state:
        st.session_state.db_ready = False

    # --- FEATURES 1 & 2: Sidebar Control Panel & Dynamic Routing ---
    st.sidebar.title("⚙️ Enterprise Controls")
    st.sidebar.write("Configure the active intelligence protocol.")
    
    agent_mode = st.sidebar.selectbox(
        "Select Agent Specialization:",
        ["IT Asset Management", "Cybersecurity Compliance", "Network Telemetry mapping"]
    )
    
    st.sidebar.divider()
    
    # Button to view the immutable ledger
    if st.sidebar.button("View Audit Ledger"):
        if os.path.exists("audit_ledger.json"):
            with open("audit_ledger.json", "r") as f:
                st.sidebar.json(json.load(f))
        else:
            st.sidebar.warning("Ledger is currently empty.")

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

    if st.session_state.db_ready:
        st.success("Knowledge Base is fully operational.")
        vectorstore = load_and_process_data()
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Dynamically change the AI's brain based on the sidebar selection
        if agent_mode == "Cybersecurity Compliance":
            system_prompt = (
                "You are a strict Enterprise Cybersecurity Auditor. "
                "Use the provided context to answer the question, but explicitly cross-reference your answer "
                "with standard security frameworks (like NIST or ISO 27001). Warn the user of any potential security risks. "
                "Context: {context}"
            )
        elif agent_mode == "Network Telemetry mapping":
            system_prompt = (
                "You are an Enterprise Network Architect. "
                "Use the provided context to answer the question. Focus heavily on subnets, bandwidth implications, "
                "telecommunications protocols, and physical hardware deployment strategy. "
                "Context: {context}"
            )
        else:
            system_prompt = (
                "You are an IT Asset Management assistant. "
                "Use the provided context from the uploaded manuals to answer the question clearly and concisely. "
                "If you do not know the answer based on the context, say you do not know. "
                "Context: {context}"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        user_input = st.chat_input("Ask a question about your Enterprise infrastructure:")

        if user_input:
            st.chat_message("user").write(user_input)
            response = rag_chain.invoke({"input": user_input})
            st.chat_message("assistant").write(response["answer"])
            
            # Write this interaction to the secure log
            log_interaction(user_input, response["answer"])

except Exception as e:
    st.error("A system error occurred. Please see the technical details below.")
    st.code(traceback.format_exc())
