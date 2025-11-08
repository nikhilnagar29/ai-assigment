import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from core.config import (
    embeddings, 
    PRODUCT_VECTOR_STORE_PATH, 
    FEEDBACK_VECTOR_STORE_PATH
)
from core.tools.sql_tool import create_sql_agent_tool

# --- Helper Function to Load Stores ---
@st.cache_resource  # This caches the loaded stores in memory
def load_vector_stores():
    """
    Loads the FAISS vector stores from disk.
    Returns None if the stores are not found.
    """
    try:
        # Check if paths exist BEFORE trying to load
        if not os.path.exists(PRODUCT_VECTOR_STORE_PATH) or not os.path.exists(FEEDBACK_VECTOR_STORE_PATH):
            print("Vector store files not found. Build script needs to be run.")
            return None, None
        
        product_db = FAISS.load_local(
            PRODUCT_VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        feedback_db = FAISS.load_local(
            FEEDBACK_VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Successfully loaded both vector stores.")
        return product_db, feedback_db
    except Exception as e:
        print(f"Error loading vector stores: {e}")
        return None, None

# --- Helper Function to Load SQL Agent ---
@st.cache_resource
def get_sql_agent():
    """Creates and caches the SQL agent."""
    try:
        return create_sql_agent_tool()
    except Exception as e:
        st.error(f"Error creating SQL Agent: {e}")
        return None

# --- Page Config ---
st.set_page_config(page_title="RAG Q&A System")
st.title("RAG Q&A System (FAISS Build) 🚀")

# --- Sidebar for System Status ---
with st.sidebar:
    st.subheader("System Status")
    
    if os.environ.get("GOOGLE_API_KEY"):
        st.success("Google API Key loaded.")
    else:
        st.error("Google API Key not found.")
        
    product_db, feedback_db = load_vector_stores()
    
    if product_db and feedback_db:
        st.success("Product Vector Store loaded.")
        st.success("Feedback Vector Store loaded.")
        # Save to session state for the app to use
        st.session_state.product_db = product_db
        st.session_state.feedback_db = feedback_db
    else:
        st.error("Vector Stores NOT FOUND.")
        st.warning("Please run the one-time build script in your terminal:")
        st.code("docker-compose exec app python core/vector_builder.py")
    
    # Load the SQL Agent
    sql_agent = get_sql_agent()
    if sql_agent:
        st.success("SQL Agent is ready.")
        st.session_state.sql_agent = sql_agent
    else:
        st.error("SQL Agent failed to initialize.")

# --- Main Chat Interface ---
st.subheader("Chat Interface")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "product_db" not in st.session_state:
        st.error("Vector stores are not loaded. Cannot process query. Run the build script.")
    else:
        response = f"Placeholder: FAISS is ready. You asked '{prompt}'"
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})