import streamlit as st
import os
from uuid import uuid4

# Import the final, compiled graph from our new file
try:
    from core.graph import runnable_graph
except ImportError as e:
    st.error(f"Error importing graph. This is a critical error. {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the graph: {e}")
    st.stop()


# --- Page Config ---
st.set_page_config(page_title="RAG Q&A System")
st.title("RAG Q&A System (Full Agent) 🚀")

# --- Sidebar ---
with st.sidebar:
    st.subheader("System Status")
    if os.environ.get("GOOGLE_API_KEY"):
        st.success("Google API Key loaded.")
    else:
        st.error("Google API Key not found.")
    
    st.info(
        "This app uses a LangGraph agent to route questions to:\n"
        "1. A SQL Database (Chinook)\n"
        "2. A Product PDF Vector Store (FAISS)\n"
        "3. A Feedback TXT Vector Store (FAISS)"
    )

# --- Session Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Input ---
if prompt := st.chat_input("Ask a multi-part question..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- This is the "Live Feedback" section ---
    with st.chat_message("assistant"):
        # Create a container for the "thinking" steps
        thinking_container = st.empty()
        thinking_container.markdown("🤔 Agent is thinking...")
        
        # Create a container for the final answer
        response_container = st.empty()
        
        # This config connects our chat to the graph's memory
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # --- Run the Graph! ---
        final_answer = ""
        
        # st.write_stream will process the output from graph.stream()
        # graph.stream() yields the output of *each node* as it runs
        for chunk in runnable_graph.stream(
            {"input": prompt, "chat_history": st.session_state.messages}, 
            config=config,
            stream_mode="values"
        ):
            # 'chunk' is the state of the graph *after* a node runs
            if "messages" in chunk:
                # This is the final answer from the generator
                final_answer = chunk["messages"][-1].content
            elif "tool_calls" in chunk and chunk["tool_calls"]:
                # This is the output of the Router
                tools = [tc.name for tc in chunk["tool_calls"]]
                thinking_container.markdown(f"🧠 Router decided to use: **{', '.join(tools)}**")
            elif "tool_responses" in chunk and chunk["tool_responses"]:
                # This is the output of the Tool Node
                thinking_container.markdown("✅ Tools finished running. Generating final answer...")

        # Write the final answer to its container
        response_container.markdown(final_answer)
        
        # Clear the "thinking" message
        thinking_container.empty()
        
    # Add the final answer to session history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})