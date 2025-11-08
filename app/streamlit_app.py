import streamlit as st
import os
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage

# Import the final, compiled graph
try:
    from core.graph import runnable_graph
except ImportError as e:
    st.error(f"Error importing graph. This is a critical error: {e}")
    st.info("Please ensure all files are correct and dependencies are installed.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the graph: {e}")
    st.stop()


# --- Page Config ---
st.set_page_config(page_title="RAG Q&A System")
st.title("RAG Q&A System (LangGraph) 🚀")

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
    st.warning("Ensure your vector stores are built. If not, run:\n`docker-compose exec app python core/vector_builder.py`")

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

    # --- Live Feedback section ---
    with st.chat_message("assistant"):
        thinking_container = st.empty()
        response_container = st.empty()
        
        thinking_container.markdown("🤔 Agent is thinking...")
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Prepare the input for the graph
        # We pass the new prompt as a HumanMessage
        graph_input = {"messages": [HumanMessage(content=prompt)]}
        
        final_answer = ""
        
        try:
            # Use .stream() to get live updates
            for chunk in runnable_graph.stream(
                graph_input, 
                config=config,
                stream_mode="values" # This mode gives us the full state
            ):
                # 'chunk' is the state of the graph *after* a node runs
                
                if "tool_calls" in chunk and chunk["tool_calls"]:
                    # This is the output of the Agent node
                    tools = [tc["name"] for tc in chunk["tool_calls"]]
                    thinking_container.markdown(f"🧠 Calling tool(s): **{', '.join(tools)}**")
                
                elif "tool_responses" in chunk and chunk["tool_responses"]:
                    # This is the output of the Tool Executor
                    thinking_container.markdown("✅ Tools finished. Agent is processing results...")
                
                elif "messages" in chunk:
                    # This is the final answer from the agent
                    last_message = chunk["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        final_answer = last_message.content

            # Write the final answer
            response_container.markdown(final_answer)
            thinking_container.empty()
            
            # Add the final answer to session history
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"An error occurred: {e}")
            thinking_container.empty()