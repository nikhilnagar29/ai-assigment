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


def extract_text_content(message_content):
    """
    Extract text content from message content, handling various formats:
    - Plain string
    - List of content blocks (e.g., [{'type': 'text', 'text': '...'}])
    - Other structured formats
    """
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, list):
        # Handle list of content blocks (e.g., from Gemini)
        text_parts = []
        for block in message_content:
            if isinstance(block, dict):
                if block.get('type') == 'text' and 'text' in block:
                    text_parts.append(block['text'])
                elif 'text' in block:
                    text_parts.append(block['text'])
            elif isinstance(block, str):
                text_parts.append(block)
        return '\n'.join(text_parts) if text_parts else ""
    else:
        # Fallback: convert to string
        return str(message_content) if message_content else ""


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
        graph_input = {
            "messages": [HumanMessage(content=prompt)],
            "iteration_count": 0
        }
        
        final_answer = ""
        tool_calls_made = []
        
        try:
            # Use .stream() to get live updates
            for chunk in runnable_graph.stream(
                graph_input, 
                config=config,
                stream_mode="values"
            ):
                # Handle tool calls
                if "messages" in chunk:
                    last_message = chunk["messages"][-1]
                    
                    # Check if this message has tool calls
                    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        tool_names = []
                        for tc in last_message.tool_calls:
                            if isinstance(tc, dict):
                                tool_names.append(tc.get("name", "unknown"))
                            else:
                                tool_names.append(getattr(tc, "name", "unknown"))
                        
                        if tool_names:
                            tool_calls_made.extend(tool_names)
                            thinking_container.markdown(f"🧠 Calling: **{', '.join(tool_names)}**")
                    
                    # Check if this is a ToolMessage (response from tools)
                    elif hasattr(last_message, '__class__') and last_message.__class__.__name__ == 'ToolMessage':
                        thinking_container.markdown("✅ Tools finished. Generating answer...")
                    
                    # Check if this is the final answer
                    elif isinstance(last_message, AIMessage):
                        # Extract text content
                        content = extract_text_content(last_message.content)
                        
                        # Only treat as final answer if it has actual text content
                        # (not just tool_calls with no text)
                        if content and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                            final_answer = content

            print(f"Final answer: {final_answer}")
            print(f"Tools called: {tool_calls_made}")

            # Write the final answer
            if final_answer:
                response_container.markdown(final_answer)
                thinking_container.empty()
                
                # Add the final answer to session history
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                # This shouldn't happen, but just in case
                error_msg = "⚠️ No response received from agent."
                if tool_calls_made:
                    error_msg += f" (Tools called: {', '.join(set(tool_calls_made))})"
                thinking_container.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            st.error(error_msg)
            thinking_container.empty()
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()