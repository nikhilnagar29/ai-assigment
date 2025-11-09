import os
import json
from typing import List, TypedDict, Annotated, Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- New Imports for direct SQL tools ---
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from core.config import llm, DB_URL_SQL_AGENT
from core.tools.feedback_tool import create_feedback_rag_tool
from core.tools.product_tool import create_product_rag_tool

# --- 1. Define the Tools ---

# --- SQL Tools with Enhanced Descriptions ---
print("\n" + "=" * 60)
print("ATTEMPTING TO CONNECT TO SQL DATABASE")
print("=" * 60)

sql_tools = []
try:
    # First, try to import required libraries
    print("Step 1: Checking required libraries...")
    import psycopg2
    print("✅ psycopg2 is installed")
    
    print("\nStep 2: Attempting database connection...")
    print(f"Connection URL: {DB_URL_SQL_AGENT}")
    
    # Try to create the database connection
    # Note: PostgreSQL table names are case-sensitive. Your schema uses lowercase.
    db = SQLDatabase.from_uri(
        DB_URL_SQL_AGENT,
        include_tables=["artist", "album", "track", "customer", "employee", "invoice", "invoice_line"]
    )
    
    print("✅ Database connection successful!")
    
    print("\nStep 3: Testing database access...")
    # Try to list tables to verify connection works
    tables = db.get_usable_table_names()
    print(f"✅ Found {len(tables)} tables: {tables}")
    
    print("\nStep 4: Creating SQL toolkit...")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    raw_sql_tools = toolkit.get_tools()
    print(f"✅ Created {len(raw_sql_tools)} SQL tools")
    
    # Wrap SQL tools with better descriptions
    print("\nStep 5: Enhancing tool descriptions...")
    for tool in raw_sql_tools:
        print(f"   - {tool.name}")
        if tool.name == "sql_db_list_tables":
            tool.description = (
                "Use this FIRST for any business/sales question. Lists all available database tables "
                "(customer, invoice, artist, album, track, employee, etc.). "
                "Use for questions about: sales, customers, invoices, artists, albums, employees."
            )
        elif tool.name == "sql_db_schema":
            tool.description = (
                "Get the schema/structure of database tables. Use this AFTER sql_db_list_tables "
                "to understand table columns before querying. Input: comma-separated table names."
            )
        elif tool.name == "sql_db_query":
            tool.description = (
                "Execute SQL queries on the business database. Use for questions about: "
                "customer count, sales data, invoice totals, artist/album sales, employee info. "
                "Input: a valid SQL SELECT query."
            )
        elif tool.name == "sql_db_query_checker":
            tool.description = (
                "Validates SQL queries before execution. Use this to check if your SQL is correct."
            )
        sql_tools.append(tool)
    
    print(f"\n✅ SQL DATABASE CONNECTION SUCCESSFUL!")
    print(f"✅ Loaded {len(sql_tools)} SQL tools")
    
except ImportError as e:
    print(f"\n❌ ERROR: Missing required library!")
    print(f"Error: {e}")
    print("\nTo fix this, run:")
    print("   pip install psycopg2-binary")
    print("   or")
    print("   apt-get install python3-psycopg2")
    
except Exception as e:
    print(f"\n❌ ERROR: Could not connect to database!")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    print("\nPossible causes:")
    print("1. Database is not running")
    print("   Fix: docker-compose up -d db")
    print("2. Wrong credentials in .env file")
    print("   Fix: Check DB_USER, DB_PASSWORD, DB_NAME")
    print("3. Wrong host/port")
    print("   Fix: Check DB_HOST and DB_PORT")
    print("4. Database 'chinook' does not exist")
    print("   Fix: Create the database first")
    print("\nSQL tools will NOT be available until database is connected.")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("=" * 60 + "\n")

# --- RAG Tools ---
try:
    feedback_tool = create_feedback_rag_tool()
    product_tool = create_product_rag_tool()
    rag_tools = [feedback_tool, product_tool]
except Exception as e:
    print(f"Warning: Could not create RAG tools: {e}")
    rag_tools = []

# --- Full Tool List ---
tools = sql_tools + rag_tools

if not tools:
    raise RuntimeError("No tools available! Please check database connection and vector stores.")

# Debug: Print all tool names and descriptions
print(f"\n✅ Loaded {len(tools)} tools:")
for tool in tools:
    print(f"   - {tool.name}: {tool.description[:80]}...")
print()


# --- 2. Define the Graph State ---
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # Accumulate all messages
    iteration_count: int  # Track iterations to prevent infinite loops


# --- 3. Define the Agent Node ---
def agent_node(state: AgentState):
    """
    The main agent that decides which tools to call or provides a final answer.
    """
    print("\n--- AGENT NODE ---")
    
    # Check iteration count to prevent infinite loops
    iteration_count = state.get('iteration_count', 0)
    MAX_ITERATIONS = 5
    
    if iteration_count >= MAX_ITERATIONS:
        print(f"⚠️ Maximum iterations ({MAX_ITERATIONS}) reached. Forcing final answer.")
        final_response = AIMessage(
            content="I apologize, but I'm having difficulty processing your request. Please try rephrasing your question."
        )
        return {"messages": [final_response], "iteration_count": iteration_count}
    
    messages = state.get('messages', [])
    
    # Enhanced system prompt that explicitly instructs when to use tools
    system_prompt = """You are a helpful assistant for BMW with access to multiple data sources.

**YOUR TOOLS:**

1. **BMW iX Product & Feedback Tools:**
   - customer_feedback_search: Customer opinions, complaints, feedback about BMW iX
   - product_details_search: Technical specs, features, details about BMW iX

2. **Business Database (SQL) Tools - Use for business/sales questions:**
   - sql_db_list_tables: **START HERE** for any sales/business question - lists available tables
   - sql_db_schema: Get table structure/columns (use after listing tables)
   - sql_db_query: Execute SQL queries for sales data, customer counts, invoices, etc.

**DECISION RULES:**

🚗 **BMW iX questions** → Use product_details_search and/or customer_feedback_search
   - "What's the charging time?"
   - "What do people think about steering?"
   - "Tell me about BMW iX features"

💼 **Business/Sales questions** → ALWAYS start with sql_db_list_tables
   - "How many customers?" → sql_db_list_tables
   - "What are total sales?" → sql_db_list_tables
   - "Show me invoices" → sql_db_list_tables
   - "Artists with most sales" → sql_db_list_tables

**CRITICAL:**
- For ANY question about numbers, counts, sales, invoices, artists, albums → Use SQL tools!
- ALWAYS call sql_db_list_tables FIRST for business questions
- After getting tool results, provide a clear, synthesized answer
- Assume "the car" = BMW iX
"""
    
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    # ALWAYS bind tools to allow the agent to use them
    agent_llm = llm.bind_tools(tools)
    agent_chain = agent_prompt | agent_llm
    
    # Call the agent
    response_message = agent_chain.invoke({"messages": messages})
    
    print(f"Agent response type: {type(response_message)}")
    print(f"Has tool_calls: {hasattr(response_message, 'tool_calls') and bool(response_message.tool_calls)}")
    
    # Check if the agent wants to use tools
    if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
        tool_names = [tc.get('name') if isinstance(tc, dict) else tc.name for tc in response_message.tool_calls]
        print(f"🔧 Agent calling tools: {tool_names} (iteration {iteration_count + 1})")
        
        return {
            "messages": [response_message],
            "iteration_count": iteration_count + 1
        }
    
    # No tool calls - this is the final answer
    print("✅ Agent provided final answer")
    return {
        "messages": [response_message],
        "iteration_count": iteration_count
    }


# --- 4. Define the Tool Executor Node ---
def tool_executor_node(state: AgentState):
    """
    Executes the tools requested by the agent.
    """
    print("\n--- TOOL EXECUTOR NODE ---")
    
    messages = state.get('messages', [])
    last_message = messages[-1]
    
    # Create a map of tool names to tool objects
    tool_map = {tool.name: tool for tool in tools}
    
    tool_responses = []
    
    # Execute each tool call
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else tool_call.args
        tool_call_id = tool_call.get("id", "unknown") if isinstance(tool_call, dict) else tool_call.id
        
        print(f"🔧 Executing: {tool_name}")
        print(f"   Args: {tool_args}")
        
        tool_to_call = tool_map.get(tool_name)
        
        if not tool_to_call:
            error_msg = f"Tool '{tool_name}' not found. Available: {list(tool_map.keys())}"
            print(f"❌ {error_msg}")
            tool_responses.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=error_msg,
                    name=tool_name
                )
            )
            continue
        
        # Extract the query/input from tool_args
        query = None
        if isinstance(tool_args, dict):
            # Try common parameter names
            query = (tool_args.get('query') or 
                    tool_args.get('arg1') or 
                    tool_args.get('input') or 
                    tool_args.get('question') or
                    tool_args.get('tool_input'))
            
            # If still no query, get first string value
            if not query:
                for value in tool_args.values():
                    if isinstance(value, str):
                        query = value
                        break
        elif isinstance(tool_args, str):
            query = tool_args
        
        if not query:
            query = str(tool_args)
        
        print(f"   Query: '{query[:100]}...'")
        
        try:
            # Execute the tool
            response = tool_to_call.invoke(query)
            response_text = str(response)
            
            # Truncate long responses
            if len(response_text) > 2000:
                response_text = response_text[:2000] + "...(truncated)"
            
            print(f"✅ Tool '{tool_name}' succeeded. Response length: {len(str(response))}")
            
            tool_responses.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=response_text,
                    name=tool_name
                )
            )
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            tool_responses.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=error_msg,
                    name=tool_name
                )
            )
    
    return {"messages": tool_responses}


# --- 5. Define Conditional Logic ---
def should_continue(state: AgentState):
    """
    Determines whether to continue calling tools or end.
    """
    messages = state.get('messages', [])
    last_message = messages[-1] if messages else None
    
    # If the last message has tool calls, continue to tool execution
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("→ Routing to tools")
        return "continue"
    
    # Otherwise, we're done
    print("→ Routing to END")
    return "end"


# --- 6. Assemble the Graph ---
def create_graph():
    """
    Assembles the graph with proper routing.
    """
    print("\n=== Creating LangGraph ===")
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_executor_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional routing from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # After tools execute, always return to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    graph = workflow.compile(checkpointer=MemorySaver())
    
    print("✅ LangGraph compiled successfully\n")
    return graph


# Create the runnable graph
runnable_graph = create_graph()