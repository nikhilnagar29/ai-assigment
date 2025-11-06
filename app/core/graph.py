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

# --- SQL Tools ---
# Connect to the DB (with error handling)
try:
    db = SQLDatabase.from_uri(
        DB_URL_SQL_AGENT,
        include_tables=["Artist", "Album", "Track", "Customer", "Employee", "Invoice", "InvoiceLine"]
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()  # returns ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDatabaseTool, etc.
    print("Connected to SQL Database for Agent.")
except Exception as e:
    print(f"Warning: Could not connect to database: {e}")
    print("SQL tools will not be available. Make sure the database is running.")
    sql_tools = []

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


# --- 2. Define the Graph State ---
class AgentState(TypedDict):
    input: str
    chat_history: list
    messages: Annotated[list, lambda x, y: x + y]
    tool_calls: list
    tool_responses: Annotated[list, lambda x, y: x + y]  # Accumulate tool responses
    iteration_count: int  # Track iterations to prevent infinite loops


# --- 3. Define the Graph Nodes ---

# --- NODE 1: The ROUTER ---
class ToolRouter(BaseModel):
    """A router to select the appropriate tool(s) for a user query."""
    tool_names: List[Literal[
        "sql_db_list_tables", 
        "sql_db_schema", 
        "sql_db_query", 
        "customer_feedback_search", 
        "product_details_search"
    ]] = Field(
        ..., 
        description="A list of tool names to call. Only use the tools provided."
    )

structured_llm_router = llm.with_structured_output(ToolRouter)

def router_node(state: AgentState):
    """Analyzes the user's query and decides which tools to call."""
    print("--- CALLING ROUTER NODE ---")
    
    # We build a prompt from the last human message
    # This ensures the router only focuses on the *newest* question
    last_message = state['messages'][-1].content
    
    prompt = f"""
    You are an expert router. Your job is to analyze the user's query and 
    determine which tool(s) are needed to answer it.

    The available tools are:
    1. sql_db_list_tables: Use this to see what tables are in the database.
    2. sql_db_schema: Use this to see the schema of a specific table.
    3. sql_db_query: Use this to run a SQL query on the database.
    4. customer_feedback_search: For questions about customer opinions, feedback, or complaints (about BMW iX).
    5. product_details_search: For questions about BMW iX product features or technical specs.

    User Query: "{last_message}"

    Respond with a list of the *exact* tool names required.
    
    *Examples:*
    - "How many customers?" -> ["sql_db_list_tables"] (to see tables first)
    - "What's the schema for the Invoice table?" -> ["sql_db_schema"]
    - "What do people think of the steering wheel?" -> ["customer_feedback_search"]
    - "What is the charging time and are there any complaints about it?" -> ["product_details_search", "customer_feedback_search"]
    - "Show me sales for AC/DC and any feedback for them" -> ["sql_db_list_tables"] (then the agent will use schema and query tools)
    
    *IMPORTANT*: For any SQL-related question, the agent's first step should be to list the tables.
    So, if the query mentions 'sales', 'artists', 'invoices', etc., ALWAYS include 'sql_db_list_tables' in your response.
    """
    
    router_output = structured_llm_router.invoke([HumanMessage(content=prompt)])
    
    tool_calls = []
    if router_output.tool_names:
        # The agent will decide the input for these tools in the next step
        for tool_name in router_output.tool_names:
            tool_calls.append(
                ToolMessage(
                    tool_call_id=f"call_{tool_name}", 
                    content=last_message, # Pass the original query as context
                    name=tool_name
                )
            )
    
    print(f"Router decided to call: {router_output.tool_names}")
    return {"tool_calls": tool_calls, "messages": []} # Clear messages to avoid loop


# --- NODE 2: The Agent Node (Tool Caller) ---
# This node uses the main LLM to decide *how* to call the tools
# --- NODE 2: The Agent Node (Tool Caller) ---
# This node uses the main LLM to decide *how* to call the tools
def agent_node(state: AgentState):
    """
    The main "worker" node. It processes tool results and either calls more tools
    or provides a final answer.
    """
    print("--- CALLING AGENT/TOOL NODE ---")
    
    # Check iteration count to prevent infinite loops
    iteration_count = state.get('iteration_count', 0)
    MAX_ITERATIONS = 5  # Maximum number of tool call iterations
    
    if iteration_count >= MAX_ITERATIONS:
        print(f"⚠️ Maximum iterations ({MAX_ITERATIONS}) reached. Forcing final answer.")
        # Force a final answer
        final_response = AIMessage(
            content="I apologize, but I'm having difficulty processing your request. Please try rephrasing your question or breaking it into smaller parts."
        )
        return {"messages": [final_response], "tool_calls": []}
    
    # Build the conversation context
    messages = state.get('messages', [])
    tool_responses = state.get('tool_responses', [])
    
    # Check if we have tool responses
    has_tool_responses = len(tool_responses) > 0
    
    # Combine messages and tool responses for context
    chat_history = messages + tool_responses

    # --- THIS IS THE NEW LOGIC ---
    if has_tool_responses:
        # We have tool results. Our ONLY job is to synthesize an answer.
        # We do NOT bind tools, so the LLM is *forced* to answer.
        print("Agent is in 'Answering Mode'. Tools are disabled.")
        system_prompt = """You are a helpful assistant for BMW. Your job is to answer the user's question using the available tools.

Available tools:
- customer_feedback_search: Use this for any question about customer feedback, opinions, or complaints.
- product_details_search: Use this for any question about product features or technical specs.
- SQL tools: Query the database for sales, customers, invoices, etc.

IMPORTANT Instructions:
1. **You MUST assume the user is asking about the BMW iX** if they ask about "the car", "BMW car", "bmq car", or any other general car question.
2. ALWAYS try to use 'customer_feedback_search' or 'product_details_search' for any question about feedback, features, specs, or the car.
3. Call the appropriate tool(s) to get information.
"""
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history")
        ])
        
        # Note: We use the base 'llm', NOT 'llm.bind_tools(tools)'
        agent_chain = agent_prompt | llm 
        
    else:
        # This is the FIRST run. We need to call tools.
        print("Agent is in 'Tool-Calling Mode'.")
        system_prompt = """You are a helpful assistant for BMW. Use the available tools to answer the user's question.

Available tools:
- customer_feedback_search: Search for customer feedback about BMW iX
- product_details_search: Find product specifications and features
- SQL tools: Query the database for sales, customers, invoices, etc.

Instructions:
1. Call the appropriate tool(s) to get information.
"""
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history")
        ])
        
        # We bind the tools to the LLM
        agent_llm = llm.bind_tools(tools)
        agent_chain = agent_prompt | agent_llm
    # --- END OF NEW LOGIC ---

    
    # Call the agent
    response_message = agent_chain.invoke({
        "chat_history": chat_history
    })
    
    # If the LLM responded with tool calls, we return them (but increment iteration count)
    if response_message.tool_calls:
        # This should now only happen if 'has_tool_responses' was False
        tool_names = [tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in response_message.tool_calls]
        print(f"Agent decided to call tools: {tool_names} (iteration {iteration_count + 1})")
        
        return {
            "tool_calls": response_message.tool_calls,
            "iteration_count": iteration_count + 1
        }
    
    # If the LLM responded with a final answer, we return it
    print("Agent provided a final answer.")
    return {"messages": [response_message], "tool_calls": []}




# --- NODE 3: The Tool Executor Node ---
def tool_executor_node(state: AgentState):
    """
    This node *only* executes tools.
    """
    print("--- CALLING TOOL EXECUTOR NODE ---")
    tool_map = {tool.name: tool for tool in tools}
    new_tool_responses = []  # New responses from this execution
    
    for tool_call in state["tool_calls"]:
        # Handle both dict and object formats for tool_calls
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "unknown")
        else:
            # LangChain ToolCall object
            tool_name = getattr(tool_call, "name", None)
            tool_args = getattr(tool_call, "args", {})
            tool_call_id = getattr(tool_call, "id", "unknown")
        
        if not tool_name:
            print(f"Warning: Skipping tool call with no name: {tool_call}")
            continue
            
        tool_to_call = tool_map.get(tool_name)
        
        if tool_to_call:
            # Extract query string from tool_args
            # Tools expect a string, but agent might pass dict like {'arg1': 'query'} or {'query': 'text'}
            query_string = None
            if isinstance(tool_args, dict):
                # Try common keys
                query_string = tool_args.get('query') or tool_args.get('arg1') or tool_args.get('input') or tool_args.get('question')
                # If still None, try to get the first string value
                if not query_string:
                    for key, value in tool_args.items():
                        if isinstance(value, str):
                            query_string = value
                            break
                # If still None, convert dict to string
                if not query_string:
                    query_string = str(tool_args)
            elif isinstance(tool_args, str):
                query_string = tool_args
            else:
                query_string = str(tool_args)
            
            print(f"Executing tool: {tool_name} with query: '{query_string[:100]}...'")
            try:
                # Invoke tool with the query string
                response = tool_to_call.invoke(query_string)
                
                # Format the response nicely
                response_text = str(response)
                if len(response_text) > 1000:
                    response_text = response_text[:1000] + "... (truncated)"
                
                new_tool_responses.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"Tool '{tool_name}' returned: {response_text}",
                        name=tool_name
                    )
                )
                print(f"✅ Tool '{tool_name}' executed successfully. Response length: {len(str(response))}")
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {e}"
                print(f"❌ {error_msg}")
                new_tool_responses.append(
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"Error: {error_msg}",
                        name=tool_name
                    )
                )
        else:
            error_msg = f"Tool '{tool_name}' not found. Available tools: {list(tool_map.keys())}"
            print(f"⚠️ {error_msg}")
            new_tool_responses.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=error_msg,
                    name=tool_name
                )
            )
    
    # Return new tool responses (they will be accumulated by the reducer)
    return {"tool_responses": new_tool_responses, "tool_calls": []}


# --- 4. Define the Graph Edges (Conditional Logic) ---
def should_continue(state: AgentState):
    """
    This is our main conditional edge. It decides whether to
    call tools again or to finish and go to the final generator.
    """
    if state.get("tool_calls"):
        # If the agent node produced tool calls, we go execute them
        return "continue"
    else:
        # If the agent node produced a final answer, we are done
        return "end"

# --- 5. Assemble the Graph ---
def create_graph():
    """
    This function assembles all the nodes and edges into the final graph.
    """
    
    workflow = StateGraph(AgentState)

    # We only have two main nodes now:
    # 1. "agent": The LLM worker that decides what to do
    # 2. "tools": The executor that runs the tools
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_executor_node)

    # The entry point is the "agent"
    workflow.set_entry_point("agent")

    # This is the conditional routing
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",  # If tools need to be called, go to 'tools'
            "end": END            # If no tools, end the graph
        }
    )

    # After the "tools" node runs, it *always* goes back to the "agent"
    # to analyze the results and decide what to do next.
    workflow.add_edge("tools", "agent")

    # Compile the graph
    print("Compiling LangGraph...")
    graph = workflow.compile(
        checkpointer=MemorySaver() # This gives our graph persistent memory
    )
    print("LangGraph compiled successfully.")
    return graph

# Create the runnable graph object for our app to import
runnable_graph = create_graph()