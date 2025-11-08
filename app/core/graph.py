import os
import json
from typing import List, TypedDict, Annotated, Literal
from langchain_core.prompts import ChatPromptTemplate
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
# Connect to the DB
db = SQLDatabase.from_uri(
    DB_URL_SQL_AGENT,
    include_tables=["Artist", "Album", "Track", "Customer", "Employee", "Invoice", "InvoiceLine"]
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()  # returns ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDatabaseTool, etc.


# print("Connected to SQL Database for Agent." , db.get_table_names()) ;
# Create the direct tools
# list_tables_tool = ListSQLDatabaseTool(db=db)
# get_schema_tool = InfoSQLDatabaseTool(db=db)
# run_query_tool = QuerySQLDatabaseTool(db=db)

# --- RAG Tools ---
feedback_tool = create_feedback_rag_tool()
product_tool = create_product_rag_tool()

# --- Full Tool List ---
tools = [
    list_tables_tool, 
    get_schema_tool, 
    run_query_tool, 
    feedback_tool, 
    product_tool
]


# --- 2. Define the Graph State ---
class AgentState(TypedDict):
    input: str
    chat_history: list
    messages: Annotated[list, lambda x, y: x + y]
    tool_calls: list
    tool_responses: list


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
def agent_node(state: AgentState):
    """
    The main "worker" node. It takes the list of tool calls from the router,
    decides what inputs to use, and then calls them.
    """
    print("--- CALLING AGENT/TOOL NODE ---")
    
    # We need a prompt that tells the LLM *how* to use the tools
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You must use the tools provided to answer the user's request."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="tool_responses")
    ])
    
    # Bind the tools to the LLM
    agent_llm = llm.bind_tools(tools)
    
    # Create the agent chain
    agent_chain = agent_prompt | agent_llm
    
    # Get the tool responses from the previous state
    tool_responses = state.get('tool_responses', [])
    
    # Call the agent
    # We pass the full message history and any tool responses
    response_message = agent_chain.invoke({
        "messages": state['messages'] + tool_responses,
        "tool_responses": []
    })
    
    # If the LLM responded with tool calls, we return them
    if response_message.tool_calls:
        print(f"Agent decided to call tools: {[tc['name'] for tc in response_message.tool_calls]}")
        return {"tool_calls": response_message.tool_calls}
    
    # If the LLM responded with a final answer, we return it
    print("Agent provided a final answer.")
    return {"messages": [response_message]}


# --- NODE 3: The Tool Executor Node ---
def tool_executor_node(state: AgentState):
    """
    This node *only* executes tools.
    """
    print("--- CALLING TOOL EXECUTOR NODE ---")
    tool_map = {tool.name: tool for tool in tools}
    tool_responses = []
    
    for tool_call in state["tool_calls"]:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        tool_to_call = tool_map.get(tool_name)
        
        if tool_to_call:
            print(f"Executing tool: {tool_name} with args {tool_args}")
            try:
                # We use .invoke() for tools
                response = tool_to_call.invoke(tool_args)
                tool_responses.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=str(response),
                        name=tool_name
                    )
                )
            except Exception as e:
                print(f"Error executing tool {tool_name}: {e}")
                tool_responses.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"Error: {e}",
                        name=tool_name
                    )
                )
    
    return {"tool_responses": tool_responses, "tool_calls": []}


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