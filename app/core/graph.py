import os
import json
from typing import List, TypedDict, Annotated, Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.config import llm
from core.tools.sql_tool import create_sql_agent_tool
from core.tools.feedback_tool import create_feedback_rag_tool
from core.tools.product_tool import create_product_rag_tool

# --- 1. Define the Tools ---
# Initialize our "expert" tools
sql_tool = create_sql_agent_tool()
feedback_tool = create_feedback_rag_tool()
product_tool = create_product_rag_tool()

tools = [sql_tool, feedback_tool, product_tool]


# --- 2. Define the Graph State ---
# This is the "memory" of our graph as it runs.
class AgentState(TypedDict):
    input: str
    chat_history: list
    # This `messages` list is the core of the graph's memory
    messages: Annotated[list, lambda x, y: x + y]
    # This will hold the tool calls our Router decides on
    tool_calls: list
    # This will hold the outputs from our tools
    tool_responses: list


# --- 3. Define the Graph Nodes ---

# --- NODE 1: The ROUTER ---
# This defines the *output format* we want from our router LLM
class ToolRouter(BaseModel):
    """A router to select the appropriate tool(s) for a user query."""
    tool_names: List[Literal["chinook_database_sql", "customer_feedback_search", "product_details_search"]] = Field(
        ..., 
        description="A list of tool names to call. Only use the tools provided."
    )

# This connects our LLM (Gemini) to the ToolRouter format
structured_llm_router = llm.with_structured_output(ToolRouter)

def router_node(state: AgentState):
    """
    This node analyzes the user's query and decides which tools to call.
    """
    print("--- CALLING ROUTER NODE ---")
    
    # Build the prompt for the router
    prompt = f"""
    You are an expert router. Your job is to analyze the user's query and 
    determine which tool(s) are needed to answer it.

    The available tools are:
    1. chinook_database_sql: For questions about sales, customers, artists, albums, etc.
    2. customer_feedback_search: For questions about customer opinions, feedback, or complaints.
    3. product_details_search: For questions about BMW iX product features or technical specs.

    User Query: "{state['input']}"

    Respond with a list of the *exact* tool names required.
    If a query needs data from multiple sources (e.g., "sales for BMW" and "feedback about it"),
    you MUST include all relevant tool names.
    """
    
    # Call the router LLM
    # We use HumanMessage to ensure it's a new turn
    router_output = structured_llm_router.invoke([HumanMessage(content=prompt)])
    
    tool_calls = []
    if router_output.tool_names:
        for tool_name in router_output.tool_names:
            # We create a "tool call" object for each tool
            tool_calls.append(
                ToolMessage(
                    tool_call_id=f"call_{tool_name}", 
                    content=state['input'], # Pass the original query to the tool
                    name=tool_name
                )
            )
    
    print(f"Router decided to call: {router_output.tool_names}")
    # Add the tool calls to the state
    return {"tool_calls": tool_calls}


# --- NODE 2: The TOOL CALLER ---
def tool_node(state: AgentState):
    """
    This node runs the tools chosen by the router.
    It runs them in parallel (LangGraph handles this).
    """
    print("--- CALLING TOOL NODE ---")
    tool_responses = []
    
    # Find the right tool from our list
    tool_map = {tool.name: tool for tool in tools}

    for tool_call in state["tool_calls"]:
        tool_name = tool_call.name
        tool_to_call = tool_map.get(tool_name)
        
        if tool_to_call:
            print(f"Running tool: {tool_name}")
            
            # Call the tool (e.g., sql_tool.invoke(...))
            response = tool_to_call.invoke(
                {"input": tool_call.content} if tool_name == "chinook_database_sql" 
                else tool_call.content
            )
            
            # Store the tool's output
            tool_responses.append(
                ToolMessage(
                    tool_call_id=tool_call.tool_call_id,
                    content=str(response),
                    name=tool_name
                )
            )
        else:
            print(f"Warning: Tool '{tool_name}' not found.")
            
    return {"tool_responses": tool_responses}


# --- NODE 3: The FINAL GENERATOR/COMBINER ---
def final_generator_node(state: AgentState):
    """
    This is the final node. It takes all the tool results and the original query
    and generates a single, clean answer for the user.
    """
    print("--- CALLING FINAL GENERATOR NODE ---")
    
    # Get the original query and the tool responses
    query = state['input']
    tool_responses = state['tool_responses']

    # Build a prompt for the final answer
    prompt_context = "Based on your request, here is the information I found:\n\n"
    
    # Add each tool's response to the context
    for response in tool_responses:
        if response.name == 'chinook_database_sql':
            prompt_context += f"--- SQL Database Results ---\n{response.content}\n\n"
        elif response.name == 'customer_feedback_search':
            prompt_context += f"--- Customer Feedback Results ---\n{response.content}\n\n"
        elif response.name == 'product_details_search':
            prompt_context += f"--- Product Details Results ---\n{response.content}\n\n"
            
    prompt = f"""
    You are a final summarizer. Your job is to take the user's original question
    and the results from different tools, and write a single, clean,
    and helpful answer.

    Original Question: {query}

    Here is the data you have to work with:
    {prompt_context}

    Combine this information into a final, comprehensive answer.
    """
    
    # Generate the final response
    final_response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [AIMessage(content=final_response.content)]}


# --- 4. Define the Graph Edges (The Logic) ---
def create_graph():
    """
    This function assembles all the nodes and edges into the final graph.
    """
    
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("router", router_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("final_generator", final_generator_node)

    # --- This is the logic you asked for ---
    
    # 1. Set the Entry Point
    workflow.set_entry_point("router")
    
    # 2. Add the Conditional Edge
    # After the "router" node, this edge will call the "tool_node"
    workflow.add_edge("router", "tool_node")
    
    # 3. Add the Final Edge
    # After the "tool_node" runs, it will always go to the "final_generator"
    workflow.add_edge("tool_node", "final_generator")

    # 4. Set the Finish Point
    workflow.add_edge("final_generator", END)

    # 5. Compile the graph
    print("Compiling LangGraph...")
    graph = workflow.compile(
        checkpointer=MemorySaver() # This gives our graph memory
    )
    print("LangGraph compiled successfully.")
    return graph

# --- 5. Create the runnable graph object ---
# We create this once so our Streamlit app can import it
runnable_graph = create_graph()