from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from core.config import llm, DB_URL_SQL_AGENT
from langchain.tools import Tool

# --- This is the key to fulfilling your "up to 10 tables" requirement ---
# We will not show the AI all 11 tables. We will only show it the 7
# most important ones for answering business questions.
# This makes the AI smarter, faster, and cheaper.
INCLUDE_TABLES = [
    "Artist",
    "Album",
    "Track",
    "Customer",
    "Employee",
    "Invoice",
    "InvoiceLine"
]

def create_sql_agent_tool():
    """
    This function creates our specialized "SQL Expert" agent.
    """
    print("Initializing SQL Database connection...")
    
    # 1. Connect to the database
    db = SQLDatabase.from_uri(
        DB_URL_SQL_AGENT,
        include_tables=INCLUDE_TABLES  # <-- Only give the agent these tables
    )
    
    # 2. Create the toolkit
    # This provides the agent with all the functions it needs
    # (e.g., "list tables," "check table schema," "run query")
    sql_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-functions",
        handle_parsing_errors=True
    )
    
    # --- THIS IS THE NEW PART ---
    # We wrap the agent in a Tool. This is the standard
    # way to make one agent a "tool" for another agent.
    sql_tool = Tool(
        name="chinook_database_sql",
        # The agent's .invoke method will be called when this tool is used
        func=sql_agent_executor.invoke, 
        description=(
            "Use this tool to answer questions about Chinook database sales, customers, "
            "artists, albums, tracks, invoices, and employees. "
            "Input should be a full natural language question."
        )
    )
    
    print("SQL Agent Tool created successfully.")
    return sql_tool