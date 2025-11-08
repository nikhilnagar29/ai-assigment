from langchain_core.tools import Tool  # <-- FIX 1
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from core.config import llm, DB_URL_SQL_AGENT

# The 7 most important tables for business questions.
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
    This function creates our specialized "SQL Expert" agent
    and wraps it in a Tool for LangGraph to use.
    """
    print("Initializing SQL Database connection...")
    db = SQLDatabase.from_uri(
        DB_URL_SQL_AGENT,
        include_tables=INCLUDE_TABLES
    )
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    sql_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-functions",
        handle_parsing_errors=True
    )
    
    # We wrap the agent in a Tool
    sql_tool = Tool(
        name="chinook_database_sql",
        func=sql_agent_executor.invoke, 
        description=(
            "Use this tool to answer questions about Chinook database sales, customers, "
            "artists, albums, tracks, invoices, and employees. "
            "Input should be a full natural language question."
        )
    )
    
    print("SQL Agent Tool created successfully.")
    return sql_tool