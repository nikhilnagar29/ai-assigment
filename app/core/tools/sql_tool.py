from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from core.config import llm, DB_URL_SQL_AGENT

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
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # 3. Create the SQL Agent
    # This combines the LLM and the toolkit into a runnable agent
    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,  # This will print the agent's "thoughts"
        agent_type="openai-functions", # This is the standard, most reliable agent type
        handle_parsing_errors=True # This helps the agent self-correct
    )
    
    print("SQL Agent created successfully.")
    return sql_agent