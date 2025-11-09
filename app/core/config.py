import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from .env file (for your GOOGLE_API_KEY)
load_dotenv()

# --- LLM and Embedding Models ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


# --- Database Connection ---
# This is for the main SQL database
DB_USER = os.environ.get("DB_USER", "chinook")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "chinook")

# Database host configuration:
# Smart detection: Try Docker first, fall back to localhost
DB_HOST = os.environ.get("DB_HOST")
if not DB_HOST:
    # Check if we're running inside a Docker container
    _is_docker = os.path.exists("/.dockerenv")
    if _is_docker:
        DB_HOST = "db"  # Docker service name
    else:
        # Running locally - try to detect if database is accessible
        import socket
        # First try 'db' (in case of Docker network)
        try:
            socket.create_connection(("db", 5432), timeout=1)
            DB_HOST = "db"
        except (socket.gaierror, socket.timeout, OSError):
            # If 'db' fails, use localhost
            DB_HOST = "localhost"
            print("⚠️  Running locally. Using localhost for database connection.")
            print("   If database connection fails, make sure you have:")
            print("   1. Started the database: docker-compose up -d db")
            print("   2. Set DB_HOST=localhost in your .env file")

DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "chinook")

# Connection string for the SQL Agent
DB_URL_SQL_AGENT = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Print connection info for debugging (without password)
print("=" * 60)
print("DATABASE CONNECTION CONFIGURATION")
print("=" * 60)
print(f"Host: {DB_HOST}")
print(f"Port: {DB_PORT}")
print(f"Database: {DB_NAME}")
print(f"User: {DB_USER}")
print(f"Full URL: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
print("=" * 60)


# --- File Paths ---
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data paths
PRODUCT_DATA_PATH = os.path.join(APP_DIR, "data", "products")
FEEDBACK_DATA_PATH = os.path.join(APP_DIR, "data", "feedback")

# Output FAISS vector store paths
PRODUCT_VECTOR_STORE_PATH = os.path.join(APP_DIR, "vector_stores", "product_db")
FEEDBACK_VECTOR_STORE_PATH = os.path.join(APP_DIR, "vector_stores", "feedback_db")