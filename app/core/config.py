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
# - If DB_HOST is explicitly set, use it (highest priority)
# - If running inside Docker container, use "db" (Docker service name)
# - Otherwise, default to "localhost" (for local development)
DB_HOST = os.environ.get("DB_HOST")
if not DB_HOST:
    # Check if we're running inside a Docker container
    _is_docker = os.path.exists("/.dockerenv")
    DB_HOST = "db" if _is_docker else "localhost"

DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "chinook")

# Connection string for the SQL Agent
DB_URL_SQL_AGENT = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Print connection info for debugging (without password)
print(f"Database connection configured: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# --- File Paths ---
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data paths
PRODUCT_DATA_PATH = os.path.join(APP_DIR, "data", "products")
FEEDBACK_DATA_PATH = os.path.join(APP_DIR, "data", "feedback")

# Output FAISS vector store paths
PRODUCT_VECTOR_STORE_PATH = os.path.join(APP_DIR, "vector_stores", "product_db")
FEEDBACK_VECTOR_STORE_PATH = os.path.join(APP_DIR, "vector_stores", "feedback_db")