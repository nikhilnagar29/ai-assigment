"""
Database Connection Test Script
Run this to diagnose database connection issues:
    python test_database_connection.py
"""

import os
from dotenv import load_dotenv

print("=" * 70)
print("DATABASE CONNECTION DIAGNOSTIC TOOL")
print("=" * 70)

import os

# Try to load dotenv, but continue if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.")
    print("   To install: pip install python-dotenv")
except Exception:
    print("⚠️  No .env file found. Using system environment variables.")


# Step 1: Check environment variables
print("\n1️⃣ CHECKING ENVIRONMENT VARIABLES")
print("-" * 70)

DB_USER = os.environ.get("DB_USER", "chinook")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "chinook")
DB_HOST = os.environ.get("DB_HOST")
if not DB_HOST:
    _is_docker = os.path.exists("/.dockerenv")
    DB_HOST = "db" if _is_docker else "localhost"
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "chinook")

print(f"DB_HOST: {DB_HOST}")
print(f"DB_PORT: {DB_PORT}")
print(f"DB_NAME: {DB_NAME}")
print(f"DB_USER: {DB_USER}")
print(f"DB_PASSWORD: {'*' * len(DB_PASSWORD)}")

# Step 2: Check if psycopg2 is installed
print("\n2️⃣ CHECKING REQUIRED LIBRARIES")
print("-" * 70)

try:
    import psycopg2
    print("✅ psycopg2 is installed")
    print(f"   Version: {psycopg2.__version__}")
except ImportError:
    print("❌ psycopg2 is NOT installed!")
    print("\nTo install, run:")
    print("   pip install psycopg2-binary")
    exit(1)

# Step 3: Test raw psycopg2 connection
print("\n3️⃣ TESTING RAW DATABASE CONNECTION (psycopg2)")
print("-" * 70)

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    print("✅ Raw connection successful!")
    
    # Try to list tables
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = cursor.fetchall()
    print(f"✅ Found {len(tables)} tables:")
    for table in tables:
        print(f"   - {table[0]}")
    
    cursor.close()
    conn.close()
    
except psycopg2.OperationalError as e:
    print("❌ Connection FAILED!")
    print(f"Error: {e}")
    print("\nPossible causes:")
    print("1. Database server is not running")
    print("   Fix: docker-compose up -d db")
    print("2. Wrong hostname")
    print("   Fix: Try 'localhost' or '127.0.0.1' instead of 'db'")
    print("3. Wrong credentials")
    print("   Fix: Check your .env file")
    exit(1)

# Step 4: Test SQLAlchemy connection
print("\n4️⃣ TESTING SQLALCHEMY CONNECTION")
print("-" * 70)

try:
    from sqlalchemy import create_engine, inspect
    
    DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"Connection string: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    engine = create_engine(DB_URL)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print("✅ SQLAlchemy connection successful!")
    print(f"✅ Found {len(tables)} tables: {tables}")
    
except Exception as e:
    print("❌ SQLAlchemy connection FAILED!")
    print(f"Error: {e}")
    exit(1)

# Step 5: Test LangChain SQLDatabase
print("\n5️⃣ TESTING LANGCHAIN SQLDATABASE")
print("-" * 70)

try:
    from langchain_community.utilities import SQLDatabase
    
    db = SQLDatabase.from_uri(
        DB_URL,
        include_tables=["artist", "album", "track", "customer", "employee", "invoice", "invoice_line"]
    )
    
    print("✅ LangChain SQLDatabase created successfully!")
    
    usable_tables = db.get_usable_table_names()
    print(f"✅ Usable tables: {usable_tables}")
    
    # Try a simple query
    result = db.run("SELECT COUNT(*) FROM customer;")
    print(f"✅ Test query successful! Customer count: {result}")
    
except Exception as e:
    print("❌ LangChain SQLDatabase FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Test SQL Toolkit
print("\n6️⃣ TESTING SQL TOOLKIT")
print("-" * 70)

try:
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    print("✅ SQL Toolkit created successfully!")
    print(f"✅ Number of tools: {len(tools)}")
    print("\nAvailable tools:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description[:60]}...")
    
except Exception as e:
    print("❌ SQL Toolkit FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Final summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED! Database connection is working correctly.")
print("=" * 70)
print("\nYour SQL tools should now work in the agent.")
print("If they still don't work, check your graph.py file.")