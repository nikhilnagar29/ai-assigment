import os
import sys
import time
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, DirectoryLoader

# --- FIX 1: Python Path Correction ---
# Add the parent directory ('/app') to the Python path
# This allows us to import from the 'core' module (e.g., core.config)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.config import (
        embeddings, 
        PRODUCT_DATA_PATH, 
        FEEDBACK_DATA_PATH, 
        PRODUCT_VECTOR_STORE_PATH, 
        FEEDBACK_VECTOR_STORE_PATH
    )
except ImportError:
    print("❌ Error: Could not import from core.config.")
    print("Please ensure __init__.py exists in the /app/core/ folder.")
    sys.exit(1)


def load_documents(path, loader_class, **loader_kwargs):
    """Loads documents from a directory using a specific loader."""
    loader = DirectoryLoader(
        path,
        loader_cls=loader_class,
        loader_kwargs=loader_kwargs,
        show_progress=True,
        use_multithreading=True
    )
    return loader.load()

def build_vector_store(store_path: str, documents: list, embedding_model):
    """
    Builds and saves a FAISS vector store one document at a time
    to respect the Gemini free tier rate limits.
    """
    try:
        if not documents:
            print(f"⚠️ No documents found to build {store_path}.")
            return

        print(f"Building store for {len(documents)} documents. This will take time...")
        
        # --- FIX 2: Rate Limit Handling ---
        # Embed the *first* document to initialize the store
        db = FAISS.from_documents([documents[0]], embedding_model)
        print("Embedded 1 document...")
        
        # Loop through the *rest* of the documents, one by one
        for doc in documents[1:]:
            try:
                # Add one doc at a time
                db.add_documents([doc])
                # IMPORTANT: Wait 1.5 seconds between each request
                # This respects the Google API free tier rate limit.
                time.sleep(1.5) 
                print("Embedded 1 document...")
            except Exception as e:
                print(f"Error embedding document: {e}. Waiting 5 seconds and retrying.")
                time.sleep(5) # Wait longer if we hit an error

        # Once all documents are added, save the final store to disk
        db.save_local(store_path)
        print(f"✅ Successfully built and saved vector store at: {store_path}")

    except Exception as e:
        print(f"❌ CRITICAL Error building vector store at {store_path}: {e}")

def main():
    print("Starting vector store build process...")

    # --- 1. Product Vector Store (from PDFs) ---
    if os.path.exists(PRODUCT_VECTOR_STORE_PATH):
        print(f"Product vector store already exists at {PRODUCT_VECTOR_STORE_PATH}. Skipping.")
    else:
        print(f"Product vector store not found. Building from {PRODUCT_DATA_PATH}...")
        product_docs = load_documents(PRODUCT_DATA_PATH, PyPDFLoader)
        build_vector_store(PRODUCT_VECTOR_STORE_PATH, product_docs, embeddings)

    # --- 2. Feedback Vector Store (from TXT/CSV) ---
    if os.path.exists(FEEDBACK_VECTOR_STORE_PATH):
        print(f"Feedback vector store already exists at {FEEDBACK_VECTOR_STORE_PATH}. Skipping.")
    else:
        print(f"Feedback vector store not found. Building from {FEEDBACK_DATA_PATH}...")
        feedback_docs = load_documents(
            FEEDBACK_DATA_PATH, 
            CSVLoader, 
            csv_args={"delimiter": ","},
            source_column="Text"
        )
        build_vector_store(FEEDBACK_VECTOR_STORE_PATH, feedback_docs, embeddings)
            
    print("Vector store build process complete.")

if __name__ == "__main__":
    main()