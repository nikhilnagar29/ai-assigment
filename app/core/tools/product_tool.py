import os
from langchain.tools import Tool
from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from core.config import llm, embeddings, PRODUCT_VECTOR_STORE_PATH

def create_product_rag_tool():
    """
    This function creates our specialized "Product Expert" tool.
    It loads the product FAISS index and builds a RAG chain.
    """
    print("Initializing Product RAG tool...")

    # 1. Load the existing product vector store
    if not os.path.exists(PRODUCT_VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Product vector store not found at {PRODUCT_VECTOR_STORE_PATH}. "
            "Please run: docker-compose exec app python core/vector_builder.py"
        )
    
    db = FAISS.load_local(
        PRODUCT_VECTOR_STORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 2. Create a retriever
    retriever = db.as_retriever(search_kwargs={"k": 5}) # Get top 5 pages

    # 3. Create a custom prompt for technical specs
    PROMPT_TEMPLATE = """
You are a BMW Product Specialist. Your goal is to answer technical and feature-related
questions about the BMW iX based *only* on the provided context from the product brochure.

- Be precise and factual.
- Answer only with information found in the context.
- If the answer is not in the context, state clearly: "I do not have that specific information in the product brochure."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # 4. Create the RAG (RetrievalQA) chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # 5. Create the final Tool for the LangGraph agent
    product_tool = Tool(
        name="product_details_search",
        func=lambda q: rag_chain.invoke(q)["result"],
        description=(
            "Use this tool to find technical specifications, features, or details about the "
            "BMW iX. This includes information on range, charging, performance, "
            "interior features (like Sky Lounge), driver assistance, and sustainability."
        )
    )
    
    print("Product RAG tool created successfully.")
    return product_tool