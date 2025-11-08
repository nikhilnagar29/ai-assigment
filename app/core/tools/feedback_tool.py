import os
from langchain.tools import tool, BaseTool
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from core.config import llm, embeddings, FEEDBACK_VECTOR_STORE_PATH

def create_feedback_rag_tool():
    """
    This function creates our specialized "Feedback Expert" tool.
    It loads the FAISS index and builds a RAG chain.
    """
    print("Initializing Feedback RAG tool...")

    # 1. Load the existing vector store
    if not os.path.exists(FEEDBACK_VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Feedback vector store not found at {FEEDBACK_VECTOR_STORE_PATH}. Please run the build script.")
    
    db = FAISS.load_local(
        FEEDBACK_VECTOR_STORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 2. Create a retriever
    retriever = db.as_retriever(search_kwargs={"k": 5}) # Get top 5 results

    # 3. Create a custom prompt
    # This prompt is designed to use the metadata in your feedback file.
    PROMPT_TEMPLATE = """
You are a helpful assistant for BMW. Your task is to find and summarize customer feedback.
Answer the user's question based *only* on the following feedback documents.
For each piece of feedback you use, you MUST cite the 'Source', 'User ID', and 'Sentiment'.

If the query is general (e.g., "what do people think?"), try to find both positive and negative themes.
If no relevant feedback is found, just say "I could not find any customer feedback on that topic."

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
    # This chain will "stuff" the retrieved documents into the prompt
    prompt = ChatPromptTemplate.from_template("Your prompt here")

    retriever = vectorstore.as_retriever()

    rag_chain = create_retrieval_chain(
        retriever,
        llm,
    )


    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True # This lets us see what it found
    )
    
    # 5. Create the final Tool
    # The 'description' is critical - it tells the LangGraph agent
    # *when* to use this tool.
    feedback_tool = tool(
        name="customer_feedback_search",
        func=lambda q: rag_chain.invoke(q)["result"], # We only want the final answer
        description=(
            "Use this tool to search for customer feedback, opinions, complaints, or sentiments "
            "about the BMW iX, its features (e.g., Sky Lounge, charging, range), or the "
            "dealership experience. Input should be a specific topic to search for."
        )
    )
    
    print("Feedback RAG tool created successfully.")
    return feedback_tool