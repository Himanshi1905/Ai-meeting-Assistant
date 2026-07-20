import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from utils.vector_store import get_vectorstore

def format_docs(docs) -> str:
    """
    Formmats retrieved Documents into a single block of text for LLM context.
    """
    formatted_chunks = []
    for doc in docs:
        content = doc.page_content.strip()
        formatted_chunks.append(content)
    return "\n\n---\n\n".join(formatted_chunks)

def query_meeting(meeting_id: int, question: str, api_key: str = None, model_name: str = "open-mistral-7b") -> str:
    """
    Retrieves chunks relevant to a specific meeting ID and uses Mistral AI to answer.
    """
    # Fetch API key
    mistral_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("Mistral AI API key is missing. Please set the MISTRAL_API_KEY environment variable.")
        
    if not question or not question.strip():
        return "Please ask a valid question."
        
    print(f"[RAG Engine] Querying meeting ID {meeting_id} with question: '{question}'...")
    
    # 1. Initialize vector store
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        return f"Error loading vector store database: {e}"
        
    # 2. Configure filtered retriever (only search documents matching this meeting ID)
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": {"meeting_id": meeting_id},
            "k": 4
        }
    )
    
    # 3. Initialize Mistral Chat model
    llm = ChatMistralAI(
        model=model_name,
        mistral_api_key=mistral_key,
        temperature=0.0  # Zero temperature for factual grounded answers
    )
    
    # 4. Construct Prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting assistant. Answer the user's question about the meeting based ONLY on the provided transcript context. "
            "Be concise, professional, and truthful.\n\n"
            "If the provided context does not contain enough information to answer the question, respond with exactly: "
            "\"I could not find the answer in the meeting transcript.\" Do not make up facts or extrapolate."
        ),
        (
            "user",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
    ])
    
    # 5. Define chain using LangChain LCEL
    # We retrieve the docs, format them, and pipe the inputs through prompt, llm, and output parser
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        response = rag_chain.invoke(question)
        return response.strip()
    except Exception as e:
        print(f"[RAG Engine] Error invoking RAG chain: {e}")
        return f"An error occurred while retrieval-querying the model: {e}"
