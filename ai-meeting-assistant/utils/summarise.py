import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

def generate_summary(transcript: str, api_key: str = None, model_name: str = "open-mistral-7b") -> str:
    """
    Generates a structured, professional meeting-style summary of a transcript
    using Mistral AI via LangChain Expression Language (LCEL).
    
    Includes a Title, overall summary section, and a list of key discussion points.
    """
    # Fetch API key
    mistral_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("Mistral AI API key is missing. Please set the MISTRAL_API_KEY environment variable or pass it to the function.")
        
    if not transcript or not transcript.strip():
        return "No transcript content available to summarize."
        
    print(f"[Summarizer] Initializing ChatMistralAI (Model: {model_name})...")
    
    # Initialize the Mistral Chat LLM
    llm = ChatMistralAI(
        model=model_name,
        mistral_api_key=mistral_key,
        temperature=0.3
    )
    
    # Construct the summary prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert AI meeting assistant. Your task is to analyze meeting transcripts and generate a professional, structured summary in Markdown format."
        ),
        (
            "user",
            """
            Analyze the following transcript of a meeting/recording. Generate a clean and readable Markdown document.
            
            Ensure the output structure follows this template exactly:
            # [Provide a descriptive and professional Title for the meeting]
            
            ## Meeting Overview
            [A brief high-level paragraph summarizing the main purpose, tone, and major outcomes of the meeting]
            
            ## Key Discussion Points
            [A structured list of main topics discussed. For each point, provide a brief description of the context, decisions made, or opinions shared]
            
            ---
            Transcript:
            {transcript}
            """
        )
    ])
    
    # Define LangChain LCEL pipeline
    chain = prompt | llm | StrOutputParser()
    
    print("[Summarizer] Invoking LangChain LCEL pipeline...")
    summary = chain.invoke({"transcript": transcript})
    
    return summary.strip()
