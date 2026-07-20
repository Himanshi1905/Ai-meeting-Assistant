import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai import ChatMistralAI

class ActionItem(BaseModel):
    description: str = Field(description="The clear, actionable task or checklist item.")
    owner: Optional[str] = Field(None, description="The person assigned to or responsible for the task. Set to null if not mentioned.")
    deadline: Optional[str] = Field(None, description="The deadline or timeframe for the task. Set to null if not mentioned.")

class Decision(BaseModel):
    description: str = Field(description="The key decision or alignment reached during the meeting.")
    context: Optional[str] = Field(None, description="Any brief background context or reasoning behind this decision.")

class OpenQuestion(BaseModel):
    question: str = Field(description="An unresolved question, blocker, or open discussion topic.")
    context: Optional[str] = Field(None, description="Who raised the question or what context it was raised in.")

class ExtractedElements(BaseModel):
    action_items: List[ActionItem] = Field(description="List of actionable items extracted from the transcript.")
    decisions: List[Decision] = Field(description="List of key decisions made.")
    open_questions: List[OpenQuestion] = Field(description="List of unresolved or open questions.")

def extract_meeting_elements(transcript: str, api_key: str = None, model_name: str = "open-mistral-7b") -> dict:
    """
    Analyzes the meeting transcript and extracts actionable items, decisions,
    and open questions as a structured dictionary.
    """
    # Fetch API key from argument or environment
    mistral_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("Mistral AI API key is missing. Please set the MISTRAL_API_KEY environment variable.")
        
    if not transcript or not transcript.strip():
        return {
            "action_items": [],
            "decisions": [],
            "open_questions": []
        }
        
    print(f"[Extractor] Initializing ChatMistralAI (Model: {model_name}) for extraction...")
    
    # Initialize the Chat LLM
    llm = ChatMistralAI(
        model=model_name,
        mistral_api_key=mistral_key,
        temperature=0.1  # lower temperature for analytical extraction
    )
    
    # Set up parser
    parser = JsonOutputParser(pydantic_object=ExtractedElements)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert project manager and assistant. Your task is to analyze meeting transcripts and extract structured information, specifically action items, key decisions, and unresolved/open questions.\n{format_instructions}"
        ),
        (
            "user",
            "Please analyze the following meeting transcript and extract the action items (along with who is assigned and deadlines, if mentioned), the key decisions, and any unresolved questions or blockers.\n\nTranscript:\n{transcript}"
        )
    ])
    
    # Inject output format instructions
    prompt_with_instructions = prompt.partial(format_instructions=parser.get_format_instructions())
    
    # Build LangChain LCEL chain
    chain = prompt_with_instructions | llm | parser
    
    try:
        print("[Extractor] Invoking extraction chain...")
        result = chain.invoke({"transcript": transcript})
        
        # Verify result contains the expected keys, filling in defaults if missing
        final_result = {
            "action_items": result.get("action_items", []),
            "decisions": result.get("decisions", []),
            "open_questions": result.get("open_questions", [])
        }
        return final_result
    except Exception as e:
        print(f"[Extractor] Warning: Extraction chain failed: {e}. Returning empty structure.")
        return {
            "action_items": [],
            "decisions": [],
            "open_questions": []
        }
