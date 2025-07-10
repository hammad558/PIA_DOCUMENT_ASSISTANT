from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import secrets

# --- Configuration ---
load_dotenv()
app = FastAPI(title="PIA Document Assistant API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
FAISS_INDEX_PATH = "faiss_vector_store"
EMBEDDING_MODEL = 'models/embedding-001'
LLM_MODEL = "llama3-8b-8192"

# --- Models ---
class QueryRequest(BaseModel):
    question: str
    translate_to_urdu: bool = False
    api_key: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    response_time: float
    translated_answer: Optional[str] = None
    is_grounded: bool

# --- Core Service ---
class DocumentAssistant:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.environ['GROQ_API_KEY'],
            model_name=LLM_MODEL
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        """Load pre-built FAISS index"""
        if not os.path.exists(FAISS_INDEX_PATH):
            raise HTTPException(
                status_code=400,
                detail="Vector store not found. Please build it first using the Streamlit app."
            )
        return FAISS.load_local(
            FAISS_INDEX_PATH, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )

# Improved intent classification in process_query() method
    def process_query(self, question: str) -> dict:
        """Enhanced query processing with better intent detection"""
        # 1. More flexible intent classification
        intent_prompt = ChatPromptTemplate.from_template(
        """Classify the query type:
        - 'DocumentQuestion': Specific questions about document content
        - 'GeneralQuestion': Other valid questions
        - 'Other': Greetings/chit-chat
        
        Examples:
        Q: What was PIA's revenue in 2009? → DocumentQuestion
        Q: Tell me about PIA → GeneralQuestion  
        Q: Hello → Other
        
        Query: {input}
        Classification:"""
        )
        intent = (intent_prompt | self.llm).invoke({"input": question}).content.strip()

        if "DocumentQuestion" not in intent and "GeneralQuestion" not in intent:
            return {
            "answer": "Please ask a complete question about your documents.",
            "sources": [],
            "response_time": 0.0,
            "is_grounded": False
        }

    # 2. Enhanced document retrieval prompt
        prompt_template = ChatPromptTemplate.from_template(
            """Answer the question precisely using only this context:
            {context}
        
        Question: {input}
        Answer in complete sentences with relevant numbers/data:"""
        )
    
    # Rest of your retrieval logic remains the same...        
        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        retriever = self.vector_store.as_retriever()
        chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.time()
        response = chain.invoke({"input": question})
        response_time = time.time() - start_time

        # 3. Grounding verification
        context = "\n".join(d.page_content for d in response["context"])
        is_grounded = (ChatPromptTemplate.from_template(
            """Verify if the answer is fully supported by the context. 
            Respond exactly 'Yes' or 'No':
            Context: {context}
            Answer: {answer}"""
        ) | self.llm).invoke({
            "context": context,
            "answer": response["answer"]
        }).content.strip() == "Yes"

        return {
            "answer": response["answer"],
            "sources": list(set(
                os.path.basename(d.metadata["source"]) 
                for d in response["context"]
            )),
            "response_time": response_time,
            "is_grounded": is_grounded
        }

    def translate(self, text: str) -> str:
        """Urdu translation"""
        return (ChatPromptTemplate.from_template(
            "Translate this to Urdu while preserving legal terminology: {text}"
        ) | self.llm).invoke({"text": text}).content

# --- API Routes ---
assistant = DocumentAssistant()

@app.post("/query")
async def query_documents(request: QueryRequest):
    if not secrets.compare_digest(request.api_key, os.environ['API_KEY']):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    result = assistant.process_query(request.question)
    
    if request.translate_to_urdu:
        result["translated_answer"] = assistant.translate(result["answer"])
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)