from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
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
from twilio.rest import Client

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
BASE_DOCUMENT_URL = "https://piacconnect.piac.com.pk/documents"  # Updated base URL

# --- Twilio Client ---
twilio_client = Client(
    os.getenv('TWILIO_ACCOUNT_SID'),
    os.getenv('TWILIO_AUTH_TOKEN')
)

# --- Models ---
class DocumentSource(BaseModel):
    filename: str
    folder: str  # '20' or '21'
    url: str
    content: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    translate_to_urdu: bool = False
    api_key: str
    include_sources: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentSource] = []
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

    def _get_document_folder(self, filepath: str) -> str:
        """Determine if document is from 20 or 21 folder"""
        if "20" in filepath:
            return "20"
        elif "21" in filepath:
            return "21"
        return "unknown"

    def _create_document_url(self, filename: str, folder: str) -> str:
        """Generate the correct document URL with /documents/ prefix"""
        encoded_filename = filename.replace(' ', '%20')
        return f"{BASE_DOCUMENT_URL}/{folder}/{encoded_filename}"

    def process_query(self, question: str, include_sources: bool = True) -> dict:
        # Intent classification
        intent_prompt = ChatPromptTemplate.from_template(
            """Classify the query type:
            - 'DocumentQuestion': Specific questions about document content
            - 'GeneralQuestion': Other valid questions
            - 'Other': Greetings/chit-chat

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

        # Setup retrieval chain
        prompt_template = ChatPromptTemplate.from_template(
            """Answer the question precisely using only this context:
            {context}

            Question: {input}
            Answer in complete sentences with relevant numbers/data:"""
        )
        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        chain = create_retrieval_chain(retriever, document_chain)

        # Execute query
        start_time = time.time()
        response = chain.invoke({"input": question})
        response_time = time.time() - start_time

        # Process sources
        sources = []
        if include_sources:
            unique_sources = set()
            for doc in response["context"]:
                filename = os.path.basename(doc.metadata["source"])
                if filename not in unique_sources:
                    folder = self._get_document_folder(doc.metadata["source"])
                    sources.append({
                        "filename": filename,
                        "folder": folder,
                        "url": self._create_document_url(filename, folder),
                        "content": doc.page_content[:500] + "..."  # First 500 chars
                    })
                    unique_sources.add(filename)

        # Verify grounding
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
            "sources": sources,
            "response_time": response_time,
            "is_grounded": is_grounded
        }

    def translate(self, text: str) -> str:
        return (ChatPromptTemplate.from_template(
            """Translate to Urdu while preserving:
            1. Legal/technical terms
            2. Numbers and dates
            3. Original meaning

            Text: {text}
            Urdu Translation:"""
        ) | self.llm).invoke({"text": text}).content

# --- Initialize Services ---
assistant = DocumentAssistant()

# --- API Routes ---
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not secrets.compare_digest(request.api_key, os.environ['API_KEY']):
        raise HTTPException(status_code=401, detail="Invalid API key")

    result = assistant.process_query(
        request.question,
        include_sources=request.include_sources
    )

    if request.translate_to_urdu:
        result["translated_answer"] = assistant.translate(result["answer"])

    return result

@app.post("/twilio-webhook")
async def handle_whatsapp_message(
    From: str = Form(...),
    Body: str = Form(...)
):
    # Process query
    result = assistant.process_query(Body)

    # Format WhatsApp response
    response = f"📄 Answer:\n{result['answer']}\n\n"
    if result['sources']:
        response += "🔍 Sources:\n"
        for source in result['sources']:
            response += f"- {source['filename']}\n  {source['url']}\n"

    # Send reply
    twilio_client.messages.create(
        body=response,
        from_="whatsapp:+14155238886",  # Twilio sandbox number
        to=From
    )

    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
