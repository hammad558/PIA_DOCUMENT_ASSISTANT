import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Page Configuration ---
st.set_page_config(page_title="PIA ChatBot", page_icon="⚖️", layout="centered")

# --- Load Environment Variables and API Keys ---
load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')
google_api_key = os.environ.get('GOOGLE_API_KEY')
if not groq_api_key or not google_api_key:
    st.error("API keys for Groq and Google are not set. Please check your .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = google_api_key

# --- UI Styling ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { background-color: #FFFFFF; }
        h1, h2, h3 { text-align: center; }
        .stButton>button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Main Title ---
st.markdown("""
    <h1 style='text-align: center;'>PIA Document Assistant</h1>
    <p style='text-align: center;'>Ask questions about your documents and get answers instantly.</p>
""", unsafe_allow_html=True)

# --- Initialize LLM and Session State ---
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

for key in ["last_response", "translated_answer", "response_time", "vectors", "context_docs"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Backend Functions ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def vector_embedding():
    FAISS_INDEX_PATH = "faiss_vector_store"
    if st.session_state.vectors is None:
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model='models/embedding-001')
            if os.path.exists(FAISS_INDEX_PATH):
                with st.spinner("Loading existing Vector Store..."):
                    st.session_state.vectors = FAISS.load_local(
                        FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
                    st.success("Vector Store loaded successfully!")
            else:
                with st.spinner("Creating new Vector Store..."):
                    pdf_folder = "./static"
                    if not os.path.isdir(pdf_folder) or not os.listdir(pdf_folder):
                        st.sidebar.error(
                            f"The '{pdf_folder}' directory is empty or does not exist.")
                        return
                    loader = PyPDFDirectoryLoader(pdf_folder)
                    docs = loader.load()
                    if not docs:
                        st.sidebar.warning(
                            "No PDF documents were found to process.")
                        return
                    # Basic validation of document content
                    if any(len(doc.page_content.strip()) == 0 for doc in docs):
                        st.sidebar.warning("Some PDFs may contain no extractable text. Consider using OCR for scanned documents.")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800, chunk_overlap=200)
                    final_documents = text_splitter.split_documents(docs)
                    st.session_state.vectors = FAISS.from_documents(
                        final_documents, embeddings_model, normalize_L2=True)
                    st.session_state.vectors.save_local(FAISS_INDEX_PATH)
                    st.success(
                        "New Vector Store created and saved successfully!")
        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")
    else:
        st.success("Vector Store is already loaded.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Navigation")
    if st.button("📂 Create Vector Store"):
        vector_embedding()

    if st.session_state.vectors:
        st.sidebar.success("✅ Documents Loaded & Ready")

    st.markdown("---")
    st.markdown("### ℹ️ Instructions")
    st.info("""
    1. Place PDFs in the `./static` folder.
    2. Click the button above to process.
    3. Ask any question!
    """)

# --- Main Content Area ---
st.markdown("---")
st.markdown("### 📝 Ask Your Question")
prompt1 = st.text_area("🔎 Type your legal question below:",
                       height=150, label_visibility="collapsed")

col1, col2 = st.columns(2)
with col1:
    submit = st.button("✅ Get Answer")
with col2:
    translate = st.button("🌐 Translate to Urdu")

# --- Answering Logic with Intent Classifier ---
if submit and prompt1:
    st.session_state.last_response = None
    st.session_state.context_docs = None
    st.session_state.translated_answer = None

    if st.session_state.vectors is None:
        st.error(
            "Please create the vector store first using the button in the sidebar.")
    else:
        intent_prompt = ChatPromptTemplate.from_template(
            "You are a query classifier. Your task is to determine if the user's input is a request for information from a document.\n"
            "- If the input is a question, a command to summarize, or a topic to find information about, classify it as 'Informational'.\n"
            "- For anything else, such as greetings, single vague words, or conversational chit-chat, classify it as 'Other'.\n"
            "Respond with only the category name: 'Informational' or 'Other'.\n\nUser Input: {user_input}"
        )
        intent_chain = intent_prompt | st.session_state.llm
        with st.spinner("Analyzing query..."):
            intent = intent_chain.invoke(
                {"user_input": prompt1}).content.strip()

        if intent == "Informational":
            prompt_template = ChatPromptTemplate.from_template(
                "Provide a response based on the available context. If no relevant information is found, state that no matching content was identified in the documents.\n<context>{context}</context>\nQuestion: {input}")
            with st.spinner("Searching documents and generating answer..."):
                try:
                    document_chain = create_stuff_documents_chain(
                        st.session_state.llm, prompt_template)
                    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10, "score_threshold": 0.8})
                    retrieval_chain = create_retrieval_chain(
                        retriever, document_chain)

                    start_time = time.process_time()
                    response = retrieval_chain.invoke({"input": prompt1})
                    response_time = time.process_time() - start_time

                    st.session_state.last_response = response.get(
                        'answer', "No answer found.")
                    st.session_state.context_docs = response.get("context", [])
                    st.session_state.response_time = f"{response_time:.2f}"

                except Exception as e:
                    st.error(
                        f"An error occurred while generating the answer: {e}")

        else:
            st.session_state.last_response = "It seems you've entered a keyword or an incomplete phrase. Please ask a complete question about your documents."

# --- TRANSLATION LOGIC ---
if translate and st.session_state.last_response:
    with st.spinner("Translating to Urdu..."):
        try:
            translation_prompt = ChatPromptTemplate.from_template(
                "Translate the following English text to Urdu: {text}")
            translation_chain = translation_prompt | st.session_state.llm
            translated_text = translation_chain.invoke(
                {"text": st.session_state.last_response}).content
            st.session_state.translated_answer = translated_text
        except Exception as e:
            st.error(f"Translation failed: {e}")

# --- DISPLAYING RESULTS ---
if st.session_state.last_response:
    st.markdown("---")
    if st.session_state.response_time:
        st.info(f"**Response time:** {st.session_state.response_time} seconds")
    st.subheader("📌 Answer")
    st.write(st.session_state.last_response)

    if st.session_state.context_docs:
        st.markdown("---")
        st.subheader("📚 Source References")
        displayed_files = set()
        for doc in st.session_state.context_docs:
            filename = os.path.basename(doc.metadata.get("source", ""))
            if filename and filename not in displayed_files:
                file_path = os.path.join("./static", filename)
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label=f"Download {filename}",
                        data=file_data,
                        file_name=filename,
                        mime="application/pdf",
                        key=f"download_{filename}_{time.time()}"
                    )
                    displayed_files.add(filename)

        with st.expander("📄 View Relevant Document Chunks"):
            for i, doc in enumerate(st.session_state.context_docs):
                st.markdown(
                    f"**Chunk {i+1} from `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`**")
                st.write(doc.page_content)
                st.markdown("---")

if st.session_state.translated_answer:
    st.markdown("---")
    st.success("Translation Complete!")
    st.write("**🌍 Translated Answer (Urdu):**")
    st.write(st.session_state.translated_answer)