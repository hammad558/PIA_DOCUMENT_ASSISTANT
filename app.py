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
st.set_page_config(
    page_title="PIA Document Assistant",
    page_icon="images.png",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Dark Green Theme CSS ---
st.markdown("""
    <style>
        :root {
            --primary-dark: #004d26;
            --primary: #006633;
            --primary-light: #008040;
            --secondary: #99cc00;
            --accent: #ffcc00;
            --text-light: #f0f0f0;
            --text-dark: #333333;
            --bg-dark: #002211;
            --bg-medium: #003322;
            --bg-light: #e6f2ed;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .stApp {
            background-color: var(--bg-dark);
            color: var(--text-light);
        }

        .sidebar .sidebar-content {
            background-color: var(--primary-dark) !important;
            color: var(--text-light);
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--secondary) !important;
        }

        .stTextArea>div>div>textarea, .stTextInput>div>div>input {
            background-color: var(--bg-medium) !important;
            color: var(--text-light) !important;
            border: 1px solid var(--primary-light) !important;
        }

        .stButton>button {
            background-color: var(--primary) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: bold !important;
            transition: all 0.3s !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }

        .stButton>button:hover {
            background-color: var(--primary-light) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        }

        textarea, input {
            caret-color: var(--accent) !important;
        }

        .stMarkdown {
            color: var(--text-light) !important;
        }

        .st-expander {
            background-color: var(--bg-medium) !important;
            border: 1px solid var(--primary-light) !important;
            border-radius: 8px !important;
        }

        .stAlert {
            background-color: var(--primary-dark) !important;
            border-left: 4px solid var(--secondary) !important;
        }

        .stProgress > div > div > div > div {
            background-color: var(--secondary) !important;
        }

        .css-1aumxhk {
            background-color: var(--primary-dark) !important;
            color: var(--text-light) !important;
        }

        /* Download button styling */
        .stDownloadButton>button {
            background-color: var(--accent) !important;
            color: var(--text-dark) !important;
            margin: 8px 0 !important;
            width: 100% !important;
            text-align: center !important;
            padding: 12px 15px !important;
            border-radius: 6px !important;
            font-size: 1em !important;
            font-weight: bold !important;
            border: none !important;
        }

        .stDownloadButton>button:hover {
            background-color: var(--secondary) !important;
            color: white !important;
        }

        /* Document container styling */
        .document-container {
            margin: 10px 0;
            padding: 15px;
            background-color: var(--bg-medium);
            border-radius: 8px;
            border-left: 4px solid var(--accent);
        }

        .document-name {
            font-weight: bold;
            color: var(--text-light);
            margin-bottom: 5px;
            word-break: break-word;
            font-size: 1.1em;
        }

        .document-url {
            font-size: 0.9em;
            color: var(--secondary);
            margin-bottom: 10px;
            word-break: break-all;
        }

        .document-meta {
            font-size: 0.85em;
            color: var(--accent);
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Environment Variables and API Keys ---
load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')
google_api_key = os.environ.get('GOOGLE_API_KEY')
if not groq_api_key or not google_api_key:
    st.error("API keys for Groq and Google are not set. Please check your .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = google_api_key

# --- Header Section ---
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='color: var(--secondary); margin-bottom: 0;'>PIA Document Assistant</h1>
        <p style='color: var(--text-light); margin-top: 0;'>Your intelligent document analysis companion</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Initialize LLM and Session State ---
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="llama3-8b-8192")

for key in ["last_response", "translated_answer", "response_time", "vectors", "context_docs"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Backend Functions ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def vector_embedding(force_recreate=False):
    FAISS_INDEX_PATH = "faiss_vector_store"

    if force_recreate and os.path.exists(FAISS_INDEX_PATH):
        import shutil
        shutil.rmtree(FAISS_INDEX_PATH)
        st.session_state.vectors = None
        st.sidebar.success("Cleared existing vector store cache")

    if st.session_state.vectors is None:
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model='models/embedding-001')

            if os.path.exists(FAISS_INDEX_PATH) and not force_recreate:
                with st.spinner("Loading existing Vector Store..."):
                    st.session_state.vectors = FAISS.load_local(
                        FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
                    st.sidebar.success("Vector Store loaded successfully!")
            else:
                with st.spinner("Creating new Vector Store..."):
                    # Check both 20 and 21 folders
                    pdf_folders = ["./20", "./21"]
                    all_docs = []

                    for folder in pdf_folders:
                        if os.path.isdir(folder) and os.listdir(folder):
                            loader = PyPDFDirectoryLoader(folder)
                            docs = loader.load()
                            if docs:
                                all_docs.extend(docs)
                            else:
                                st.sidebar.warning(f"No PDFs found in {folder} folder")

                    if not all_docs:
                        st.sidebar.error("No PDF documents found in either 20 or 21 folders")
                        return

                    if any(len(doc.page_content.strip()) == 0 for doc in all_docs):
                        st.sidebar.warning("Some PDFs may contain no extractable text. Consider using OCR for scanned documents.")

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800, chunk_overlap=200)
                    final_documents = text_splitter.split_documents(all_docs)
                    st.session_state.vectors = FAISS.from_documents(
                        final_documents, embeddings_model, normalize_L2=True)

                    # Save vector store
                    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
                    st.session_state.vectors.save_local(FAISS_INDEX_PATH)
                    st.sidebar.success("New Vector Store created and saved successfully!")
        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")
    else:
        st.sidebar.success("Vector Store is already loaded.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div style='background-color: var(--primary-dark); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: var(--secondary); text-align: center;'>Document Processing</h3>
        </div>
    """, unsafe_allow_html=True)

    force_recreate = st.checkbox("Force recreate vector store (ignore cache)", value=False)
    if st.button("🔄 Create/Update Vector Store", key="vector_button"):
        vector_embedding(force_recreate=force_recreate)

    if st.session_state.vectors:
        st.success("✅ Documents Loaded & Ready")

    st.markdown("---")

    st.markdown("""
        <div style='background-color: var(--primary-dark); padding: 15px; border-radius: 10px;'>
            <h3 style='color: var(--secondary); text-align: center;'>Document Status</h3>
        </div>
    """, unsafe_allow_html=True)

    # Check both 20 and 21 folders
    pdf_folders = ["./20", "./21"]
    total_pdfs = 0

    for folder in pdf_folders:
        if os.path.exists(folder):
            pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
            if pdf_files:
                total_pdfs += len(pdf_files)
                st.success(f"Found {len(pdf_files)} PDF documents in {folder}:")
                for pdf in pdf_files:
                    st.markdown(f"""
                        <div style='background-color: var(--bg-medium); padding: 8px; border-radius: 5px; margin: 5px 0;'>
                            📄 {pdf}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"No PDFs found in {folder} folder")
        else:
            st.warning(f"Folder {folder} does not exist")

    if total_pdfs == 0:
        st.error("No PDFs found in either 20 or 21 folders")

    st.markdown("---")

    st.markdown("""
        <div style='background-color: var(--primary-dark); padding: 15px; border-radius: 10px;'>
            <h3 style='color: var(--secondary); text-align: center;'>Instructions</h3>
            <ol style='color: var(--text-light);'>
                <li>Place PDFs in either the <code>20</code> or <code>21</code> folder</li>
                <li>Click the button above to process</li>
                <li>Ask any question!</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# --- Main Content Area ---
st.markdown("""
    <div style='background-color: var(--primary-dark); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: var(--secondary); text-align: center;'>Ask About Your Documents</h2>
    </div>
""", unsafe_allow_html=True)

prompt1 = st.text_area(
    "Enter your question here:",
    height=150,
    placeholder="Type your question about the documents here...",
    key="question_input",
    help="Ask detailed questions about the content of your uploaded documents"
)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    submit = st.button("🔍 Get Answer", key="submit_button", type="primary")
with col2:
    translate = st.button("🌐 Translate to Urdu", key="translate_button")
with col3:
    clear = st.button("🗑️ Clear", key="clear_button")

if clear:
    st.session_state.last_response = None
    st.session_state.translated_answer = None
    st.experimental_rerun()

# --- Answering Logic ---
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
                """You are a legal document assistant. Provide accurate, clear responses based on the context.
                If you don't know the answer, say you couldn't find relevant information in the documents.

                Context:
                {context}

                Question: {input}

                Answer:"""
            )
            with st.spinner("Searching documents and generating answer..."):
                try:
                    document_chain = create_stuff_documents_chain(
                        st.session_state.llm, prompt_template)
                    retriever = st.session_state.vectors.as_retriever(
                        search_kwargs={"k": 10, "score_threshold": 0.8})
                    retrieval_chain = create_retrieval_chain(
                        retriever, document_chain)

                    start_time = time.time()
                    response = retrieval_chain.invoke({"input": prompt1})
                    response_time = time.time() - start_time

                    st.session_state.last_response = response.get(
                        'answer', "No answer found.")
                    st.session_state.context_docs = response.get("context", [])
                    st.session_state.response_time = f"{response_time:.2f}"

                except Exception as e:
                    st.error(
                        f"An error occurred while generating the answer: {str(e)}")

        else:
            st.session_state.last_response = "Please ask a complete question about your documents for the best assistance."

# --- Translation Logic ---
if translate and st.session_state.last_response:
    with st.spinner("Translating to Urdu..."):
        try:
            translation_prompt = ChatPromptTemplate.from_template(
                """Translate the following English legal text to Urdu while maintaining:
                1. All legal terminology accuracy
                2. Formal tone
                3. Complete meaning

                Text: {text}

                Urdu Translation:"""
            )
            translation_chain = translation_prompt | st.session_state.llm
            translated_text = translation_chain.invoke(
                {"text": st.session_state.last_response}).content
            st.session_state.translated_answer = translated_text
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")

# --- Displaying Results ---
if st.session_state.last_response:
    st.markdown("---")

    # Response Card
    st.markdown(f"""
        <div style='background-color: var(--bg-medium); padding: 20px; border-radius: 10px; border-left: 5px solid var(--secondary); margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <h3 style='color: var(--secondary); margin: 0;'>Answer</h3>
                <span style='color: var(--accent); font-size: 0.9em;'>Response time: {st.session_state.response_time} seconds</span>
            </div>
            <div style='color: var(--text-light); padding: 10px; background-color: var(--primary-dark); border-radius: 5px;'>
                {st.session_state.last_response}
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.context_docs:
        # Source References Section
        st.markdown("""
            <div style='background-color: var(--primary-dark); padding: 15px; border-radius: 10px; margin: 20px 0;'>
                <h3 style='color: var(--secondary); text-align: center;'>Source References</h3>
            </div>
        """, unsafe_allow_html=True)

        displayed_files = set()
        for doc in st.session_state.context_docs:
            filename = os.path.basename(doc.metadata.get("source", ""))
            if filename and filename not in displayed_files:
                # Determine which folder the document came from (20 or 21)
                folder = "20" if "20" in doc.metadata.get("source", "") else "21"
                file_path = os.path.join(f"./{folder}", filename)

                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        file_data = f.read()

                    # Create document URL
                    doc_url = f"https://piacconnect.piac.com.pk/documents/{folder}/{filename.replace(' ', '%20')}"

                    # Custom document container
                    with st.container():
                        st.markdown(f"""
                            <div class="document-container">
                                <div class="document-name">{filename}</div>
                                <div class="document-url">
                                    <a href="{doc_url}" target="_blank">{doc_url}</a>
                                </div>
                                <div class="document-meta">Source: {folder} folder</div>
                        """, unsafe_allow_html=True)

                        st.download_button(
                            label="⬇️ DOWNLOAD DOCUMENT",
                            data=file_data,
                            file_name=filename,
                            mime="application/pdf",
                            key=f"download_{filename}_{time.time()}",
                            help=f"Download {filename}"
                        )

                        st.markdown("</div>", unsafe_allow_html=True)

                    displayed_files.add(filename)

        # Document Chunks Section
        with st.expander("📑 View Relevant Document Chunks", expanded=False):
            for i, doc in enumerate(st.session_state.context_docs):
                st.markdown(f"""
                    <div style='background-color: var(--bg-medium); padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                        <h4 style='color: var(--secondary); margin-top: 0;'>Chunk {i+1} from <code>{os.path.basename(doc.metadata.get('source', 'Unknown'))}</code></h4>
                        <div style='color: var(--text-light); background-color: var(--primary-dark); padding: 10px; border-radius: 5px;'>
                            {doc.page_content}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

if st.session_state.translated_answer:
    st.markdown("---")
    st.markdown(f"""
        <div style='background-color: var(--bg-medium); padding: 20px; border-radius: 10px; border-left: 5px solid var(--accent); margin-bottom: 20px;'>
            <h3 style='color: var(--accent); margin-top: 0;'>Urdu Translation</h3>
            <div style='color: var(--text-light); padding: 10px; background-color: var(--primary-dark); border-radius: 5px; direction: rtl; text-align: right; font-size: 1.1em;'>
                {st.session_state.translated_answer}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: var(--text-light); font-size: 0.9em; padding: 10px;'>
        PIA Document Assistant  • Pakistan International Airlines • Great People To Fly With
    </div>
""", unsafe_allow_html=True)
