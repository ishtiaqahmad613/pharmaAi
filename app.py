# pharma_ai.py - Streamlit app for a smart medical assistant powered by LangChain and Gemini

import streamlit as st
import os
import random
import re
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load and cache the embedding model
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Attempt to correct fuzzy user input based on common medical keywords
def fuzzy_correct(query, terms, threshold=80):
    result = process.extractOne(query, terms, scorer=fuzz.token_sort_ratio)
    if result:
        match, score, _ = result
        return match if score >= threshold else query
    return query

# Keywords to help guide fuzzy matching
all_known_terms = [
    "medicine", "medication", "tablet", "dose", "dosage", "usage", "fever", "pain",
    "cough", "symptom", "treatment", "side effect", "can i", "should i",
    "list of medication", "what medicine", "headache", "vomiting", "asthma", "diarrhea", "acne"
]

# Load and cache the FAISS vector store
@st.cache_resource(show_spinner="🔄 Loading and indexing PDFs...")
def load_vector_store():
    embeddings = get_embeddings()

    if os.path.exists("faiss_index/index.faiss"):
        # Load previously saved FAISS index
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        # Load and split documents if no index exists
        pdf_paths = ["docs/essential_medicines.pdf"]
        all_pages = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for page in pages:
                page.metadata["source"] = os.path.basename(path)
            all_pages.extend(pages)

        # Break documents into smaller chunks for better vector search
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )

        docs = text_splitter.split_documents(all_pages)
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store

# Determine user intent from their question
def detect_intent(question: str) -> str:
    q = question.lower().strip()

    greeting_patterns = [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bwho are you\b"]
    gratitude_patterns = [r"\bthank you\b", r"\bthanks\b", r"\bthx\b"]
    medical_patterns = [
        r"\b(medicine|medicines|medication|tablets?|dose|dosage|usage|fever|pain|cough|symptom|treatment|vomiting|diarrhea|acne|headache)\b",
        r"\b(can i|should i|what medicine|side effects?|take for|list of medication)\b"
    ]

    def match_any(patterns):
        return any(re.search(p, q) for p in patterns)

    if match_any(greeting_patterns):
        return "greeting"
    elif match_any(gratitude_patterns):
        return "gratitude"
    elif match_any(medical_patterns) or len(q.split()) > 3:
        return "medical"
    return "about"

# Prompt template for Gemini responses
custom_prompt = PromptTemplate.from_template('''
You are *PharmaAI*, a highly professional AI assistant trained to provide accurate, safe, and medically relevant information from reliable pharmaceutical references.

Respond based on intent:

1. *Greeting* → "Hello! I'm PharmaAI, your assistant for pharmaceutical and medical queries. Feel free to ask anything."

2. *Gratitude* → Respond warmly like GPT. Randomly choose one of the following:
   - "You're very welcome!"
   - "No problem at all — happy to help!"
   - "Glad I could assist!"
   - "Anytime! Let me know if there's anything else."
   - "You're welcome! Feel free to ask me anything else."

3. *About* → "I'm PharmaAI, designed to provide trustworthy medical insights using reliable sources. Ask me any medicine or health-related question."

4. *Medical* →
   - Start with:
     - "Sure! Here's what I found:"
     - "Absolutely! Here's the information you requested:"
   - Then provide medically relevant content.
   - If context is missing, say:
     "I'm sorry, I couldn't find that information at the moment. Let me know if there's anything else I can assist you with."
   - Add disclaimer:
     - "⚠️ This is general medical information and should not replace advice from a licensed healthcare provider."
     - "📌 Always consult a qualified doctor before starting or stopping any medication."
     - "⚕️ Proper diagnosis is essential before using any treatment for symptoms."

---
Context:
{context}
---
User Question:
{question}
Answer:
''')

# LangChain setup
vector_store = load_vector_store()
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_with_memory = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# Streamlit UI configuration
st.set_page_config(page_title="Pharma AI", layout="wide")
st.title("💊 Pharma AI")
st.markdown("Ask me about any *medicine* or *disease* to get usage, dosage, side effects, and more.")

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'greeted' not in st.session_state:
    st.session_state.greeted = False

# Sidebar controls
with st.sidebar:
    def clear_chat():
        st.session_state.chat_history.clear()
        st.session_state.greeted = False

    st.button("🗑️ Clear Chat", on_click=clear_chat)
    st.markdown("---")

# Display previous conversation
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# User input box
if prompt := st.chat_input("Ask anything about medicine or health..."):
    corrected_prompt = fuzzy_correct(prompt, all_known_terms)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("PharmaAI is thinking..."):
            try:
                intent = detect_intent(corrected_prompt)

                responses = {
                    "greeting": "👋 Hello! I'm PharmaAI, your assistant for pharmaceutical and medical queries. Feel free to ask anything.",
                    "gratitude": random.choice([
                        "🙏 You're very welcome!",
                        "😊 No problem at all — happy to help!",
                        "👍 Glad I could assist!",
                        "🫶 Anytime! Let me know if there's anything else.",
                        "🙌 You're welcome! Feel free to ask me anything else."
                    ]),
                    "about": "ℹ️ I'm PharmaAI, designed to provide trustworthy medical insights using reliable sources. Ask me any medicine or health-related question."
                }

                if intent == "greeting" and not st.session_state.greeted:
                    response = responses["greeting"]
                    st.session_state.greeted = True
                elif intent in responses:
                    response = responses[intent]
                else:
                    result = qa_with_memory({"question": corrected_prompt})
                    response = result["answer"]

            except Exception as e:
                response = "⚠️ Sorry, something went wrong while processing your request. Please try again later."

    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("assistant", response))
    st.markdown(response)

    # Scroll to bottom after each response
    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
