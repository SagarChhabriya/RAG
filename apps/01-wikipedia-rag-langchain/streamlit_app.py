import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Cached Wikipedia loader
@st.cache_data(show_spinner=False)
def load_docs_cached(query: str, lang: str = "en", load_max_docs: int = 3):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    return loader.load()

# Vectorstore builder
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedding=embeddings)

# Gemini query generator
def generate_gemini_answer(context: str, question: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a helpful assistant. Based on the following context, answer the question.

Context:
{context}

Question: {question}
"""
    response = model.generate_content(prompt)
    return response.text

# UI Logic
st.title(" Wikipedia RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about any topic", disabled=st.session_state.is_generating)

if user_input:
    st.session_state.is_generating = True
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        docs = load_docs_cached(user_input)
        vector_store = create_vectorstore(docs)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(user_input)

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        bot_reply = generate_gemini_answer(context, user_input)

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

        with st.expander("Show sources"):
            for i, doc in enumerate(retrieved_docs):
                url = doc.metadata.get("source") or f"https://en.wikipedia.org/wiki/{doc.metadata.get('title', '').replace(' ', '_')}"
                st.markdown(f"[Source {i+1}]({url})")

    except Exception as e:
        st.error(f"❌ Error: {e}")

    st.session_state.is_generating = False
