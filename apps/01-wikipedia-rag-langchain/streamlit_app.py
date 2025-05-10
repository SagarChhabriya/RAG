from langchain_community.document_loaders import WikipediaLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import streamlit as st  
from dotenv import load_dotenv



load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Step 1: Set Retriever

def load_docs(query: str, lang: str = "en", load_max_docs: int = 3):

    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    return loader.load()

# Cached docs
@st.cache_data(show_spinner=False)
def load_docs_cached(query: str, lang: str = "en", load_max_docs: int = 3):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    return loader.load()



    
# Step 2: Create and Store Embeddings

def create_vectorstore(docs):

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    return vector_store


# Step 3: RAG Chain

def create_qa_chain(vector_store):

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

    qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever, return_source_documents=True)

    return qa_chain


# Step 4: Streamlit UI

st.title("Wikipedia RAG Assistant")

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
user_input = st.chat_input("Say something...")

# Handle input and response
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    # Generate reply
    bot_reply = f"You said: {user_input}"
    try:
        docs = load_docs_cached(user_input)
        vector_store = create_vectorstore(docs)
        qa_chain = create_qa_chain(vector_store)
        response = qa_chain.invoke({"query":user_input})
        bot_reply = response["result"]


        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        
        with st.expander("Show sources"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content)
    except Exception as e:
        st.error(e.args)


