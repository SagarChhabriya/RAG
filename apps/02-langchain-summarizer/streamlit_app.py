import os 
import streamlit as st
from langchain_core.prompts import load_prompt
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

st.title("LangChain Summarizer")

# Step 1: Load the prompt template
prompt = load_prompt("template.json")

# Step 2: Setup Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
chat_model = genai.GenerativeModel("gemini-1.5-flash")

# Step 3: Streamlit UI
paper = st.selectbox("Select Paper", ["Attention is All you Need", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"])
style = st.selectbox("Select Style", ["Soft", "Hard"])
length = st.selectbox("Select Length", ["Short", "Long"])

# Step 4: Format the prompt manually
formatted_prompt = prompt.format(
    paper_input=paper,
    style_input=style,
    length_input=length
)

# Step 5: Send to Gemini
if st.button("Summarize"):
    response = chat_model.generate_content(formatted_prompt)

    # Step 6: Display result
    st.write(response.text)
