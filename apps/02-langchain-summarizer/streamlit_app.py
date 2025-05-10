import os 
import time
import streamlit as st
from langchain_core.prompts import load_prompt
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

st.title("LangChain Summarizer")

# Path to template.json relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(script_dir,"template.json")

prompt = load_prompt(prompt_path)

# Step 2: Setup Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
chat_model = genai.GenerativeModel("gemini-2.0-flash-lite")


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
    # response = chat_model.generate_content(formatted_prompt)
    response = chat_model.generate_content(formatted_prompt, stream=True)

    # Iterate over each chunk of the response and print as it arrives
    placeholder = st.empty()
    full_response = ""

    for chunk in response:
        if chunk.text:
            full_response += chunk.text
            # Show partial content (use .markdown for formatting)
            placeholder.markdown(full_response)
            time.sleep(0.05)  # Optional: slow typing effect