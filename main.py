
# main.py

import streamlit as st
import os
from rag_engine import load_txt_to_faiss, query_sop_docs
from utils import query_llm_openrouter
from dotenv import load_dotenv

st.set_page_config(page_title="Threat Intelligence Assistant", layout="centered")
st.title("🔐 Threat Intelligence Assistant (FAISS + LLM)")

st.sidebar.header("Upload SOPs")
uploaded_file = st.sidebar.file_uploader("Upload SOP (.txt only)", type=["txt"])

load_dotenv()
API_KEY = os.getenv("API_KEY")
REFERER_URL = "https://sahanagenai.streamlit.app"

# Directory to store SOPs
os.makedirs("sops", exist_ok=True)

if uploaded_file:
    file_path = os.path.join("sops", uploaded_file.name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(uploaded_file.read().decode("utf-8"))
    load_txt_to_faiss(file_path)
    st.sidebar.success("Uploaded and indexed!")

st.header("Ask about your SOPs 📄")

user_query = st.text_input("Enter your question")
if st.button("Ask") and user_query:
    with st.spinner("Retrieving..."):
        context = query_sop_docs(user_query)
        full_prompt = f"Refer to this SOP content and answer: \n\n{context}\n\nQuestion: {user_query}"
        answer = query_llm_openrouter(full_prompt, API_KEY, REFERER_URL)
        st.success("Answer:")
        st.write(answer)
