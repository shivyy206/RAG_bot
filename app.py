# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["USER_AGENT"] = "Mozilla/5.0"

# Load vectorstore
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1.0)

# QA Chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("Drake")
query = st.text_input("Ask me anything regarding brakes. I am bhery smart")

if query:
    with st.spinner("Thinking...Good question"):
        response = qa.run(query)
        st.success(response)
