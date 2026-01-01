import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# ---------------- UI ----------------
st.title("📄 Traditional PDF RAG — Chat with PDF")
st.set_page_config(page_title="PDF RAG App", layout="centered")

uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
question = st.text_input("Ask a question from the PDF")

if uploaded_pdf:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    st.success("PDF uploaded successfully!")

# ---------------- LOAD PDF ----------------
loader = PyPDFLoader("temp.pdf")
docs = loader.load()

transcript = "\n".join(doc.page_content for doc in docs)

# ---------------- SPLIT ----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ---------------- PROMPT ----------------
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, say "I don't know."

    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

# ---------------- LLM MODEL ----------------
# Replace with your HuggingFace token & model
model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta", 
        task="text-generation", 
        temperature=0.2
    )
)

main_chain = parallel_chain | prompt | model | parser

# ---------------- QUESTION ----------------
with st.spinner("Thinking..."):
    answer = main_chain.invoke(question)

st.subheader("Answer")
st.write(answer)


