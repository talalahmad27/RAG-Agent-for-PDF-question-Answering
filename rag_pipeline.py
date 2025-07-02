from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import requests

# Embedder model
embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def load_and_split_pdfs(file_paths):
    documents = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks):
    db = FAISS.from_documents(chunks, embedder)
    return db

def get_relevant_chunks(db, query, k=4):
    retriever = db.as_retriever(search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)
    return results


def generate_answer_with_gemini(question, docs):
    
    # Setting up API Key 
    with open(os.path.join(script_dir, "your_api_key_here.txt"), "r") as f:
        GOOGLE_API_KEY = f.read().strip()
    # Set up the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following context to answer the question. Answer helpfully.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )

    # Initialize the ChatGoogleGenerativeAI model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Prepare the context from the documents
    context = "\n".join([doc.page_content for doc in docs])

    # Run the chain to generate the answer
    answer = chain.run({"context": context, "question": question})
    return answer
