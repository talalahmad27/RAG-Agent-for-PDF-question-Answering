
# 📄 RAG Agent for PDF Question Answering (LangChain + Gemini)

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer user questions based on the contents of uploaded PDF files using Google's Gemini language model.

## 🚀 Features

- 📚 Load and parse PDF documents using `LangChain` and `PyPDFLoader`
- 🔍 Chunk and embed documents with `sentence-transformers` and `FAISS`
- 💡 Retrieve relevant chunks from the document for a given user query
- 🤖 Use `ChatGoogleGenerativeAI` (Gemini 2.0) to generate detailed answers based on context
- 🔐 Secure API usage via external `.txt` file

## 🧰 Tech Stack

- Python 3.10+
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Gemini (Google Generative AI)](https://ai.google.dev/)


## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/talalahmad27/RAG-Agent-for-PDF-question-Answering.git
   cd RAG-Agent-for-PDF-question-Answering
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your Gemini API key**:
   - Save your key to a file named `gemini_api_key.txt` in the root directory.

## ▶️ Usage

```python
from pdf_rag import load_and_split_pdfs, create_vectorstore, get_relevant_chunks, generate_answer_with_gemini

# Load and split documents
chunks = load_and_split_pdfs(["sample.pdf"])

# Create vector store
db = create_vectorstore(chunks)

# Retrieve relevant chunks
docs = get_relevant_chunks(db, "What is the summary of the report?")

# Generate answer using Gemini
answer = generate_answer_with_gemini("What is the summary of the report?", docs)
print(answer)
```

## 🔒 API Key Note

Make sure you **DO NOT upload or share** your `your_api_key_here.txt` file. It should be added to your `.gitignore` if you're using version control.

