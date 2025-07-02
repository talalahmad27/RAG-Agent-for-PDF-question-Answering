import streamlit as st
import tempfile
import os
from rag_pipeline import load_and_split_pdfs, create_vectorstore, get_relevant_chunks, generate_answer_with_gemini

st.title("📄 PDF Question Answering - RAG Agent")

uploaded_files = st.file_uploader("Upload PDF file(s)", type="pdf", accept_multiple_files=True)

if "db" not in st.session_state:
    st.session_state.db = None

if uploaded_files and st.session_state.db is None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            file_paths.append(path)

        with st.spinner("Processing documents..."):
            chunks = load_and_split_pdfs(file_paths)
            st.session_state.db = create_vectorstore(chunks)


# Input for questions
if st.session_state.db:
    st.markdown("### ❓ Ask questions about your PDFs")
    query = st.text_input("Enter your question")

    if query:
        with st.spinner("💬 Generating answer..."):
            docs = get_relevant_chunks(st.session_state.db, query)
            answer = generate_answer_with_gemini(query, docs)
            st.markdown("### ✅ Answer")
            st.write(answer)