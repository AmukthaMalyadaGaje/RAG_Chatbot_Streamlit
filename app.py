import streamlit as st
import tempfile
import os
from utils import process_documents, setup_rag_chain, get_file_summary
from typing import List, Dict, Any
import warnings
from datetime import datetime

# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "failed_files" not in st.session_state:
    st.session_state.failed_files = []

def save_uploaded_files(uploaded_files: List[Any]) -> List[str]:
    """Save uploaded files to temporary directory and return their paths."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    return file_paths

def save_to_responses_file(question: str, answer: str, processed_files: List[str]):
    """Save the question, answer, and file information to responses.txt."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    files_info = ", ".join([os.path.basename(f) for f in processed_files])

    with open("responses.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Files: {files_info}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"{'='*80}\n")

def main():
    st.title("🤖 RAG Chatbot")
    st.markdown("Upload your documents and chat with them!")

    # File upload section
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose your files",
            type=["pdf", "txt", "csv", "json"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        # Save uploaded files
                        file_paths = save_uploaded_files(uploaded_files)
                        st.session_state.uploaded_files = file_paths

                        # Process documents
                        documents, processed_files, failed_files = process_documents(file_paths)
                        st.session_state.processed_files = processed_files
                        st.session_state.failed_files = failed_files

                        # Setup RAG chain
                        st.session_state.rag_chain = setup_rag_chain(documents)

                        st.success("Documents processed successfully!")

                        # Display file summaries
                        st.subheader("📄 File Summaries")
                        for file_path in processed_files:
                            st.text(get_file_summary(file_path))

                        # Display failed files if any
                        if failed_files:
                            st.warning("Some files failed to process:")
                            for file_path, error in failed_files:
                                st.error(f"{os.path.basename(file_path)}: {error}")

                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        st.session_state.rag_chain = None

        # Display processed files
        if st.session_state.processed_files:
            st.subheader("✅ Processed Files")
            for file_path in st.session_state.processed_files:
                st.text(f"✓ {os.path.basename(file_path)}")

        # Reset button
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.session_state.rag_chain = None
            st.session_state.uploaded_files = []
            st.session_state.processed_files = []
            st.session_state.failed_files = []
            st.rerun()

    # Chat interface
    st.header("💬 Chat")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        if st.session_state.rag_chain:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_chain.invoke({"question": prompt})
                        answer = response["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

                        # Save to responses.txt
                        save_to_responses_file(
                            prompt,
                            answer,
                            st.session_state.processed_files
                        )

                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
        else:
            with st.chat_message("assistant"):
                st.warning("Please upload and process documents first!")

if __name__ == "__main__":
    main()