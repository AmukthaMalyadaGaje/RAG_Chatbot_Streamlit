import os
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    JSONLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_document_loader(file_path: str):
    """Get the appropriate document loader based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()

    loaders = {
        '.csv': CSVLoader,
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.json': JSONLoader,
    }

    if file_extension not in loaders:
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension == '.csv':
            return CSVLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path)
        elif file_extension == '.json':
            return JSONLoader(file_path)
    except Exception as e:
        raise ValueError(f"Error loading {file_extension} file: {str(e)}")

def process_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Process multiple documents and return their contents."""
    documents = []
    processed_files = []
    failed_files = []

    for file_path in file_paths:
        try:
            loader = get_document_loader(file_path)
            doc = loader.load()

            # Additional validation for PDF files
            if file_path.lower().endswith('.pdf'):
                if not doc or len(doc) == 0:
                    failed_files.append((file_path, "PDF file appears to be empty or corrupted"))
                    continue
                # Check if the content is meaningful
                if all(len(page.page_content.strip()) < 10 for page in doc):
                    failed_files.append((file_path, "PDF content appears to be invalid or corrupted"))
                    continue

            if doc:  # Only add if document was successfully loaded
                documents.extend(doc)
                processed_files.append(file_path)
            else:
                failed_files.append((file_path, "Empty document"))
        except Exception as e:
            failed_files.append((file_path, str(e)))
            continue

    if not documents:
        raise ValueError("No valid documents were processed. Please check your files.")

    return documents, processed_files, failed_files

def create_text_splitter():
    """Create a text splitter for chunking documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

def setup_rag_chain(documents: List[Dict[str, Any]]):
    """Set up the RAG chain with FAISS vector store and Groq."""
    if not documents:
        raise ValueError("No documents provided for RAG setup")

    # Initialize text splitter
    text_splitter = create_text_splitter()

    # Split documents
    splits = text_splitter.split_documents(documents)

    if not splits:
        raise ValueError("No text chunks were created from the documents")

    # Initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize embeddings: {str(e)}")

    # Create vector store
    try:
        vectorstore = FAISS.from_documents(splits, embeddings)
    except Exception as e:
        raise ValueError(f"Failed to create vector store: {str(e)}")

    # Initialize Groq chat model
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    llm = ChatGroq(
        api_key=api_key,
        model_name="gemma2-9b-it",
        temperature=0.7,
    )

    # Initialize memory with explicit output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    # Create RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
    )

    return chain

def get_file_summary(file_path: str) -> str:
    """Generate a brief summary of the file contents."""
    try:
        loader = get_document_loader(file_path)
        documents = loader.load()

        if not documents:
            return f"File: {os.path.basename(file_path)}\nType: {os.path.splitext(file_path)[1]}\nStatus: Empty file"

        # For PDF files, include number of pages
        if file_path.lower().endswith('.pdf'):
            return f"File: {os.path.basename(file_path)}\n" \
                   f"Type: PDF\n" \
                   f"Number of pages: {len(documents)}\n" \
                   f"Number of chunks: {len(documents)}"

        # For other files
        return f"File: {os.path.basename(file_path)}\n" \
               f"Type: {os.path.splitext(file_path)[1]}\n" \
               f"Number of chunks: {len(documents)}"
    except Exception as e:
        return f"Error processing file: {str(e)}"