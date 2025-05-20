# RAG Chatbot

A professional RAG-based chatbot application that allows users to upload custom datasets and chat with them using LangChain, Groq API, and Streamlit.

## Features

- 📁 Upload multiple file types (CSV, PDF, TXT, JSON)
- 🤖 Chat interface with document-based responses
- 🔍 RAG implementation using LangChain and Groq
- 📊 Document processing and summarization
- 💾 Chat history management
- 🔄 Reset functionality

## Prerequisites

- Python 3.8+
- Groq API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload your documents using the sidebar interface

4. Click "Process Documents" to initialize the RAG system

5. Start chatting with your documents!

## Supported File Types

- PDF (.pdf)
- Text (.txt)
- CSV (.csv)
- JSON (.json)

## Notes

- The application uses FAISS for vector storage
- Document chunks are created using RecursiveCharacterTextSplitter
- The gemma2-9b-it model from Groq is used for generation
- All uploaded files are temporarily stored and processed in memory

