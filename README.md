# RAG Chatbot with Groq API

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit and the Groq API. The chatbot can load external data, set up a RAG pipeline using LangChain, and answer user questions based on the uploaded documents.

## Features

- Upload various document formats (CSV, PDF, TXT)
- Process documents with LangChain
- Create embeddings and vector storage with FAISS
- Retrieve relevant context for user questions
- Generate responses using Groq's llama3-8b-8192 model
- Save chat history for later reference

## Installation

1. Clone this repository:
```
git clone https://github.com/Devarajan8/Rag-App
cd Rag-App
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run chat.py
```

## Usage

1. Enter your Groq API key in the sidebar
2. Upload a document (CSV, PDF, or TXT)
3. Wait for the document to be processed
4. Ask questions related to the uploaded document
5. Save the chat history if needed

## Components

- **RAGChatbot**: Main class that handles document loading, RAG setup, and API calls
- **Document Loading**: Supports CSV, PDF, and text documents
- **Text Splitting**: Chunks documents for better retrieval
- **Vector Storage**: Uses FAISS for efficient similarity search with custom Groq embeddings
- **RAG Pipeline**: Combines retrieval and generation with LangChain
- **API Integration**: Custom Groq API implementation with error handling and retries

## Requirements

See `requirements.txt` for the full list of dependencies.

## Assignment Deliverables

This project fulfills the following tasks from the AI Intern Assignment:

1. **Data Loading**: Implemented in the `load_data` method, supporting various file formats
2. **RAG Setup with LangChain**: Implemented in the `setup_rag` method
3. **Chatbot Building**: Implemented as a Streamlit application with the RAG pipeline

The code includes proper documentation, and the application allows saving chat histories for review.


## Notes

- Make sure to keep your Groq API key secure
- For large documents, processing may take some time
- The chatbot will only answer questions based on the information in the uploaded documents
