import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import time
import json

class CustomGroqEmbeddings(Embeddings):
    """Custom implementation of Groq embeddings using direct API calls."""
    
    def __init__(self, api_key, model="llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        
    def embed_documents(self, texts):
        """Embed a list of documents using Groq API."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings
    
    def embed_query(self, text):
        """Embed a query using Groq API."""
        url = "https://api.groq.com/openai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-8b-8192",  # Using same model as chat completions
            "input": text
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Process the response and extract embedding
            data = response.json()
            embedding = data.get("data", [{}])[0].get("embedding", [])
            
            # If no embedding is returned, create a random one (fallback)
            if not embedding:
                embedding = list(np.random.rand(1536))
                
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a random embedding as fallback
            return list(np.random.rand(1536))

class RAGChatbot:
    def __init__(self, groq_api_key):
        """Initialize the RAG Chatbot with necessary components."""
        self.groq_api_key = groq_api_key
        self.documents = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # Set API key for langchain components
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def load_data(self, uploaded_file):
        """Load and process data from uploaded file."""
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Create a temporary file
        with open("temp_file." + file_extension, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load document based on file type
        if file_extension == 'csv':
            loader = CSVLoader(file_path="temp_file.csv")
        elif file_extension == 'pdf':
            loader = PyPDFLoader(file_path="temp_file.pdf")
        elif file_extension in ['txt', 'md']:
            loader = TextLoader(file_path="temp_file." + file_extension)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load documents
        self.documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        split_documents = text_splitter.split_documents(self.documents)
        
        return split_documents

    def setup_rag(self, documents):
        """Set up the RAG pipeline with LangChain."""
        # Create vector store and retriever
        self.vectorstore = FAISS.from_documents(
            documents, 
            CustomGroqEmbeddings(api_key=self.groq_api_key)
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create prompt template
        template = """
        You are a helpful assistant. Use the following context to answer the user's question.
        If you don't know the answer based on the context, just say you don't know.
        
        Context:
        {context}
        
        User Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.call_groq_api_langchain
            | StrOutputParser()
        )
        
        return "RAG pipeline setup complete!"

    def call_groq_api(self, messages: list, max_retries: int = 3) -> str:
        """Call Groq API with robust error handling and retry mechanism."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                response_data = response.json()
                answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return answer
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    return f"Error calling Groq API after {max_retries} attempts: {str(e)}"
    
    def call_groq_api_langchain(self, prompt):
        """Convert LangChain prompt to messages and call the Groq API."""
        # Extract content from prompt
        if hasattr(prompt, "content"):
            content = prompt.content
        else:
            content = str(prompt)
            
        # Create a message for the API
        messages = [{"role": "user", "content": content}]
        
        # Call the API and return the response
        return self.call_groq_api(messages)
    
    def chat(self, user_input):
        """Process user input through the RAG pipeline and return a response."""
        if not self.rag_chain:
            return "Please upload a document and set up the RAG pipeline first."
        
        response = self.rag_chain.invoke(user_input)
        return response
    
    def save_chat_history(self, filename="chat_history.txt"):
        """Save the chat history to a file."""
        with open(filename, "w") as f:
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                f.write(f"{role}: {content}\n\n")
        return filename


# Streamlit App
def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
    
    st.title("📚 RAG Chatbot with Groq API")
    
    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a document (CSV, PDF, TXT):", type=["csv", "pdf", "txt"])
        
        # Initialize chatbot
        if groq_api_key:
            if "chatbot" not in st.session_state:
                st.session_state.chatbot = RAGChatbot(groq_api_key)
                st.success("Chatbot initialized")
            
            # Process uploaded file
            if uploaded_file is not None:
                if "file_processed" not in st.session_state or st.session_state.file_processed != uploaded_file.name:
                    with st.spinner("Processing document..."):
                        try:
                            documents = st.session_state.chatbot.load_data(uploaded_file)
                            setup_message = st.session_state.chatbot.setup_rag(documents)
                            st.session_state.file_processed = uploaded_file.name
                            st.success(f"Document processed: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
        
        # Save chat history
        if st.button("Save Chat History"):
            if "chatbot" in st.session_state and "messages" in st.session_state:
                filename = st.session_state.chatbot.save_chat_history()
                st.download_button(
                    label="Download Chat History",
                    data=open(filename, "rb").read(),
                    file_name="rag_chatbot_history.txt",
                    mime="text/plain"
                )
    
    # Main chat interface
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask a question based on the uploaded document:"):
        if "chatbot" not in st.session_state:
            st.error("Please enter your Groq API key in the sidebar.")
        elif not st.session_state.chatbot.rag_chain:
            st.error("Please upload a document first.")
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.chat(user_input)
                    st.markdown(response)
            
            # Add assistant message to chat
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()