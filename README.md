# PDF Chat with RAG Pipeline

A simple implementation of a Retrieval-Augmented Generation (RAG) pipeline for conversational AI over PDF documents. This project demonstrates practical application of Large Language Models (LLMs), vector databases, and document retrieval systems.

## Overview

This project implements a complete RAG system that allows users to have contextual conversations with their PDF documents. The system uses local LLMs via Ollama, efficient document embeddings, and vector similarity search to provide accurate, context-aware responses.

### Key Features

- **Local LLM Integration**: Uses Ollama with Llama 3.1 (8B parameter model) for privacy and cost-efficiency
- **Efficient Document Processing**: Implements chunking strategies with overlap for optimal retrieval
- **Vector Store**: ChromaDB for persistent storage and fast similarity search
- **Dual Interface**: Both CLI and Streamlit web UI for different use cases
- **Async Support**: Asynchronous operations for better performance
- **Production-Ready**: Error handling, logging, and modular architecture

## Technical Architecture

### RAG Pipeline Components

1. **Document Ingestion**
   - PDF parsing with PyPDFLoader
   - Recursive text splitting
   - Batch processing for multiple documents

2. **Embedding Generation**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Fast inference with HuggingFace integration

3. **Vector Storage**
   - ChromaDB for persistent vector storage
   - Metadata preservation for source tracking

4. **LLM Integration**
   - Ollama with Llama 3.1 (8B parameters, ~4.7GB)
   - Temperature: 0 for deterministic responses
   - Context-aware prompting with retrieval augmentation

5. **Response Generation**
   - LangChain orchestration
   - Context formatting and prompt engineering
   - Streaming support for real-time responses

## Prerequisites

- Python 3.12+ (tested on Python 3.12.3 in WSL2 Ubuntu on Windows 11)
- 5GB+ disk space (for model storage)

## Setup

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com/)

### 2. Pull the LLM Model

```sh
# This pulls the 8B parameter Llama 3.1 model (approx 4.7GB)
ollama pull llama3.1
```

**Note**: Using the 8B model for local deployment. For production use cases, consider larger models like 70B for improved accuracy.

### 3. Install Python Dependencies

```sh
pip install -r requirements.txt
```

**Dependencies include**:
- `langchain` and `langchain-community`: RAG pipeline orchestration
- `langchain-ollama`: Ollama integration
- `langchain-chroma`: Vector store integration
- `langchain-huggingface`: Embedding models
- `chromadb`: Vector database
- `streamlit`: Web UI framework
- `pypdf`: PDF processing

### 4. Verify Installation

Test the Ollama connection:

```sh
python tests/test_ollama.py
```

Expected output: A joke response from the LLM with full response metadata.

## Usage

### 1. Ingest PDF Documents

Add PDFs to the vector store:

```sh
# Single file
python ingest.py --file path/to/document.pdf
```

The script will:
- Load and parse the PDF
- Split into optimized chunks
- Generate embeddings
- Store in ChromaDB at `./chroma_db`

### 2. Chat Interface

#### CLI Mode

Terminal-based interactive chat:

```sh
python app.py
```

**Commands**:
- Type your question and press Enter
- `exit` or `quit`: End session
- `clear`: Clear screen

#### Streamlit Web UI

Browser-based chat interface:

```sh
streamlit run app.py -- --streamlit
```

**Features**:
- Chat history persistence
- Markdown rendering
- Sidebar with system information
- Clear conversation button

Access the UI at `http://localhost:8501`


## License

This is a personal learning project. Feel free to use and modify for educational purposes.

## Author

Mrinaal Dogra ([mrinaald](https://github.com/mrinaald))

---

**Note**: This project prioritizes local execution and privacy. All processing happens on your machine with no external API calls (except for HuggingFace model downloads).
