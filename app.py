# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (azadmrinaal@gmail.com)

import argparse

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Configuration
OLLAMA_MODEL = "llama3.1"
CHROMA_DB_PATH = "./chroma_db"


def initialize_embeddings():
    """Initialize HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def initialize_vectorstore():
    """Initialize Chroma vectorstore with existing data"""
    embeddings = initialize_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():
    """Create the RAG pipeline using LangChain"""

    # Initialize LLM
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    # Initialize vectorstore and retriever
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create prompt template
    template = """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer: Provide a detailed answer based on the context above. If the answer cannot be found in the context, say "I cannot find this information in the provided documents."
"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def run_cli():
    """Run the CLI version of the PDF chat"""
    print("=" * 70)
    print("PDF Chat Assistant (CLI Mode)")
    print("=" * 70)
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Vector DB: {CHROMA_DB_PATH}")
    print("=" * 70)
    print("Initializing RAG pipeline...")

    try:
        rag_chain = create_rag_chain()
        print("RAG pipeline initialized successfully!")
        print("=" * 70)
        print("\nType 'exit' or 'quit' to end the conversation.")
        print("Type 'clear' to clear the screen.")
        print("=" * 70)

        while True:
            print()
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break

            if question.lower() == 'clear':
                print("\033[2J\033[H")  # Clear screen
                continue

            print("\nAssistant: ", end="", flush=True)

            try:
                response = rag_chain.invoke(question)
                print(response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {str(e)}")
        exit(1)


def run_streamlit():
    """Run the Streamlit UI version"""
    pass
    # st.set_page_config(page_title="PDF Chat with RAG")

    # st.title("PDF Chat Assistant")
    # st.markdown("Ask questions about your PDF documents using RAG (Retrieval-Augmented Generation)")

    # # Initialize RAG chain
    # if 'rag_chain' not in st.session_state:
    #     with st.spinner("Initializing RAG pipeline..."):
    #         st.session_state.rag_chain = create_rag_chain()
    #     st.success("RAG pipeline initialized successfully!")

    # # Initialize chat history
    # if 'messages' not in st.session_state:
    #     st.session_state.messages = []

    # # Display chat history
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # # Chat input
    # if question := st.chat_input("Ask a question about your documents..."):
    #     # Add user message to chat history
    #     st.session_state.messages.append({"role": "user", "content": question})

    #     # Display user message
    #     with st.chat_message("user"):
    #         st.markdown(question)

    #     # Generate response
    #     with st.chat_message("assistant"):
    #         with st.spinner("Thinking..."):
    #             try:
    #                 response = st.session_state.rag_chain.invoke(question)
    #                 st.markdown(response)

    #                 # Add assistant response to chat history
    #                 st.session_state.messages.append({"role": "assistant", "content": response})
    #             except Exception as e:
    #                 error_msg = f"Error: {str(e)}"
    #                 st.error(error_msg)
    #                 st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # # Sidebar
    # with st.sidebar:
    #     st.header("ℹ️ Information")
    #     st.markdown(f"**Model:** {OLLAMA_MODEL}")
    #     st.markdown(f"**Vector DB:** ChromaDB")
    #     st.markdown(f"**Embeddings:** all-MiniLM-L6-v2")

    #     if st.button("Clear Chat History"):
    #         st.session_state.messages = []
    #         st.rerun()

    #     st.markdown("---")
    #     # st.markdown("Made with love using LangChain & Streamlit")


def _main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PDF Chat Assistant with RAG")
    parser.add_argument(
        '--streamlit',
        action='store_true',
        help='Run with Streamlit UI (default: CLI mode)'
    )

    args = parser.parse_args()

    if args.streamlit:
        run_streamlit()
    else:
        run_cli()


if __name__ == "__main__":
    _main()