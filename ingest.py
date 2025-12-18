# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (azadmrinaal@gmail.com)

import argparse
import asyncio
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def initialize_chroma():
    return Chroma(persist_directory="./chroma_db")


async def _main():
    parser = argparse.ArgumentParser(description="Ingest PDF into vector store")
    parser.add_argument(
        '--file',
        type=str,
        default="",
        help='The path to the PDF file',
    )
    parser.add_argument(
        '--dir',
        type=str,
        default="",
        help='The path to the directory containing multiple PDF file',
    )

    args = parser.parse_args()

    if not args.file and not args.dir:
        print("Please provide a PDF file path using --file")
        exit(1)

    if not os.path.exists(args.file):
        print(f"Provided file path [{args.file}] does not exist")
        exit(1)

    # Load PDF
    loader = PyPDFLoader(args.file)
    documents = await loader.aload()

    print(f"Loaded {len(documents)} documents from PDF")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Initialize embeddings and Chroma
    embeddings = initialize_embeddings()

    vectorstore = initialize_chroma()

    # Store in Chroma DB
    ids = await vectorstore.aadd_documents(documents=splits, embedding=embeddings)

    print(f"Processed {len(splits)} document chunks and stored in Chroma DB")


if __name__ == "__main__":
    asyncio.run(_main())