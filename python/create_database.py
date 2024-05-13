import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHROMA_PATH = "chroma"
DATA_PATH = "data/example"
API_KEY = os.getenv('OPENAI_API_KEY')

def main():
    if API_KEY is None:
        logging.error("OPENAI_API_KEY is not set. Please set the environment variable.")
        return
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    try:
        loader = DirectoryLoader(DATA_PATH, glob="*.txt")
        documents = loader.load()
        return documents
    except Exception as e:
        logging.error(f"Failed to load documents: {e}")
        return []

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if not chunks:
        logging.info("No chunks to process.")
        return
    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    try:
        
        db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
        db.persist()
        logging.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    except Exception as e:
        logging.error(f"Failed to save chunks to Chroma: {e}")

if __name__ == "__main__":
    main()
