import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
import pickle

# Hard-code your API key here
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Configuration
DOCS_DIR = "legal_docs"
INDEX_PATH = "faiss_index"
EMBEDDINGS_PATH = "embeddings.pkl"
def ingest_documents():
    # Create directory if it doesn't exist
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    # Load PDF documents from directory
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Create embeddings using HuggingFace model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the index and embeddings
    vectorstore.save_local(INDEX_PATH)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    print(f"Index saved to {INDEX_PATH}")

if __name__ == "__main__":
    ingest_documents()



