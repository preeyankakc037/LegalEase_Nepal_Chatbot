import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


DOCS_DIR = r"C:\Users\kgu95\Desktop\RAG\legal_docs"  # absolute path to directory
INDEX_PATH = "faiss_index"
EMBEDDINGS_PATH = "embeddings.pkl"

def ingest_documents():
   
    os.makedirs(DOCS_DIR, exist_ok=True)
    
   
    loader = DirectoryLoader(DOCS_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_PATH)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    print(f"Index saved to {INDEX_PATH}")

if __name__ == "__main__":
    ingest_documents()



