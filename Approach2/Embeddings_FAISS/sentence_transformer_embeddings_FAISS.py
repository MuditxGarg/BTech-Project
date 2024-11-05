import os
import uuid
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import faiss
import numpy as np

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings_FAISS/Chroma_SentenceTransformer"
FAISS_PATH = "./Embeddings_FAISS/faiss_SentenceTransformer.index"
EMBEDDING_DIM = 384  # Sentence-Transformer embedding dimension
MAX_BATCH_SIZE = 256

# Sentence-Transformer embeddings
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        return self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)

# Load and split documents
def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                doc = Document(page_content=text, metadata={"source": filename})
                documents.append(doc)
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Save chunks and embeddings to Chroma and create FAISS index
def save_to_chroma_faiss(chunks, embedder):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)

    # Initialize FAISS index
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)

    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        documents = [chunk.page_content for chunk in batch]
        embeddings = embedder.embed_documents(documents)
        metadatas = [chunk.metadata for chunk in batch]
        ids = [str(uuid.uuid4()) for _ in range(len(batch))]

        # Save embeddings to Chroma
        db._collection.upsert(embeddings=embeddings, metadatas=metadatas, documents=documents, ids=ids)

        # Add embeddings to FAISS index
        faiss_index.add(embeddings)

    # Save FAISS index to disk
    faiss.write_index(faiss_index, FAISS_PATH)
    print(f"Saved FAISS index to {FAISS_PATH}.")

# Main function
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    sentence_transformer_embedder = SentenceTransformerEmbeddings()
    save_to_chroma_faiss(chunks, sentence_transformer_embedder)

if __name__ == "__main__":
    generate_data_store()
