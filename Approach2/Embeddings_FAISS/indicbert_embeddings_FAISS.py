import os
import uuid
import shutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import faiss
import numpy as np

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings_FAISS/Chroma_IndicBert"
FAISS_PATH = "./Embeddings_FAISS/faiss_IndicBert.index"
EMBEDDING_DIM = 768  # IndicBERT embedding dimension
MAX_BATCH_SIZE = 256

# IndicBERT embeddings
class IndicBERTEmbeddings:
    def __init__(self, model_name="ai4bharat/IndicBERTv2-MLM-only"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def embed_documents(self, documents):
        inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states[:, 0, :]  # [CLS] token embeddings
        return embeddings.cpu().numpy()

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
    indicbert_embedder = IndicBERTEmbeddings()
    save_to_chroma_faiss(chunks, indicbert_embedder)

if __name__ == "__main__":
    generate_data_store()
