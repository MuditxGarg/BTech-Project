import os
import uuid
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings/Chroma_mT5"
MAX_BATCH_SIZE = 256

# mT5 model setup
class MT5Embeddings:
    def __init__(self, model_name="google/mt5-small"):
        # Use the slow tokenizer by setting use_fast=False to avoid unknown tokens issue
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False) 
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, documents):
        # Tokenize the documents and provide inputs for the encoder only
        inputs = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Ensure that inputs are on the same device as the model (if using GPU, it will move tensors to the correct device)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            # Only using encoder outputs for embeddings (ignoring the decoder part)
            outputs = self.model.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings.cpu().numpy()

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                doc = Document(page_content=text, metadata={"source": filename})
                documents.append(doc)
    print(f"Loaded {len(documents)} documents from {DATA_PATH}.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document], embedder):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)

    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        documents = [chunk.page_content for chunk in batch]
        embeddings = embedder.embed_documents(documents)
        metadatas = [chunk.metadata for chunk in batch]
        ids = [str(uuid.uuid4()) for _ in range(len(batch))]

        db._collection.upsert(embeddings=embeddings, metadatas=metadatas, documents=documents, ids=ids)

    print(f"Total saved chunks: {len(chunks)} to {CHROMA_PATH}.")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    embedder = MT5Embeddings()
    save_to_chroma(chunks, embedder)

if __name__ == "__main__":
    generate_data_store()
