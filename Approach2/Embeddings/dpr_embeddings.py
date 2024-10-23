import os
import uuid
import shutil
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings/Chroma_DPR"
MAX_BATCH_SIZE = 256
DPR_CONTEXT_MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"

# DPR Context Encoder setup for document embedding
class DPRContextEmbeddings:
    def __init__(self, model_name=DPR_CONTEXT_MODEL_NAME):
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRContextEncoder.from_pretrained(model_name)

    def embed_documents(self, documents):
        inputs = self.tokenizer(documents, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy()  # Use pooler_output for embeddings
        return embeddings

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

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks, embedder):
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
    embedder = DPRContextEmbeddings()
    save_to_chroma(chunks, embedder)

if __name__ == "__main__":
    generate_data_store()
