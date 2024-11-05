import os
import uuid
import shutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

# Constants
DATA_PATH = "./Extracted_Data_3"
CHROMA_PATH = "./Chroma_IndicBert_2"
MAX_BATCH_SIZE = 256  # Adjust based on system capacity

# Load IndicBERTv2 embedding model and tokenizer from Hugging Face
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
        return embeddings.numpy()

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Save chunks and embeddings to Chroma
def save_to_chroma(chunks, indicbert_embedder):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = None
    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        documents = [chunk.page_content for chunk in batch]
        embeddings = indicbert_embedder.embed_documents(documents)
        metadatas = [chunk.metadata for chunk in batch]
        ids = [str(uuid.uuid4()) for _ in range(len(batch))]

        if db is None:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)

        db._collection.upsert(embeddings=embeddings, metadatas=metadatas, documents=documents, ids=ids)
        db.persist()

# Main function to generate the data store
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    indicbert_embedder = IndicBERTEmbeddings()
    save_to_chroma(chunks, indicbert_embedder)

if __name__ == "__main__":
    generate_data_store()