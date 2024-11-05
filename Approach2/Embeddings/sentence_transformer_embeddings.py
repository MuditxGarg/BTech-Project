import os
import uuid
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings/Chroma_SentenceTransformer"
MAX_BATCH_SIZE = 256  # Adjust based on system capacity

# Load Sentence-Transformers model
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
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
    embedder = SentenceTransformerEmbeddings()
    save_to_chroma(chunks, embedder)

if __name__ == "__main__":
    generate_data_store()
