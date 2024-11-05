import os
import shutil
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Constants
PASSAGES_FILE = './passages.txt'  # Path to the file with processed passages
CHROMA_PATH = './Embeddings/Chroma'  # Local storage path for Chroma
MAX_BATCH_SIZE = 5461  # Maximum batch size allowed by Chroma
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model

# Function to load passages from 'passages.txt' into Document objects
def load_documents():
    documents = []
    if not os.path.exists(PASSAGES_FILE):
        raise FileNotFoundError(f"Passages file '{PASSAGES_FILE}' not found.")

    with open(PASSAGES_FILE, 'r', encoding='utf-8') as f:
        passages = [line.strip() for line in f.readlines()]

    for idx, text in enumerate(passages):
        doc = Document(page_content=text, metadata={"source": f"passage_{idx}"})
        documents.append(doc)

    print(f"Loaded {len(documents)} documents from {PASSAGES_FILE}.")
    return documents

# Split the documents into chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# Save chunks to Chroma in batches
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize HuggingFace embeddings
    hf_embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    db = None

    # Process and save chunks in batches
    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]

        # Create a new Chroma database or add to the existing one
        if db is None:
            db = Chroma.from_documents(
                batch,
                hf_embedder,
                persist_directory=CHROMA_PATH
            )
        else:
            db.add_documents(batch)

        # Persist the database after each batch
        db.persist()
        print(f"Saved batch {i // MAX_BATCH_SIZE + 1} to {CHROMA_PATH}.")

    print(f"Total saved chunks: {len(chunks)} to {CHROMA_PATH}.")

# Main function to generate the data store
def generate_data_store():
    documents = load_documents()  # Load passages as documents
    chunks = split_text(documents)  # Split into smaller chunks
    save_to_chroma(chunks)  # Save to Chroma with embeddings

if __name__ == "__main__":
    generate_data_store()
