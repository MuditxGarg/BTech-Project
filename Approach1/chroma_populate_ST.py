import os
import uuid
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

# Constants
DATA_PATH = "./Extracted_Data_3"
CHROMA_PATH = "./Chroma_Multilingual_2"
MAX_BATCH_SIZE = 256  # Adjust based on system capacity

# Load Sentence-Transformers model
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # Load the sentence-transformers model
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        # Embed the documents using SentenceTransformer
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

# Load documents from .txt files
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

# Split the documents into chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Adjust chunk size to fit memory
        chunk_overlap=60,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# Save chunks and embeddings to Chroma in batches
def save_to_chroma(chunks: list[Document], embedder):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize Chroma vector store
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=None  # Explicitly set embedding_function to None, since embeddings are precomputed
    )

    # Process and save chunks in smaller batches
    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        
        # Get the text content from each document chunk
        documents = [chunk.page_content for chunk in batch]

        # Embed the batch using SentenceTransformer embeddings
        embeddings = embedder.embed_documents(documents)

        # Prepare metadata (e.g., sources) for each document
        metadatas = [chunk.metadata for chunk in batch]

        # Generate unique IDs for each document chunk
        ids = [str(uuid.uuid4()) for _ in range(len(batch))]

        # Add precomputed embeddings along with texts, metadata, and unique IDs to Chroma
        db._collection.upsert(
            embeddings=embeddings, 
            metadatas=metadatas, 
            documents=documents,
            ids=ids  # Pass the generated IDs
        )

    print(f"Total saved chunks: {len(chunks)} to {CHROMA_PATH}.")

# Main function to generate the data store
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)

    # Initialize Sentence-Transformers embedding model
    embedder = SentenceTransformerEmbeddings()

    save_to_chroma(chunks, embedder)

if __name__ == "__main__":
    generate_data_store()
