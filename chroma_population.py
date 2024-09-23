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
CHROMA_PATH = "./Chroma_IndicBert"
MAX_BATCH_SIZE = 256  # Adjust based on system capacity

# Load IndicBERTv2 embedding model and tokenizer from Hugging Face
class IndicBERTEmbeddings:
    def __init__(self, model_name="ai4bharat/IndicBERTv2-MLM-only"):
        # Load the tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def embed_documents(self, documents):
        # Tokenize and embed the documents using IndicBERT
        inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)  # Get the model outputs
            hidden_states = outputs.hidden_states[-1]  # Extract the last hidden layer
            embeddings = hidden_states[:, 0, :]  # Extract [CLS] token embeddings
        return embeddings.numpy()

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
        chunk_size=200,  # Adjust chunk size to fit memory
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# Save chunks and embeddings to Chroma in batches
def save_to_chroma(chunks: list[Document], indicbert_embedder):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = None

    # Process and save chunks in smaller batches
    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        
        # Get the text content from each document chunk
        documents = [chunk.page_content for chunk in batch]

        # Embed the batch using IndicBERT embeddings
        embeddings = indicbert_embedder.embed_documents(documents)

        # Prepare metadata (e.g., sources) for each document
        metadatas = [chunk.metadata for chunk in batch]

        # Generate unique IDs for each document chunk
        ids = [str(uuid.uuid4()) for _ in range(len(batch))]

        # Create a new Chroma database or add to the existing one
        if db is None:
            db = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=None  # Explicitly set embedding_function to None
            )
        
        # Add precomputed embeddings along with texts, metadata, and unique IDs to Chroma
        db._collection.upsert(
            embeddings=embeddings, 
            metadatas=metadatas, 
            documents=documents,
            ids=ids  # Pass the generated IDs
        )

        # Persist the database after each batch
        db.persist()
        print(f"Saved batch {i // MAX_BATCH_SIZE + 1} to {CHROMA_PATH}.")

    print(f"Total saved chunks: {len(chunks)} to {CHROMA_PATH}.")

# Main function to generate the data store
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)

    # Initialize IndicBERT embedding model
    indicbert_embedder = IndicBERTEmbeddings()

    save_to_chroma(chunks, indicbert_embedder)

if __name__ == "__main__":
    generate_data_store()
