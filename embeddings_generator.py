import os
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import numpy as np
import faiss  # Library used for nearest neighbor search

# Constants
CHUNK_DIR = './processed_chunks'
INDEX_PATH = './faiss_index'

# Initialize DPR Model and Tokenizer
model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output.cpu().numpy()
    return embeddings

def create_embeddings():
    all_embeddings = []

    for filename in os.listdir(CHUNK_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(CHUNK_DIR, filename), 'r', encoding='utf-8') as f:
                chunks = f.readlines()
            
            # Generate embeddings for each chunk
            for chunk in chunks:
                chunk_embedding = generate_embeddings(chunk)
                all_embeddings.append(chunk_embedding)

    # Convert to FAISS Index
    embeddings_np = np.vstack(all_embeddings)
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f'FAISS index saved at {INDEX_PATH}')

if __name__ == "__main__":
    create_embeddings()
