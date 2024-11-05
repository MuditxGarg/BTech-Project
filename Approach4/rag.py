import sys
import os
import torch
import faiss
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, MT5ForConditionalGeneration, MT5Tokenizer

# Workaround to allow multiple OpenMP runtime libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fix for Unicode error in printing
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Constants
INDEX_PATH = './faiss_index'
TOP_K = 5  # Number of top results to retrieve
CHUNK_DIR = './processed_chunks'

# Initialize mT5 model and tokenizer
mt5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
mt5_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

# Load DPR Question Encoder for Query Embedding
q_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# Load FAISS Index
index = faiss.read_index(INDEX_PATH)

def query_faiss(query):
    inputs = q_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        query_embedding = q_encoder(**inputs).pooler_output.cpu().numpy()

    # Search the FAISS index for the top_k closest matches
    distances, indices = index.search(query_embedding, TOP_K)
    return indices

def retrieve_passages(indices):
    retrieved_passages = []
    for idx in indices[0]:
        file_path = os.path.join(CHUNK_DIR, f"chunk_{idx}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                retrieved_passages.append(f.read().strip())
        else:
            print(f"File {file_path} not found.")
    return retrieved_passages

def query_pipeline(query):
    indices = query_faiss(query)
    passages = retrieve_passages(indices)
    return passages

def generate_answer(passages, question):
    context = " ".join(passages)
    input_text = f"Question: {question}\nContext: {context}"
    
    inputs = mt5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    output = mt5_model.generate(**inputs, max_length=150)
    answer = mt5_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer

if __name__ == "__main__":
    # Sample query
    query_text = "How do I adjust the temperature on ThawEasy Lite?"
    
    # Retrieve passages
    results = query_pipeline(query_text)
    print("Retrieved Passages:")
    for passage in results:
        print(passage)
    
    # Generate the answer
    answer = generate_answer(results, query_text)
    print(f"Generated Answer: {answer}")
