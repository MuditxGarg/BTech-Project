import torch
from transformers import AutoModel, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Constants
CHROMA_PATH = './Embeddings/Chroma'
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize MiniLM model and tokenizer for query
def load_minilm_model():
    model = AutoModel.from_pretrained(MINILM_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL_NAME)
    return model, tokenizer

# Query Chroma for relevant passages using MiniLM embeddings
def query_chroma(query_text):
    minilm_model, minilm_tokenizer = load_minilm_model()

    # Tokenize and encode the query using MiniLM
    inputs = minilm_tokenizer(query_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = minilm_model(**inputs)
        query_embedding = outputs.pooler_output.cpu().numpy()

    db = Chroma(persist_directory=CHROMA_PATH)

    # Search Chroma using MiniLM embeddings
    results = db.similarity_search_by_vector(query_embedding, k=5)

    if len(results) == 0:
        print("No matching results found.")
        return None

    retrieved_passages = [doc.page_content for doc in results]

    return retrieved_passages
