import os
import torch
import sys
from transformers import AutoTokenizer, AutoModel, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, MT5ForConditionalGeneration, T5Tokenizer
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# Constants
CHROMA_PATH = "./Embeddings/Chroma_DPR"
DPR_MODEL_NAME = "facebook/dpr-question_encoder-single-nq-base"
MT5_MODEL_NAME = "google/mt5-base"

# Set standard output encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Prompt template for RAG
PROMPT_TEMPLATE = """
Below are the contexts relevant to the question:
{context}
- -
Question: {question}
Answer:
"""

# DPR Retriever setup
class DPRRetriever:
    def __init__(self, model_name=DPR_MODEL_NAME):
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name)

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.pooler_output.cpu().numpy()
        return query_embedding

# Load Chroma as vector store
chroma = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=None  # Embeddings are already stored
)

# Set up the retriever
retriever = DPRRetriever()

# mT5 Generator setup for FiD with simplified prompt
class MT5FiDGenerator:
    def __init__(self, model_name=MT5_MODEL_NAME):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, input_texts, question, max_length=512):
        # Use only the most relevant context (top 1 or top 2)
        context = "\n\n".join(input_texts[:2])  # Limiting the number of contexts
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

generator = MT5FiDGenerator()

# Retrieve and Generate Response using FiD approach with updated prompt
def retrieve_and_generate(query, top_k=3):
    # Step 1: Retrieve relevant documents using DPR
    query_embedding = retriever.embed_query(query)
    retrieved_docs = chroma.similarity_search_by_vector(query_embedding, k=top_k)

    # Step 2: Check if there are any matching results
    if not retrieved_docs:
        return "No relevant documents found for the query."

    # Step 3: Extract content from retrieved documents
    contexts = [doc.page_content for doc in retrieved_docs]

    # Debug: Print retrieved contexts
    print("Retrieved Contexts:")
    for idx, context in enumerate(contexts):
        try:
            print(f"Context {idx + 1}:\n{context}\n")
        except UnicodeEncodeError:
            # Print with UTF-8 encoding or replace problematic characters
            print(f"Context {idx + 1}:\n{context.encode('utf-8', errors='ignore').decode('utf-8')}\n")

    # Step 4: Generate response using FiD approach with mT5
    response = generator.generate(input_texts=contexts, question=query)

    return response

# Example usage
if __name__ == "__main__":
    query = "Describe how to use the Amul AMCS Application?"
    response = retrieve_and_generate(query)
    print(response)
