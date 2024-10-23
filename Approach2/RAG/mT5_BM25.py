import os
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_chroma import Chroma
from langchain.llms import HuggingFaceHub

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings/Chroma_mT5"

# Load mT5 model and tokenizer
class MT5Embeddings:
    def __init__(self, model_name="google/mt5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, documents):
        # Handle both single and list of documents
        if isinstance(documents, list):
            texts = documents
        else:
            texts = [documents]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().tolist()  # Convert to list for compatibility

    def embed_query(self, query):
        return self.embed_documents(query)

# Instantiate the MT5 embedder
embedder = MT5Embeddings()

# Load Chroma as vector store
chroma = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedder  # Providing the embedding function
)

# Define a BM25 Retriever to interact with the Chroma vector store
class CustomBM25Retriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_relevant_documents(self, query):
        return self.vectorstore.similarity_search(query)

# Set up the custom BM25 retriever
retriever = CustomBM25Retriever(vectorstore=chroma)

# mT5 Generator setup using HuggingFace Hub as an LLM interface
class MT5Generator:
    def __init__(self, model_name="google/mt5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=256)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

generator = MT5Generator()

# Retrieve and Generate Response
def retrieve_and_generate(query):
    # Retrieve relevant documents using BM25
    retrieved_docs = retriever.get_relevant_documents(query)

    # Generate response
    context = " ".join([doc.page_content for doc in retrieved_docs])
    response = generator.generate(input_text=context)

    return response

# Example usage
if __name__ == "__main__":
    query = "How does the Amul logistic app help with milk collection?"
    response = retrieve_and_generate(query)
    print(response)
