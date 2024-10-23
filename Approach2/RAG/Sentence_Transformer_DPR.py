import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings import BaseEmbeddings
from langchain.retrievers import DenseRetriever
from langchain.llms import MT5

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings/Chroma_SentenceTransformer"

# Load Sentence-Transformers model
class SentenceTransformerEmbeddings(BaseEmbeddings):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

# Load Chroma as vector store
chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)

# DPR Retriever setup
embedder = SentenceTransformerEmbeddings()
retriever = DenseRetriever(vectorstore=chroma, embedding_function=embedder.embed_text)

# mT5 Generator setup
generator = MT5(model_name="google/mt5-small")

# Retrieve and Generate Response
def retrieve_and_generate(query):
    # Embed the query
    query_embedding = embedder.embed_text(query)
    
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query_embedding)
    
    # Generate response
    context = " ".join([doc.page_content for doc in retrieved_docs])
    response = generator.generate(input_text=context)
    
    return response

# Example usage
query = "How does the Amul logistic app help with milk collection?"
response = retrieve_and_generate(query)
print(response)
