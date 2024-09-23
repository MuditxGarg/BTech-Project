import os
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Constants
CHROMA_PATH = "./Chroma_IndicBert"

# Initialize HuggingFace Embeddings using IndicBERTv2
def initialize_indicbert_embeddings():
    model_name = "ai4bharat/IndicBERTv2-MLM-only"
    # Initialize HuggingFaceEmbeddings properly
    return HuggingFaceEmbeddings(model_name=model_name)

# Prompt template for RAG
PROMPT_TEMPLATE = """
Answer the following question based only on the provided context:
{context}
- - 
Question: {question}
Answer:
"""

# Retrieval and context assembly
def retrieve_context(query_text, db):
    # Using Chroma's similarity search to retrieve relevant context
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return None

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
    return context_text

# Generate answer using mT5
def generate_answer_with_mt5(contexts, query_text):
    model_name = "google/mt5-large"  # mT5 for generation
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # Prepare prompt with context and query
    prompt = f"Context: {contexts}\n\nQuestion: {query_text}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the response using mT5
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response_text

# Main RAG Query Function
def query_rag(query_text):
    # Initialize IndicBERT HuggingFace embeddings
    indicbert_embedder = initialize_indicbert_embeddings()

    # Load the pre-populated Chroma DB with proper embedding function
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=indicbert_embedder)

    # Retrieve relevant context
    contexts = retrieve_context(query_text, db)

    if contexts:
        # Generate response with mT5
        response = generate_answer_with_mt5(contexts, query_text)
        return response
    return "No relevant context found."

def main():
    # Example query
    query_text = "BovEasy detects pregnancy in"
    
    # Call the RAG query function
    response = query_rag(query_text)
    
    # Print the final response
    if response:
        print(f"Response:\n{response}")
    else:
        print("Could not find relevant content.")

if __name__ == "__main__":
    main()
