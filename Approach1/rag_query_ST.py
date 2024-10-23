import os
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma

# Constants
CHROMA_PATH = "./Chroma_Multilingual_2"

# Initialize Sentence Transformer embeddings
def initialize_sentence_transformer_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return SentenceTransformer(model_name)

# Prompt template for RAG
PROMPT_TEMPLATE = """
Answer the following question based only on the provided context:
{context}
- - 
Question: {question}
Answer:
"""

# Retrieval and context assembly
def retrieve_context(query_text, db, embedder):
    # Embed the query using SentenceTransformer
    query_embedding = embedder.encode([query_text], convert_to_tensor=True)
    
    # Perform the similarity search using precomputed embeddings
    results = db.similarity_search_with_relevance_scores(query_embedding, k=3)

    if len(results) == 0:
        print("No results found.")
        return None

    # Normalize the relevance scores to be between 0 and 1
    normalized_results = [(doc, (score + 1) / 2) for doc, score in results]

    # Filter out results with low relevance scores (e.g., below 0.5)
    relevant_results = [doc for doc, score in normalized_results if score >= 0.5]

    if len(relevant_results) == 0:
        print("Unable to find matching results with sufficient relevance.")
        return None

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc in relevant_results])
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
    # Initialize SentenceTransformer embeddings
    sentence_transformer_embedder = initialize_sentence_transformer_embeddings()

    # Load the pre-populated Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)  # No need to recompute embeddings

    # Retrieve relevant context
    contexts = retrieve_context(query_text, db, sentence_transformer_embedder)

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
