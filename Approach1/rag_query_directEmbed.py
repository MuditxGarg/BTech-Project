import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import sentencepiece  # Ensure sentencepiece is imported

# Constants
CHROMA_PATH = "./Chroma_Multilingual_2"
RELEVANCE_THRESHOLD = 0.3  # Adjusted threshold for more lenient relevance filtering

# Initialize HuggingFace Embeddings using sentence-transformers model
def initialize_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

# Prompt template for RAG
PROMPT_TEMPLATE = """
Answer the following question based only on the provided context:
{context}
- - -
Question: {question}
Answer:
"""

# Function to normalize scores and ensure they fit between 0 and 1
def normalize_scores(results):
    normalized_results = []
    for doc, score in results:
        # If scores are out of range, adjust them manually (i.e., make them between 0 and 1)
        if score < -1:
            score = -1
        if score > 1:
            score = 1
        normalized_score = (score + 1) / 2
        normalized_results.append((doc, normalized_score))
    return normalized_results

# Function to retrieve context and generate response
def query_rag(query_text):
    try:
        # Initialize the embedding function
        embedding_function = initialize_embeddings()

        # Initialize Chroma database with the embedding function
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Retrieve relevant context from Chroma using similarity search
        results = db.similarity_search_with_relevance_scores(query_text, k=3)

        if len(results) == 0:
            print("No results found for the query.")
            return None, None

        # Normalize the relevance scores to be between 0 and 1
        normalized_results = normalize_scores(results)

        # Filter out results with low relevance scores based on the adjusted threshold
        relevant_results = [doc for doc, score in normalized_results if score >= RELEVANCE_THRESHOLD]

        if len(relevant_results) == 0:
            print("No results with sufficient relevance found.")
            return None, None

        # Combine context from matching documents
        context_text = "\n\n - -\n\n".join([doc.page_content for doc in relevant_results])

        # Create prompt using the retrieved context and query
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Initialize MT5 for text generation
        model_name = "google/mt5-large"
        tokenizer = MT5Tokenizer.from_pretrained(model_name, legacy=False)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)

        # Tokenize input and generate response
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)

        # Decode and return the generated response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Get sources of the matching documents
        sources = [doc.metadata.get("source", None) for doc, _score in results]

        # Format and return the response including sources
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return formatted_response, response_text

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def main():
    # Example query
    query_text = "What is depression?"

    # Call the RAG query function
    formatted_response, response_text = query_rag(query_text)

    # Print the final response
    if response_text:
        print(f"Formatted Response:\n{formatted_response}")
    else:
        print("Unable to generate Response :(")

if __name__ == "__main__":
    main()
