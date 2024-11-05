import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings/Chroma_IndicBERT"

# Load IndicBERT model and tokenizer
class IndicBERTEmbeddings:
    def __init__(self, model_name="ai4bharat/IndicBERTv2-MLM-only"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states[-1]
                embeddings.append(hidden_states[:, 0, :].numpy())  # [CLS] token embeddings
        return np.vstack(embeddings)

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Load documents
def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                doc = Document(page_content=text, metadata={"source": filename})
                documents.append(doc)
    return documents

# Split text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Load documents and split them
documents = load_documents()
split_docs = split_text(documents)

# Convert documents to text format for vectorizer
doc_texts = [doc.page_content for doc in split_docs]

# Initialize a TF-IDF vectorizer and fit it to the documents
vectorizer = TfidfVectorizer().fit(doc_texts)

# Initialize IndicBERT for further refinement
indicbert_embedder = IndicBERTEmbeddings()

# mT5 Generator setup
class MT5Generator:
    def __init__(self, model_name="google/mt5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, input_text, max_length=512):
        inputs = = self.tokenizer(input_text, return self.token_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response

generator = MT5Generator()

# Chroma setup for similarity search using IndicBERT embeddings
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=indicbert_embedder)

# Prompt template for RAG
PROMPT_TEMPLATE = """
Answer the following question based only on the provided context:
{context}
- -
Question: {question}
Answer:
"""

# Retrieve and Generate Response
def retrieve_and_generate(query, top_n_tfidf=10, top_n_chroma=5):
    # Step 1: Use TF-IDF to filter top N documents
    query_vec = vectorizer.transform([query])
    doc_vectors = vectorizer.transform(doc_texts)
    cosine_similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_n_indices_tfidf = np.argsort(cosine_similarities)[-top_n_tfidf:][::-1]
    filtered_docs = [split_docs[i] for i in top_n_indices_tfidf]

    # Insert filtered documents into Chroma for similarity search
    db.add_texts([doc.page_content for doc in filtered_docs], metadatas=[doc.metadata for doc in filtered_docs])

    # Step 2: Use Chroma similarity search to refine ranking among the filtered documents
    results = db.similarity_search_with_relevance_scores(query, k=top_n_chroma)

    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.1:
        return "No relevant documents found for the query."

    # Combine context from matching documents
    context = "\n\n - -\n\n".join([doc.page_content[:500] for doc, _ in results])

    # Step 3: Create a prompt for the generator using retrieved documents
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query)

    # Generate response using the prompt
    response = generator.generate(input_text=prompt, max_length=512)

    return response

# Example usage
if __name__ == "__main__":
    query = "Describe how to use the Amul AMCS Application?"
    response = retrieve_and_generate(query)
    print(response)
