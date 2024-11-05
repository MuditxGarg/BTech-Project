import os
import numpy as np
from sklearn.metrics import precision_recall_curve
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Import necessary functions from scripts
from IndicBert_BM25 import retrieve_and_generate as IndicBERT_BM25
from IndicBert_DPR import retrieve_and_generate as IndicBERT_DPR
from mT5_BM25 import retrieve_and_generate as mT5_BM25
from mT5_DPR import retrieve_and_generate as mT5_DPR
from Sentence_Transformer_BM25 import retrieve_and_generate as SentenceTransformer_BM25
from Sentence_Transformer_DPR import retrieve_and_generate as SentenceTransformer_DPR
from indicbert_embeddings_FAISS import generate_data_store as IndicBERT_FAISS
from mt5_embeddings_FAISS import generate_data_store as mT5_FAISS
from sentence_transformer_embeddings_FAISS import generate_data_store as SentenceTransformer_FAISS

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Sample query and reference data
query = "How can i adjust the temperature on ThawEasy Lite Machine?"
true_relevant_documents = ["ThawEasy Lite doenot have an option to manually setup the temperature, however on Smart Pro version, temperature can be modified using cloud via the control application."] 
reference_response = "The ThawEasy Lite does not have a temperature adjustment feature. It is a manual control device, meaning you cannot adjust the temperature. The thawing temperature is fixed at 35 to 42°C (cloud control)." 

# Define evaluation functions
def recall_at_k(retrieved_docs, relevant_docs, k=5):
    retrieved_relevant = [doc for doc in retrieved_docs[:k] if doc in relevant_docs]
    return len(retrieved_relevant) / min(len(relevant_docs), k)

def mean_reciprocal_rank(retrieved_docs, relevant_docs):
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0

def average_precision(retrieved_docs, relevant_docs):
    relevant_retrieved = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
    precisions = []
    for k in range(1, len(retrieved_docs) + 1):
        if relevant_retrieved[k - 1]:
            precisions.append(sum(relevant_retrieved[:k]) / k)
    return np.mean(precisions) if precisions else 0

def bleu_score(generated_response, reference_response):
    reference_tokens = [reference_response.split()]
    generated_tokens = generated_response.split()
    return sentence_bleu(reference_tokens, generated_tokens)

def rouge_score(generated_response, reference_response):
    scores = scorer.score(reference_response, generated_response)
    return scores['rougeL'].fmeasure

def content_relevance_rating(generated_response, reference_response):
    return 3  # Replace with actual rating if available

def evaluate_approach(retrieve_and_generate_func, query, relevant_docs, reference_response):
    # Retrieve and generate response
    response, retrieved_docs = retrieve_and_generate_func(query)

    # Retrieval metrics
    recall = recall_at_k(retrieved_docs, relevant_docs, k=5)
    mrr = mean_reciprocal_rank(retrieved_docs, relevant_docs)
    ap = average_precision(retrieved_docs, relevant_docs)

    # Generation metrics
    bleu = bleu_score(response, reference_response)
    rouge = rouge_score(response, reference_response)
    relevance_rating = content_relevance_rating(response, reference_response)

    return {
        "Recall@5": recall,
        "MRR": mrr,
        "AP": ap,
        "BLEU": bleu,
        "ROUGE-L": rouge,
        "Content Relevance Rating": relevance_rating
    }

# List of approaches and their corresponding functions
approaches = {
    "IndicBERT + BM25": IndicBERT_BM25,
    "IndicBERT + DPR": IndicBERT_DPR,
    "mT5 + BM25": mT5_BM25,
    "mT5 + DPR": mT5_DPR,
    "Sentence-Transformer + BM25": SentenceTransformer_BM25,
    "Sentence-Transformer + DPR": SentenceTransformer_DPR,
    "IndicBERT + Chroma + FAISS": IndicBERT_FAISS,
    "mT5 + Chroma + FAISS": mT5_FAISS,
    "Sentence-Transformer + Chroma + FAISS": SentenceTransformer_FAISS,
}

# Evaluate each approach
results = {}
for approach_name, func in approaches.items():
    metrics = evaluate_approach(func, query, true_relevant_documents, reference_response)
    results[approach_name] = metrics

# Print results
print("Evaluation Results:")
for approach, metrics in results.items():
    print(f"\n{approach}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.2f}")
