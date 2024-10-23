import os
import torch
import uuid
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, MT5ForConditionalGeneration, T5Tokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

# Constants
DATA_PATH = "./Extracted_DataFiles"
CHROMA_PATH = "./Embeddings/Chroma_IndicBert"
DPR_MODEL_NAME = "facebook/dpr-question_encoder-single-nq-base"
DPR_CONTEXT_MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"
MT5_MODEL_NAME = "google/mt5-base"
MAX_BATCH_SIZE = 256  # Adjust based on system capacity

# Load IndicBERT model and tokenizer
class IndicBERTEmbeddings:
    def __init__(self, model_name="ai4bharat/IndicBERTv2-MLM-only"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def embed_documents(self, documents):
        if isinstance(documents, list):
            texts = documents
        else:
            texts = [documents]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states[:, 0, :]  # [CLS] token embeddings
        return embeddings.cpu().numpy().tolist()  # Convert to list of lists for compatibility

    def embed_query(self, query):
        return self.embed_documents(query)

# Instantiate the IndicBERT embedder
embedder = IndicBERTEmbeddings()

# Load and split documents
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

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Save chunks and embeddings to Chroma
def save_to_chroma(chunks, indicbert_embedder):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = None
    for i in tqdm(range(0, len(chunks), MAX_BATCH_SIZE)):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        documents = [chunk.page_content for chunk in batch]
        embeddings = indicbert_embedder.embed_documents(documents)
        metadatas = [chunk.metadata for chunk in batch]
        ids = [str(uuid.uuid4()) for _ in range(len(batch))]

        if db is None:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=None)

        db._collection.upsert(embeddings=embeddings, metadatas=metadatas, documents=documents, ids=ids)
        db.persist()

# Generate the data store
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    indicbert_embedder = IndicBERTEmbeddings()
    save_to_chroma(chunks, indicbert_embedder)

generate_data_store()

# Load Chroma as vector store, providing the embedding function
chroma = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedder.embed_documents  # Use the embed_documents method to generate embeddings
)

# Prompt template for RAG
PROMPT_TEMPLATE = """
Answer the following question based only on the provided context:
{context}
- -
Question: {question}
Answer:
"""

# Define a custom retriever class to interact with the Chroma vector store
class CustomRetriever:
    def __init__(self, vectorstore, embedding_function):
        self.vectorstore = vectorstore
        self.embedding_function = embedding_function

    def get_relevant_documents(self, query):
        # Embed the query
        query_embedding = self.embedding_function(query)
        # Use Chroma's built-in similarity search with the query embedding
        return self.vectorstore.similarity_search(query_embedding)

# Set up the custom retriever
retriever = CustomRetriever(vectorstore=chroma, embedding_function=embedder.embed_query)

# MT5 Generator setup
class MT5Generator:
    def __init__(self, model_name=MT5_MODEL_NAME):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

generator = MT5Generator()

# Retrieve and Generate Response
def retrieve_and_generate(query):
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)

    # Check if there are any matching results
    if not retrieved_docs:
        return "No relevant documents found for the query."

    # Combine context from matching documents
    context = "\n\n - -\n\n".join([doc.page_content for doc in retrieved_docs])

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query)

    # Generate response using the prompt
    response = generator.generate(input_text=prompt)

    return response

# Example usage
if __name__ == "__main__":
    query = "Describe how to use the Amul AMCS Application?"
    response = retrieve_and_generate(query)
    print(response)
