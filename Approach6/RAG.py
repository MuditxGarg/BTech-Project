import torch
import typer
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM
import getpass

from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pdf2image import convert_from_path
from typing import List, Union, Tuple, cast
from dataclasses import asdict
from pathlib import Path
from einops import rearrange
import os 
import requests 

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import load_from_dataset

from colpali_engine.interpretability.vit_configs import VIT_CONFIG

from colpali_engine.interpretability.plot_utils import plot_attention_heatmap

from colpali_engine.interpretability.gen_interpretability_plots import generate_interpretability_plots
from colpali_engine.interpretability.processor import ColPaliProcessor
from colpali_engine.interpretability.plot_utils import plot_attention_heatmap
from colpali_engine.interpretability.torch_utils import normalize_attention_map_per_query_token
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import torch
from mteb.evaluation.evaluators import RetrievalEvaluator
from colpali_engine.utils.torch_utils import get_torch_device
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch, DRESModel
from sentence_transformers import SentenceTransformer

# Load API key from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

# %%
model_name = "vidore/colpali" # specify the adapter model name

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
)

retrieval_model = ColPali.from_pretrained("google/paligemma-3b-mix-448",
                                          torch_dtype=torch.float16, # set the dtype to bfloat16
                                          device_map="cuda",quantization_config=bnb_config).eval()    # set the device to cuda

vit_config = VIT_CONFIG["google/paligemma-3b-mix-448"]

retrieval_model.load_adapter(model_name)
paligemma_processor = AutoProcessor.from_pretrained(model_name)
device = retrieval_model.device

# %%
print(retrieval_model)

# %%
# Function to index the PDF document (Get the embedding of each page)
def index(files: List[str]) -> Tuple[str, List[torch.Tensor], List[Image.Image]]:
    poppler_path = r"C:\Poppler\poppler-24.08.0\Library\bin"
    images = []
    document_embeddings = []

    # Convert PDF pages to images
    for file in files:
        print(f"Indexing now: {file}")
        images.extend(convert_from_path(file, poppler_path=poppler_path))
        

    # Create DataLoader for image batches
    dataloader = DataLoader(
        images,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: process_images(paligemma_processor, x),
    )

    # Process each batch and obtain embeddings
    for batch in dataloader:
        with torch.no_grad():
            batch = {key: value.to(device) for key, value in batch.items()}
            embeddings = retrieval_model(**batch)
        document_embeddings.extend(list(torch.unbind(embeddings.to("cpu"))))
    total_memory = sum(embedding.element_size() * embedding.nelement() for embedding in document_embeddings)
    print(f'Total Embedding Memory (CPU): {total_memory/1024 **2} MB')


    total_image_memory = sum(image.width * image.height * 3 for image in images)  # 3 for RGB channels
    print(f'Total Image Memory: {total_image_memory / (1024 ** 2)} MB')
        

    # Return document embeddings, and images
    return document_embeddings, images

# %%
DATA_FOLDER = "C:\\Users\\gargm\\Desktop\\Projects\\BTech\\Raw_DataFiles"
# pdf_files = [os.path.join(DATA_FOLDER, file) for file in os.listdir(DATA_FOLDER) if file.lower().endswith('.pdf')]
# document_embeddings, images = index(pdf_files)
# %%

class MyEvaluatorWrapper:
    def __init__(self, is_multi_vector=False, retriever_model_name='all-MiniLM-L6-v2'):
        self.is_multi_vector = is_multi_vector
        self.device = get_torch_device()

        # Load a compatible retriever model
        retriever_model = SentenceTransformer(retriever_model_name)
        
        # Define retriever as an instance of DenseRetrievalExactSearch or DRESModel
        retriever = DenseRetrievalExactSearch(DRESModel(retriever_model))
        
        # Initialize RetrievalEvaluator with the retriever
        self.evaluator = RetrievalEvaluator(retriever=retriever)

    def evaluate(self, qs, ps):
        if self.is_multi_vector:
            scores = self.evaluate_colbert(qs, ps)
        else:
            scores = self.evaluate_biencoder(qs, ps)

        assert scores.shape[0] == len(qs)

        arg_score = scores.argmax(dim=1)
        accuracy = (arg_score == torch.arange(scores.shape[0], device=scores.device)).sum().item() / scores.shape[0]
        print(arg_score)
        print(f"Top 1 Accuracy (verif): {accuracy}")

        scores = scores.to(torch.float32).cpu().numpy()
        return scores

    def evaluate_colbert(self, qs, ps, batch_size=128) -> torch.Tensor:
        scores = []
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                self.device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(self.device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores.append(scores_batch)
        scores = torch.cat(scores, dim=0)
        return scores

    def evaluate_biencoder(self, qs, ps) -> torch.Tensor:
        qs = torch.stack(qs)
        ps = torch.stack(ps)
        scores = torch.einsum("bd,cd->bc", qs, ps)
        return scores


# %%
# Document Retrieval
def retrieve_top_document(query: str, document_embeddings: List[torch.Tensor], document_images: List[Image.Image]) -> Tuple[str, Image.Image]:
    query_embeddings = []
    # Create a placeholder image
    placeholder_image = Image.new("RGB", (448, 448), (255, 255, 255))

    with torch.no_grad():
        # Process the query to obtain embeddings
        query_batch = process_queries(paligemma_processor, [query], placeholder_image)
        query_batch = {key: value.to(device) for key, value in query_batch.items()}
        query_embeddings_tensor = retrieval_model(**query_batch)
        query_embeddings = list(torch.unbind(query_embeddings_tensor.to("cpu")))

    # Evaluate the embeddings to find the most relevant document
    retriever_evaluator = MyEvaluatorWrapper(is_multi_vector=True)
    similarity_scores = retriever_evaluator.evaluate(query_embeddings, document_embeddings)

    # Identify the index of the highest scoring document
    best_index = int(similarity_scores.argmax(axis=1).item())

    # Return the best matching document text and image
    return document_images[best_index], best_index

# %%
#Gemini LLM

generation_config = {
  "temperature": 0.0,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 1024,
  "response_mime_type": "text/plain",
}

genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel(model_name="gemini-1.5-flash" , generation_config=generation_config)

# %%
def get_answer(prompt:str , image:Image):
      response = model.generate_content([prompt, image])
      return response.text

# %%
def answer_query(query: str ,prompt):
    # Retrieve the most relevant document based on the query
    best_image, best_index = retrieve_top_document(query=query, 
                                        document_embeddings=document_embeddings, 
                                        document_images=images)

    #Gemini 1.5 Flash
    answer = f"{get_answer(prompt, best_image)}"
    

    return answer, best_image, best_index

# search_query = "ThawEasy Lite"
# prompt = "How do I adjust the temperature on ThawEasy Lite?" 
# answer, best_image,best_index  = answer_query(search_query, prompt)
# retrieved_idx = best_index

# print(answer)

# best_image