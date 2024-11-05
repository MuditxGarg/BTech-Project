import streamlit as st
import torch
import os
import pickle
from PIL import Image
from io import BytesIO
from RAG import index, retrieve_top_document, get_answer
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Paths for saving/loading embeddings and images
EMBEDDINGS_PATH = "document_embeddings.pt"
IMAGES_PATH = "images.pkl"
DATA_FOLDER = "C:\\Users\\gargm\\Desktop\\Projects\\BTech\\Raw_DataFiles"

# Constants for device and batch size
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTIZATION = None  # Use "4-bit" or "8-bit" if needed for memory savings

# Initialize the translation model and tokenizer with quantization support
@st.cache_resource
def initialize_model_and_tokenizer(ckpt_dir="ai4bharat/indictrans2-en-indic-dist-200M", quantization=QUANTIZATION):
    # Configure quantization
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    # Move to device and set to eval mode
    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()
    model.eval()
    ip = IndicProcessor(inference=True)  # Initialize IndicProcessor

    return tokenizer, model, ip

en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
en_indic_tokenizer, en_indic_model, ip = initialize_model_and_tokenizer(en_indic_ckpt_dir)

# Language options
LANGUAGE_OPTIONS = {
    "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Bodo": "brx_Deva", 
    "Dogri": "doi_Deva", "English": "eng_Latn", "Gujarati": "guj_Gujr", 
    "Hindi": "hin_Deva", "Kannada": "kan_Knda", "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva", "Konkani": "gom_Deva", "Maithili": "mai_Deva", 
    "Malayalam": "mal_Mlym", "Manipuri (Bengali)": "mni_Beng", "Manipuri (Meitei)": "mni_Mtei", 
    "Marathi": "mar_Deva", "Nepali": "npi_Deva", "Odia": "ory_Orya", 
    "Punjabi": "pan_Guru", "Sanskrit": "san_Deva", "Santali": "sat_Olck", 
    "Sindhi (Arabic)": "snd_Arab", "Sindhi (Devanagari)": "snd_Deva", 
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Urdu": "urd_Arab"
}

# Function to save embeddings and images
def save_data(embeddings, images):
    torch.save(embeddings, EMBEDDINGS_PATH)
    with open(IMAGES_PATH, 'wb') as f:
        pickle.dump(images, f)

# Function to load embeddings and images
def load_data():
    try:
        embeddings = torch.load(EMBEDDINGS_PATH)
        with open(IMAGES_PATH, 'rb') as f:
            images = pickle.load(f)
        return embeddings, images
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        st.error("Error loading saved embeddings or images. Re-indexing is required.")
        return None, None

# Cache the indexed data to avoid re-indexing on every run
@st.cache_resource
def get_indexed_data():
    if os.path.isfile(EMBEDDINGS_PATH) and os.path.isfile(IMAGES_PATH):
        st.write("Loading embeddings and images from disk...")
        embeddings, images = load_data()
        if embeddings is not None and images is not None:
            return embeddings, images

    st.write("Indexing files... This might take some time.")
    pdf_files = [os.path.join(DATA_FOLDER, file) for file in os.listdir(DATA_FOLDER) if file.lower().endswith('.pdf')]
    document_embeddings, images = index(pdf_files)
    save_data(document_embeddings, images)
    return document_embeddings, images

# Load or create embeddings and images once per session
document_embeddings, images = get_indexed_data()

# Convert image to a format compatible with generative model requirements
def convert_image_for_model(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return Image.open(img_buffer)

# Define the function to handle the query and retrieve results
def answer_query(query: str, prompt: str):
    best_image, best_index = retrieve_top_document(
        query=query, 
        document_embeddings=document_embeddings, 
        document_images=images
    )
    compatible_image = convert_image_for_model(best_image)
    answer = get_answer(prompt, compatible_image)
    return answer, best_image, best_index

# Translate text to selected language
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):  # Batch size as defined
        try:
            batch = input_sentences[i:i+BATCH_SIZE]
            # Preprocess the batch
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize the batch
            inputs = tokenizer(
                batch, truncation=True, padding="longest",
                return_tensors="pt", return_attention_mask=True
            ).to(DEVICE)
            
            # Generate translations
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            
            # Decode and post-process translations
            decoded = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True)
            translations += ip.postprocess_batch(decoded, lang=tgt_lang)
            
        except Exception as e:
            st.error(f"Translation error: {e}")
            translations += ["Translation failed"] * len(input_sentences[i:i+BATCH_SIZE])
    return translations

def translate_text(text, target_lang):
    st.write(f"Translating text to {target_lang}: {text}")  # Debugging output
    if target_lang != "eng_Latn":  # Skip translation if target is English
        translation = batch_translate([text], "eng_Latn", target_lang, en_indic_model, en_indic_tokenizer, ip)
        st.write(f"Translated output: {translation}")  # Debugging output
        return translation[0]
    return text

# Streamlit UI for user interaction
st.title("PromptEase SmartAssist")

# Define session state for managing interaction
if "response" not in st.session_state:
    st.session_state.response = ""
if "best_image" not in st.session_state:
    st.session_state.best_image = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Language selection
selected_language = st.selectbox("Select your preferred language:", options=list(LANGUAGE_OPTIONS.keys()))
target_lang_code = LANGUAGE_OPTIONS[selected_language]

# Input fields for user queries
query = st.text_input("Enter the Name of Product/Search Key Terms:")
prompt = st.text_input("Enter the Issue in detail:")

if query and prompt:
    if st.button("Get Answer"):
        with st.spinner("Retrieving information..."):
            answer, best_image, best_index = answer_query(query, prompt)
            translated_answer = translate_text(answer, target_lang_code)
            st.session_state.response = answer
            st.session_state.translated_response = translated_answer
            st.session_state.best_image = best_image
            st.session_state.submitted = True

# Show the response and image only if submitted
if st.session_state.submitted:
    st.write("**Response (English):**")
    st.write(st.session_state.response)
    
    if target_lang_code != "eng_Latn":
        st.write(f"**Response ({selected_language}):**")
        st.write(st.session_state.translated_response)

    # Display the retrieved image
    if isinstance(st.session_state.best_image, Image.Image):
        st.image(st.session_state.best_image, caption=f"Best Match for Query: {query}", use_column_width=True)
    else:
        st.warning("No relevant image found for this query.")
