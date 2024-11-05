import streamlit as st
import torch
import os
import pickle
from PIL import Image
from io import BytesIO
from googletrans import Translator
from RAG import index, retrieve_top_document, get_answer

# Initialize the Google Translator
translator = Translator()

# Paths for saving/loading embeddings and images
EMBEDDINGS_PATH = "document_embeddings.pt"
IMAGES_PATH = "images.pkl"
DATA_FOLDER = "C:\\Users\\gargm\\Desktop\\Projects\\BTech\\Raw_DataFiles"

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

# Translate text with handling for numbered bullet points and inline bold text
def translate_with_formatting(text, language):
    lines = text.split("\n")
    translated_lines = []

    for line in lines:
        if line.strip():
            try:
                # Check if the line starts with a numbered list format (e.g., "1.")
                if line[0].isdigit() and line[1] == ".":
                    # Separate the numbered prefix and the rest of the line
                    prefix, content = line.split(" ", 1)
                else:
                    # No numbered prefix
                    prefix, content = "", line

                # Handle inline bold segments marked with "**"
                translated_content = ""
                parts = content.split("**")
                
                # Translate each part, alternating between bold and regular text
                for i, part in enumerate(parts):
                    if part.strip():  # Only translate non-empty parts
                        translated_part = translator.translate(part.strip(), dest=language).text
                        if i % 2 == 1:  # Odd-indexed parts are within "**" and should be bold
                            translated_content += f"**{translated_part}**"
                        else:  # Even-indexed parts are regular text
                            translated_content += translated_part

                # Reassemble the line with the prefix (if any) and the translated content
                if prefix:
                    translated_lines.append(f"{prefix} {translated_content}")
                else:
                    translated_lines.append(translated_content)

            except Exception as e:
                st.error(f"Error translating line: {line}")
                translated_lines.append(line)  # Fallback to original line if error
        else:
            translated_lines.append("")  # Preserve blank lines for readability

    # Join lines to form the final translated text
    return "\n".join(translated_lines)

# Define the function to handle the query and retrieve results
def answer_query(query: str, prompt: str, language: str):
    best_image, best_index = retrieve_top_document(
        query=query, 
        document_embeddings=document_embeddings, 
        document_images=images
    )
    compatible_image = convert_image_for_model(best_image)
    answer_english = get_answer(prompt, compatible_image)

    # Translate answer to the selected language with improved formatting
    answer_translated = translate_with_formatting(answer_english, language)
    return answer_english, answer_translated, best_image, best_index

# Streamlit UI for user interaction
st.title("PromptEase SmartAssist")

# Define session state for managing interaction
if "response_english" not in st.session_state:
    st.session_state.response_english = ""
if "response_translated" not in st.session_state:
    st.session_state.response_translated = ""
if "best_image" not in st.session_state:
    st.session_state.best_image = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Input fields for user queries
query = st.text_input("Enter the Name of Product/Search Key Terms:")
prompt = st.text_input("Enter the Issue in detail:")

# Language selection with Google Translate codes for all supported Indian languages
language_options = {
    "Assamese": "as",
    "Bengali": "bn",
    "Bhojpuri": "bho",
    "Dogri": "doi",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Konkani": "kok",
    "Maithili": "mai",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Meiteilon (Manipuri)": "mni-Mtei",
    "Mizo": "lus",
    "Nepali": "ne",
    "Odia (Oriya)": "or",
    "Punjabi": "pa",
    "Sanskrit": "sa",
    "Santali": "sat",
    "Sindhi": "sd",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur"
}

language = st.selectbox("Select Language", list(language_options.keys()))
language_code = language_options[language]

if query and prompt:
    # Display processing spinner while fetching the result
    if st.button("Get Answer"):
        if query and prompt:  # Check if both query and prompt are provided
            with st.spinner("Retrieving information..."):
                answer_english, answer_translated, best_image, best_index = answer_query(query, prompt, language_code)
                st.session_state.response_english = answer_english
                # Remove ** markers from the translated text before displaying
                st.session_state.response_translated = answer_translated.replace("**", "")
                st.session_state.best_image = best_image
                st.session_state.submitted = True  # Set flag to True after submission
        else:
            st.warning("Please enter both the product name and the issue detail.")

# Show the response and image only if submitted
if st.session_state.submitted:
    # Show the response in English
    st.write("**Response (English):**")
    st.markdown(st.session_state.response_english)

    # Show the translated response
    st.write(f"**Response ({language}):**")
    st.markdown(st.session_state.response_translated)

    # Display the retrieved image
    if isinstance(st.session_state.best_image, Image.Image):
        st.image(st.session_state.best_image, caption=f"Best Match for Query: {query}", use_column_width=True)
    else:
        st.warning("No relevant image found for this query.")
