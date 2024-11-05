import streamlit as st
import torch
import os
import pickle
from PIL import Image
from io import BytesIO
from RAG import index, retrieve_top_document, get_answer

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

# Define the function to handle the query and retrieve results
def answer_query(query: str, prompt: str):
    best_image, best_index = retrieve_top_document(
        query=query, 
        document_embeddings=document_embeddings, 
        document_images=images
    )
    # Ensure image is compatible with generative model
    compatible_image = convert_image_for_model(best_image)
    answer = get_answer(prompt, compatible_image)
    return answer, best_image, best_index

# Streamlit UI for user interaction
st.title("PromptEase SmartAssist")

# Define session state for managing interaction
if "response" not in st.session_state:
    st.session_state.response = ""
if "best_image" not in st.session_state:
    st.session_state.best_image = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Input fields for user queries
query = st.text_input("Enter the Name of Product/Search Key Terms:")
prompt = st.text_input("Enter the Issue in detail:")

if query and prompt:
    # Display processing spinner while fetching the result
    if st.button("Get Answer"):
        if query and prompt:  # Check if both query and prompt are provided
            with st.spinner("Retrieving information..."):
                answer, best_image, best_index = answer_query(query, prompt)
                st.session_state.response = answer
                st.session_state.best_image = best_image
                st.session_state.submitted = True  # Set flag to True after submission
        else:
            st.warning("Please enter both the product name and the issue detail.")

# Show the response and image only if submitted
if st.session_state.submitted:
    # Show the response text
    st.write("**Response:**")
    st.write(st.session_state.response)

    # Display the retrieved image
    if isinstance(st.session_state.best_image, Image.Image):
        st.image(st.session_state.best_image, caption=f"Best Match for Query: {query}", use_column_width=True)
    else:
        st.warning("No relevant image found for this query.")