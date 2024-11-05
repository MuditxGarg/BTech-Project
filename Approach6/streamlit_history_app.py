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
    if language == "en":  # Skip translation if language is English
        return text

    lines = text.split("\n")
    translated_lines = []
    for line in lines:
        if line.strip():
            try:
                if line[0].isdigit() and line[1] == ".":  # Handle numbered bullet points
                    prefix, content = line.split(" ", 1)
                else:
                    prefix, content = "", line

                translated_content = ""
                parts = content.split("")
                
                for i, part in enumerate(parts):
                    if part.strip():
                        translated_part = translator.translate(part.strip(), dest=language).text
                        translated_content += f"{translated_part}" if i % 2 == 1 else translated_part

                translated_lines.append(f"{prefix} {translated_content}" if prefix else translated_content)

            except Exception as e:
                st.error(f"Error translating line: {line}")
                translated_lines.append(line)
        else:
            translated_lines.append("")  # Preserve blank lines for readability

    return "\n".join(translated_lines)

def answer_query_with_context(history, query: str, prompt: str, language: str):
    context = " ".join([f"Q: {q['query']} A: {q['response_english']}" for q in history])
    prompt_with_context = f"{context} Q: {prompt}"

    best_image, best_index = retrieve_top_document(
        query=query, 
        document_embeddings=document_embeddings, 
        document_images=images
    )
    compatible_image = convert_image_for_model(best_image)
    answer_english = get_answer(prompt_with_context, compatible_image)

    answer_translated = translate_with_formatting(answer_english, language)
    return answer_english, answer_translated, best_image, best_index

# Custom CSS for ChatGPT-like styling
st.markdown("""
    <style>
        body {background-color: #2c2f33;}
        .chat-container {
            max-width: 800px;
            margin: auto;
        }
        .user-message, .bot-message {
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
            color: white;
            font-size: 16px;
        }
        .user-message {
            background-color: #4a4d52;
            text-align: left;
        }
        .bot-message {
            background-color: #3b3e45;
            text-align: left;
        }
        .image-display {
            text-align: center;
        }
        .sidebar-content {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("PromptEase SmartAssist")

# Initialize chat history in session state if not present
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current_chat_id = 1
    st.session_state.chats[st.session_state.current_chat_id] = {"history": [], "name": "Chat 1"}

# Define function to reset and start a new chat
def start_new_chat():
    st.session_state.current_chat_id += 1
    new_chat_id = st.session_state.current_chat_id
    st.session_state.chats[new_chat_id] = {"history": [], "name": f"Chat {new_chat_id}"}
    st.session_state.history = st.session_state.chats[new_chat_id]["history"]

# Load selected chat
def load_chat(chat_id):
    st.session_state.history = st.session_state.chats[chat_id]["history"]

# Set current chat history
if "history" not in st.session_state:
    st.session_state.history = st.session_state.chats[st.session_state.current_chat_id]["history"]

# Input fields for user queries
query = st.text_input("Enter the Name of Product/Search Key Terms:", key="query_input")
prompt = st.text_input("Enter the Issue in detail:", key="prompt_input")

# Language selection with default as English
language_options = {
    "English": "en",
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

language = st.selectbox("Select Language", list(language_options.keys()), index=0, key="language")
language_code = language_options[language]

# Sidebar for chat management
with st.sidebar:
    st.button("New Chat", on_click=start_new_chat)  # Start a new chat session

    # Display list of past chats
    st.sidebar.markdown("<div class='sidebar-content'><b>Chat History</b></div>", unsafe_allow_html=True)
    for chat_id, chat in st.session_state.chats.items():
        if st.sidebar.button(chat["name"]):
            load_chat(chat_id)  # Load selected chat into the main chat window

# Check if there is input to process
if query and prompt:
    if st.button("Get Answer"):
        with st.spinner("Retrieving information..."):
            answer_english, answer_translated, best_image, best_index = answer_query_with_context(
                st.session_state.history, query, prompt, language_code
            )
            new_entry = {
                "query": prompt,
                "response_english": answer_english,
                "response_translated": answer_translated if language_code != "en" else "",
                "image": best_image
            }
            # Append the new entry to the current chat history
            st.session_state.history.append(new_entry)
            st.session_state.chats[st.session_state.current_chat_id]["history"] = st.session_state.history

# Display chat history for the current session
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for entry in st.session_state.history:
    st.markdown(f"<div class='user-message'><strong>User:</strong> {entry['query']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-message'><strong>Response (English):</strong> {entry['response_english']}</div>", unsafe_allow_html=True)
    if language_code != "en":
        st.markdown(f"<div class='bot-message'><strong>Response ({language}):</strong> {entry['response_translated']}</div>", unsafe_allow_html=True)
    if isinstance(entry['image'], Image.Image):
        st.image(entry['image'], use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)
