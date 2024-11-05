import streamlit as st
import torch
import os
import pickle
from PIL import Image
from io import BytesIO
from googletrans import Translator
from RAG import index, retrieve_top_document, get_answer
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

# Initialize the Google Translator and Speech Recognizer
translator = Translator()
recognizer = sr.Recognizer()

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

                translated_content = translator.translate(content, dest=language).text
                translated_lines.append(f"{prefix} {translated_content}" if prefix else translated_content)

            except Exception as e:
                st.error(f"Error translating line: {line}")
                translated_lines.append(line)
        else:
            translated_lines.append("")  # Preserve blank lines for readability

    return "\n".join(translated_lines)

def answer_query_with_context(history, query, prompt, language_code):
    context = " ".join([f"Q: {q['query']} A: {q['response_english']}" for q in history])
    prompt_with_context = f"{context} Q: {prompt}"

    best_image, best_index = retrieve_top_document(
        query=query, 
        document_embeddings=document_embeddings, 
        document_images=images
    )
    compatible_image = convert_image_for_model(best_image)
    answer_english = get_answer(prompt_with_context, compatible_image)

    answer_translated = translate_with_formatting(answer_english, language_code)
    return answer_english, answer_translated, best_image, best_index

# Function to handle audio input, detect language, and transcribe to text in English
def transcribe_audio_to_text(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name

    audio = AudioSegment.from_wav(temp_file_path)
    audio.export(temp_file_path, format="wav")

    try:
        # Now, open the temp file again for transcription and close it immediately after use
        with sr.AudioFile(temp_file_path) as source:
            audio_data = recognizer.record(source)
            # Transcribe audio to text and detect language
            detected_text = recognizer.recognize_google(audio_data)
            detected_language = translator.detect(detected_text).lang
            transcribed_text_english = translator.translate(detected_text, src=detected_language, dest="en").text
            return transcribed_text_english, detected_language, detected_text
    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try again.")
    except sr.RequestError:
        st.error("Error with the speech recognition service.")
    finally:
        # Ensure the temp file is deleted after use
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    return "", "", ""

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
    # Clear the input fields for new chat
    st.session_state.query_input = ""
    st.session_state.prompt_input = ""

# Load selected chat
def load_chat(chat_id):
    st.session_state.history = st.session_state.chats[chat_id]["history"]

# Set current chat history
if "history" not in st.session_state:
    st.session_state.history = st.session_state.chats[st.session_state.current_chat_id]["history"]

# Real-time audio recording for prompt and processing
if st.button("Record Prompt for Issue Details"):
    with st.spinner("Listening..."):
        try:
            # Capture audio from the microphone
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.listen(source)
                
            # Transcribe the audio and detect its language
            transcribed_text_english, detected_language, original_text = transcribe_audio_to_text(audio_data.get_wav_data())
            st.success("Transcription and language detection completed.")
            
            # Set the detected language
            language_code = detected_language
            st.write(f"Detected language: {language_code}")
            st.write(f"Transcription (Original): {original_text}")
            st.write(f"Transcription (English): {transcribed_text_english}")
            
            # Update the prompt field with transcribed English text
            st.session_state.prompt_input = transcribed_text_english
            
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results; check your network connection.")
else:
    language_code = "en"  # Default to English if no language is detected

# Input fields for user queries
query = st.text_input("Enter the Name of Product/Search Key Terms:", key="query_input")
prompt = st.text_input("Enter the Issue in detail:", value=st.session_state.get("prompt_input", ""), key="prompt_input")

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
        st.markdown(f"<div class='bot-message'><strong>Response ({language_code}):</strong> {entry['response_translated']}</div>", unsafe_allow_html=True)
    if isinstance(entry['image'], Image.Image):
        st.image(entry['image'], use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)
