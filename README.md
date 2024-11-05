# BTech-Project: A Multimodal Context-Aware Generative AI Driven Chatbot 

This repository contains code and resources for various approaches, each implemented in its folder, to develop and enhance AI-driven multimodal systems with features like audio-text input, multilingual support, and contextual response generation. This README outlines each folder and file in the repository, focusing in detail on `Approach6`, which features advanced multilingual chatbot capabilities.

## Repository Structure

- `Approach1`: Contains the initial implementation of a text-only chatbot with basic functionality for handling user queries in English. This approach focuses on the foundational aspects of chatbot development, such as basic query processing and response generation.
  
- `Approach2`: Builds upon Approach1 by adding support for predefined responses based on keywords, enhancing the relevance of answers to user queries. This folder contains code and resources for keyword-based query matching.

- `Approach3`: Introduces document-based responses by integrating a document retrieval system that identifies and retrieves relevant documents from a database. This approach incorporates simple document indexing techniques for efficient retrieval.

- `Approach4`: Expands the functionality to include basic language support, allowing responses in selected Indic languages. This folder includes scripts for integrating a translation API, which is used to translate responses based on the selected language.

- `Approach5`: Integrates an advanced indexing mechanism for document embeddings, optimizing the response generation process by selecting the most relevant documents based on query similarity.

- **`Approach6`**: The most feature-rich implementation, detailed below, with advanced functionalities like multilingual support, contextual history, and both text and audio input support.

- `Text_Extraction.py`: A Python script for extracting text data from document sources, intended for preprocessing and indexing.

- `requirements.txt`: Lists the dependencies required to run the code in this repository. Use `pip install -r requirements.txt` to install them.

## Approach6 - Detailed Overview

`Approach6` introduces a robust chatbot framework with the following advanced features:

### Core Features

1. **Audio and Text Input Support**:
   - Users can interact with the chatbot by typing text or recording audio prompts.
   - Audio input is processed through Google Speech Recognition, which transcribes it into text, automatically detects the language, and translates it into English for further processing.

2. **Contextual Retrieval with History**:
   - The app maintains a chat history, preserving each question-answer pair.
   - Previous interactions are utilized to generate contextually relevant responses, enabling the chatbot to understand and refer back to prior exchanges within the same session.

3. **Multilingual Support**:
   - Users can select their preferred language for responses from a list that includes Hindi, Bengali, Tamil, Telugu, Urdu, and more.
   - The app translates responses into the selected language using Google Translate, with special handling to retain formatting elements like bold text and numbered lists for readability.

4. **Embeddings and Document Management**:
   - Predefined document embeddings are used to find the most relevant document for each user query.
   - This setup ensures that responses are based on accurate and relevant data, with caching to avoid re-indexing.

5. **Image Retrieval**:
   - For queries involving specific documents, associated images are retrieved and displayed alongside text responses.

6. **Session-based Chat History**:
   - Each chat session has its own history, allowing users to manage multiple sessions independently.
   - A sidebar enables easy navigation between sessions, with each session’s history displayed separately in the main chat window.

7. **Formatting for Translations**:
   - The app preserves text formatting during translations, ensuring readability with numbered lists, bold text, and inline formatting in translated responses.

8. **Interface Customization**:
   - The interface is designed with user-friendly custom styling, including distinct sections for user and bot messages, a sidebar for chat management, and a clear layout for query inputs, answers, and images.

### User Interaction Features

- **Chat Navigation and Management**:
  - Users can create new chat sessions, with each session’s interactions stored and displayed separately.
  - A sidebar lists all active chat sessions for easy switching between conversations.

- **Real-Time Processing Indicators**:
  - Processing indicators are displayed for tasks like audio transcription or document retrieval, providing feedback during longer processing times.

- **Dynamic Language Selection**:
  - Users can choose a response language from a dropdown menu, and `language_code` is updated for each query to ensure responses are generated in the selected language.

### Error Handling

- **Error Messaging**:
  - The app captures and displays user-friendly messages for errors such as missing files or transcription failures.
  - For example, if embeddings are missing or corrupted, users receive prompts to re-index files as needed.

- **Fallback Mechanism**:
  - In case of translation or file access issues, the app defaults to providing basic responses or suggesting necessary actions (e.g., re-indexing) without interrupting the session.

### Backend Features

- **Caching with Streamlit**:
  - Embeddings and document images are cached to reduce load times, enhancing performance by eliminating the need for repeated indexing.

- **Temporary File Handling for Audio**:
  - Audio files are stored temporarily for processing and then deleted, ensuring efficient memory use.

- **Resource Management**:
  - Embeddings and images are loaded only when needed, optimizing memory usage and minimizing storage demands.

These features enable the chatbot in `Approach6` to support diverse, contextually relevant interactions, handle multiple languages, and respond to audio and text inputs effectively, offering a user-friendly experience with robust error handling.

---

### Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request for review.
