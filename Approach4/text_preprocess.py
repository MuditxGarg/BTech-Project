import os
from transformers import AutoTokenizer

# Constants
DATA_DIR = './Extracted_DataFiles'  # Directory containing your extracted text
OUTPUT_DIR = './processed_chunks'   # Directory to store processed chunks
MAX_LENGTH = 512                    # Max token length for each chunk

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    return text.replace("\n", " ").strip()

def chunk_text(text, max_length=MAX_LENGTH):
    # Tokenize the text with truncation to ensure no sequence exceeds the max_length
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Convert token ids back to string tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Group tokens into chunks
    chunks = [' '.join(tokens[i:i+max_length]) for i in range(0, len(tokens), max_length)]
    
    return chunks

def process_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunk_index = 0  # Keep track of chunk indices

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            text = preprocess_text(text)
            chunks = chunk_text(text)

            # Save each chunk with an incremental index
            for chunk in chunks:
                output_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_index}.txt")
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(chunk + '\n')
                chunk_index += 1

if __name__ == "__main__":
    process_files()
