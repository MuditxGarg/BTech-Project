import os
import re

def preprocess_text(text):
    text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def split_into_passages(text, chunk_size=300):
    words = text.split()
    passages = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return passages

def process_text_files(directory):
    all_passages = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                clean_text = preprocess_text(text)
                passages = split_into_passages(clean_text)
                all_passages.extend(passages)
    return all_passages

if __name__ == "__main__":
    # Directory with extracted text files
    directory = 'Extracted_DataFiles'

    # Process the text files
    print("Processing text files into passages...")
    all_passages = process_text_files(directory)

    # Save the passages for later use
    with open('passages.txt', 'w', encoding='utf-8') as f:
        for passage in all_passages:
            f.write(passage + '\n')

    print(f"Processed {len(all_passages)} passages.")
