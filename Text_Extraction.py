import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for Word documents
from langdetect import detect
import os
import re
import json

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")  # Extract text in block format
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # Sort by position for column-wise extraction
            for block in blocks:
                text += block[4] + "\n"
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error processing DOCX {docx_path}: {e}")
        return None

def extract_tables_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        tables = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")['blocks']
            for block in blocks:
                if 'lines' in block:
                    table_text = []
                    for line in block['lines']:
                        line_text = " | ".join([span['text'] for span in line['spans']])
                        table_text.append(line_text)
                    tables.append("\n".join(table_text))
        return tables
    except Exception as e:
        print(f"Error extracting tables from PDF {pdf_path}: {e}")
        return []

def preprocess_text(text):
    # Basic text cleaning
    text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()
    return text

def split_into_sections(text):
    # Split based on headings (assuming headings follow a pattern)
    sections = re.split(r'(?<=\n)([A-Z][^\n]+)(?=\n)', text)
    structured_sections = {sections[i]: sections[i+1] for i in range(0, len(sections) - 1, 2)}
    return structured_sections

def extract_data_from_files(directory, output_dir):
    all_data = {}
    failed_files = []
    metadata = {}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        text = None
        tables = []

        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            tables = extract_tables_from_pdf(file_path)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        
        if text:
            # Language detection (optional, can remove if not needed)
            try:
                language = detect(text)
            except:
                language = "unknown"
            
            # Preprocess and split into sections
            clean_text = preprocess_text(text)
            sections = split_into_sections(clean_text)
            
            # Save text to a .txt file
            output_file_path = os.path.join(output_dir, filename.replace('.pdf', '.txt').replace('.docx', '.txt'))
            with open(output_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(clean_text)
            
            # Store metadata
            metadata[filename] = {
                'language': language,
                'sections': list(sections.keys()),
                'tables': len(tables),
                'output_text_file': output_file_path
            }
        else:
            failed_files.append(filename)
    
    # Save metadata
    metadata_file_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file_path, 'w', encoding='utf-8') as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=4)
    
    # Save failed files
    failed_files_path = os.path.join(output_dir, 'failed_files.txt')
    with open(failed_files_path, 'w', encoding='utf-8') as failed_file:
        for failed in failed_files:
            failed_file.write(failed + '\n')

    print(f"Processing completed. Metadata saved at {metadata_file_path}. Failed files saved at {failed_files_path}.")

# Example usage
directory = 'Raw_DataFiles' 
output_dir = 'Extracted_DataFiles'
extract_data_from_files(directory, output_dir)
