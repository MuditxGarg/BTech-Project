import os
import sys
import io

# Set UTF-8 encoding for standard output to handle all characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def replace_spaces_in_filenames(folder_path):
    for filename in os.listdir(folder_path):
        # Check if the filename has spaces
        if ' ' in filename:
            # Replace spaces with underscores in the new filename
            new_filename = filename.replace(' ', '_')
            # Full paths for the old and new filenames
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            # Print confirmation
            print(f"Renamed: '{filename}' to '{new_filename}'")

# Folder path containing files with Hindi, Gujarati, or other Unicode characters
folder_path = "C:\\Users\\gargm\\Desktop\\Projects\\BTech\\Raw_DataFiles"
replace_spaces_in_filenames(folder_path)
