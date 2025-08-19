import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markdown_it import MarkdownIt
from langchain_core.documents import Document
from langchain.text_splitter import NLTKTextSplitter
from src import config
import re # Added to clean the medicine name

# --- One-time NLTK setup (if necessary) ---
# In some environments, it may be necessary to download the NLTK data ('punkt').
# Uncomment the following two lines and run this script directly once if
# you get a 'punkt' related error the first time you use it.
# import nltk
# nltk.download('punkt')

def markdown_to_semantic_blocks(markdown_text):
    """
    Parses Markdown text and groups it into semantic blocks based on headers.
    A block contains a header and all the content up to the next header.
    """
    md = MarkdownIt()
    tokens = md.parse(markdown_text)
    
    blocks = []
    current_block_content = []
    current_path = []
    
    # Add a final "dummy" token to ensure the last block is saved
    tokens.append(type('Token', (), {'type': 'heading_open', 'tag': 'h0', 'content': 'End of Document', 'level': 0})())

    for i, token in enumerate(tokens):
        if token.type == 'heading_open':
            # When we find a new header, save the previous block if it had content.
            if current_block_content:
                blocks.append({
                    "path": " > ".join(current_path) if current_path else "General Information",
                    "content": "".join(current_block_content).strip()
                })
            
            level = int(token.tag[1]) if token.tag else 99
            
            header_content = ""
            if i + 1 < len(tokens) and tokens[i+1].type == 'inline':
                 header_content = tokens[i+1].content
            
            if level == 0:
                break
            
            while len(current_path) >= level:
                current_path.pop()
            current_path.append(header_content)
            
            # The block's content starts with its header
            current_block_content = [f"{'#' * level} {header_content}\n\n"]

        # The paragraph is ignored if it's part of a list item to avoid duplicating text
        elif token.type == 'paragraph_open':
            if i > 0 and tokens[i-1].type == 'list_item_open':
                continue
            content = tokens[i + 1].content if i + 1 < len(tokens) else ""
            current_block_content.append(f"{content}\n\n")

        elif token.type == 'bullet_list_open':
             current_block_content.append("\n")

        elif token.type == 'list_item_open':
            item_content = ""
            # Search for the list item's content in the following tokens
            for next_token in tokens[i+1:]:
                if next_token.type == 'list_item_close' and next_token.level == token.level:
                    break
                if next_token.type == 'inline':
                    item_content = next_token.content
                    break
            
            indent = "    " * (token.level // 2)
            current_block_content.append(f"{indent}- {item_content}\n")

    return blocks

def create_sentence_window_chunks(blocks, source_file, medicine_name, window_size=2):
    """
    Takes semantic blocks and creates chunks based on individual sentences,
    adding a "window context" (N sentences before and after).
    This version is more robust as it pre-processes the Markdown for NLTK,
    correctly handling multi-line paragraphs and lists.
    """
    all_chunks = []
    sentence_splitter = NLTKTextSplitter(language='spanish')

    for block in blocks:
        sentences_in_block = []
        
        # First, we remove the headers from the content so they are not processed
        content_without_headers = "\n".join([line for line in block['content'].strip().split('\n') if not line.strip().startswith('#')])
        
        lines = content_without_headers.strip().split('\n')
        paragraph_buffer = []

        for line in lines:
            stripped_line = line.strip()
            # If it's a list item, it's a semantic unit in itself.
            if stripped_line.startswith('- '):
                # First, we process any paragraph that was in the buffer
                if paragraph_buffer:
                    full_paragraph = " ".join(paragraph_buffer)
                    sentences_in_block.extend(sentence_splitter.split_text(full_paragraph))
                    paragraph_buffer = [] # Clear the buffer
                
                # We add the list item as its own "sentence"
                sentences_in_block.append(stripped_line.lstrip('- ').strip())
            # If it's an empty line, it also indicates a paragraph break
            elif not stripped_line:
                if paragraph_buffer:
                    full_paragraph = " ".join(paragraph_buffer)
                    sentences_in_block.extend(sentence_splitter.split_text(full_paragraph))
                    paragraph_buffer = []
            # If it's not a list item, we add it to the current paragraph buffer
            else:
                paragraph_buffer.append(stripped_line)

        # Don't forget to process the last paragraph if the block doesn't end with a list
        if paragraph_buffer:
            full_paragraph = " ".join(paragraph_buffer)
            sentences_in_block.extend(sentence_splitter.split_text(full_paragraph))

        # Filter out any empty results that might remain.
        sentences = [s for s in sentences_in_block if s]

        if not sentences:
            continue

        # 2. Iterate over each sentence to create a chunk (this logic doesn't change)
        for i, sentence in enumerate(sentences):
            start_index = max(0, i - window_size)
            end_index = min(len(sentences), i + window_size + 1)
            context_window = " ".join(sentences[start_index:end_index])
            
            metadata = {
                "source": source_file,
                "path": block['path'],
                "medicine_name": medicine_name,
                "main_sentence": sentence
            }
            
            final_content_to_embed = f"""---
METADATA:
- Medicine: {metadata['medicine_name']}
- Path: {metadata['path']}
---
CONTEXT:
{context_window}
""".strip()
            
            all_chunks.append(Document(page_content=final_content_to_embed, metadata=metadata))
            
    return all_chunks


# This block only runs if you execute "python scripts/_02_markdown_to_chunks.py" directly.
# It serves as a quick and clean smoke test.
if __name__ == '__main__':
    # --- Test parameters, local to this block ---
    # This block now serves as a quick test using the central configuration.
    # To process a new file, the main entry point is '03_ingest.py'.
    print("--- Running test for '02_markdown_to_chunks.py' ---")
    print("This script should not be run for ingestion. Use 'ingest.py'.")

    # We use a test file that should exist
    model_name_slug = config.PDF_PARSE_MODEL.replace('.', '-')
    INPUT_MD_NAME = f"parsed_by_{model_name_slug}_espidifen_600.md"
    
    md_file_path = os.path.join(config.MARKDOWN_PATH, INPUT_MD_NAME)

    if not os.path.exists(md_file_path):
        print(f"Error: The input Markdown file '{md_file_path}' was not found.")
        print("Make sure you have run the '01_pdf_to_markdown.py' test first.")
    else:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        print("--- Running chunking test ---")
        
        semantic_blocks = markdown_to_semantic_blocks(markdown_text)
        print(f"Step 1: {len(semantic_blocks)} semantic blocks have been created.")

        # Add a test medicine name
        test_medicine_name = "Test Medicine (Espidifen 600)"
        chunks = create_sentence_window_chunks(
            blocks=semantic_blocks, 
            source_file=INPUT_MD_NAME,
            medicine_name=test_medicine_name
        )
        print(f"Step 2: {len(chunks)} chunks have been created in total.")
        
        if chunks:
            print("\nExample of a generated chunk:")
            print(chunks[0].page_content)
            print("\nChunk metadata:")
            print(chunks[0].metadata)

        print("\nTest finished successfully.")